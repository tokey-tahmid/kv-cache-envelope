"""Launch one vLLM engine in the pinned container and read its KV cache envelope.

The measurement this project needs is not a latency number, it is what the engine
*decided* about memory, and vLLM states that decision in two places:

* ``/metrics``, where ``vllm:cache_config_info`` carries `num_gpu_blocks`,
  `block_size`, `kv_cache_size_tokens` and `kv_cache_max_concurrency` as label
  values. These are the authoritative allocated-block numbers.
* the engine's own startup log, where ``gpu_worker`` prints the memory
  decomposition: total device memory, the utilisation budget, weights plus
  non-torch, peak activation, CUDA graph capture, and the KV pool that was left.
  Nothing else exposes that breakdown, and it *is* the predicted-vs-measured
  residual, measured by the engine rather than attributed by us.

Both are captured, and the whole log is kept, because the sibling project twice
recovered a metric bug after the fact purely because raw telemetry had been
stored next to the normalised keys.

Container rules that are not optional here (learned the hard way on this cluster,
see ../llm-inference-bench/docs/notes.md): Lustre is mounted **read-only** inside
Apptainer, and torch creates its inductor cache at *import* time, so every cache
variable must point at the node's own tmpfs or ``import vllm`` dies with Errno 30
before argparse runs.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
import structlog

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class EngineSpec:
    """One engine configuration. Everything here lands in the raw artifact."""

    model_path: str
    model_id: str
    model_revision: str
    max_model_len: int
    max_num_seqs: int = 256
    gpu_memory_utilization: float = 0.90
    block_size: int | None = None
    kv_cache_dtype: str = "auto"
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    enable_prefix_caching: bool = False
    enforce_eager: bool = False
    max_num_batched_tokens: int | None = None
    swap_space_gb: float | None = None
    seed: int = 0
    port: int = 8100

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProbeResult:
    status: str  # ok | kv_cache_too_small | oom | launch_failed | timeout
    error: str | None = None
    startup_seconds: float | None = None
    launch_command: str = ""
    engine_log_path: str | None = None
    cache_config: dict[str, Any] = field(default_factory=dict)
    memory_breakdown: dict[str, Any] = field(default_factory=dict)
    raw_prometheus: dict[str, float] = field(default_factory=dict)
    nvidia_smi_used_mib: list[int] = field(default_factory=list)
    log_tail: str | None = None
    gpu_release: dict[str, Any] = field(default_factory=dict)


# --- launching ---------------------------------------------------------------


def container_argv(sif: str, spec: EngineSpec, data_dir: str) -> list[str]:
    """`apptainer exec` argv for one vLLM server.

    Every cache root points inside the container's `/tmp` (node tmpfs). Pointing
    any of them at the bound Lustre path makes torch fail at import, and pointing
    them at a *shared* location would let one launch warm the next one's compile
    cache, which is an invisible confound in a startup-sensitive sweep.
    """
    env = {
        "HF_HOME": f"{data_dir}/hf_cache",
        "HF_HUB_CACHE": f"{data_dir}/hf_cache/hub",
        "TMPDIR": "/tmp",
        "VLLM_CACHE_ROOT": "/tmp/vllm_cache",
        "TRITON_CACHE_DIR": "/tmp/triton_cache",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor_cache",
        "TORCH_HOME": "/tmp/torch_home",
        "XDG_CACHE_HOME": "/tmp/xdg_cache",
        "OUTLINES_CACHE_DIR": "/tmp/outlines_cache",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_SERVER_DEV_MODE": "1",
    }
    argv = ["apptainer", "exec", "--nv", "--cleanenv", "--bind", f"{data_dir}:{data_dir}:rw"]
    for k, v in env.items():
        argv += ["--env", f"{k}={v}"]
    argv += [sif, *server_argv(spec)]
    return argv


def server_argv(spec: EngineSpec) -> list[str]:
    argv = [
        "vllm",
        "serve",
        spec.model_path,
        "--served-model-name",
        spec.model_id,
        "--host",
        "127.0.0.1",
        "--port",
        str(spec.port),
        "--tensor-parallel-size",
        str(spec.tensor_parallel_size),
        "--dtype",
        spec.dtype,
        "--gpu-memory-utilization",
        f"{spec.gpu_memory_utilization}",
        "--max-model-len",
        str(spec.max_model_len),
        "--max-num-seqs",
        str(spec.max_num_seqs),
        "--kv-cache-dtype",
        spec.kv_cache_dtype,
        "--seed",
        str(spec.seed),
        "--load-format",
        "auto",
        "--no-enable-log-requests",
    ]
    if spec.block_size is not None:
        argv += ["--block-size", str(spec.block_size)]
    if spec.max_num_batched_tokens is not None:
        argv += ["--max-num-batched-tokens", str(spec.max_num_batched_tokens)]
    if spec.swap_space_gb is not None:
        argv += ["--swap-space", f"{spec.swap_space_gb}"]
    if spec.enforce_eager:
        argv += ["--enforce-eager"]
    # Explicit in both directions: vLLM V1 defaults prefix caching ON, so an
    # "off" arm built by omission would silently be an "on" arm.
    argv += (
        ["--enable-prefix-caching"]
        if spec.enable_prefix_caching
        else ["--no-enable-prefix-caching"]
    )
    return argv


# --- log parsing -------------------------------------------------------------

# vLLM 0.27.1 log shapes, confirmed against stored engine logs from the sibling
# project's paged-attention sweep rather than guessed. Every field is optional:
# a line that is not found yields an ABSENT key, never a zero. A missing
# breakdown is a different fact from a breakdown of zero.
_RE_AVAILABLE_KV = re.compile(r"Available KV cache memory:\s*([\d.]+)\s*GiB")
_RE_KV_TOKENS = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
_RE_MAX_CONC = re.compile(
    r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x"
)
_RE_GRAPH = re.compile(r"Graph capturing finished in\s*(\d+)\s*secs, took\s*([\d.]+)\s*GiB")
_RE_BREAKDOWN = re.compile(
    r"Free memory on device \(([\d.]+)/([\d.]+) GiB\) on startup\. "
    r"Desired GPU memory utilization is \(([\d.]+), ([\d.]+) GiB\)\. "
    r"Actual usage is ([\d.]+) GiB for consumed memory \(weights \+ non-torch\), "
    r"([\d.]+) GiB for peak activation, and ([\d.]+) GiB for CUDAGraph memory\."
)
_RE_KV_IN_USE = re.compile(r"Current kv cache memory in use is\s*([\d.]+)\s*GiB")
_RE_INIT_TOOK = re.compile(r"init engine \(profile, create kv cache, warmup model\) took\s*([\d.]+)\s*s")
_RE_MODEL_LOAD = re.compile(r"Model loading took\s*([\d.]+)\s*GiB and\s*([\d.]+)\s*seconds")

# Failure shapes worth distinguishing. Five were observed in this project's own
# sweeps, and every pattern below is copied from the engine log it came out of
# rather than written from memory. The first version of this classifier had
# guessed wordings for two of them, matched neither, and filed three genuinely
# different boundaries under one "launch_failed" bucket. That is the whole
# result thrown away: the failure point is the headline number here.
#
#   kv_cache_too_small   the pool cannot hold ONE sequence of max_model_len.
#                        vLLM refuses to start and states the longest model
#                        length the available memory *would* have supported.
#   oom_no_cache_blocks  profiling left nothing at all for the cache.
#   oom_profiling        CUDA ran out during profiling or graph capture.
#   budget_exceeds_free  the requested budget is larger than the memory free
#                        before anything was allocated; refused pre-flight.
#   backend_unsupported  the attention backend rejects the configuration itself
#                        (block_size=8 here). Not a memory boundary at all, and
#                        must never be plotted as one.
_RE_KV_TOO_SMALL = re.compile(
    r"To serve at least one request with the model's max seq len \(([\d,]+)\), "
    r"\(?([\d.]+) GiB KV cache is needed, which is larger than the available KV cache "
    r"memory \(([\d.]+) GiB\)\.",
    re.IGNORECASE,
)
_RE_EST_MAX_LEN = re.compile(r"estimated maximum model length is ([\d,]+)")
# Older wording, kept so an image pin in either direction still classifies.
_RE_KV_TOO_SMALL_ALT = re.compile(
    r"max seq len \(([\d,]+)\) is larger than the maximum number of tokens "
    r"that can be stored in KV cache \(([\d,]+)\)",
    re.IGNORECASE,
)
_RE_BUDGET_EXCEEDS_FREE = re.compile(
    r"Free memory on device \S+ \(([\d.]+)/([\d.]+) GiB\) on startup is less than "
    r"desired GPU memory utilization \(([\d.]+), ([\d.]+) GiB\)",
    re.IGNORECASE,
)
_RE_NO_CACHE_BLOCKS = re.compile(r"No available memory for the cache blocks", re.IGNORECASE)
_RE_OOM = re.compile(r"CUDA out of memory|torch\.OutOfMemoryError|CUDA error: out of memory")
_RE_NO_BACKEND = re.compile(
    r"No valid attention backend found for \w+ with (AttentionSelectorConfig\([^)]*\))"
)


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def parse_engine_log(text: str) -> dict[str, Any]:
    """Pull the memory decomposition out of a vLLM startup log.

    Returns only the keys it actually found, plus `parsed_fields` naming them, so
    an analysis can tell "the engine did not print this" from "the engine printed
    zero".
    """
    out: dict[str, Any] = {}
    if m := _RE_AVAILABLE_KV.search(text):
        out["available_kv_cache_gib"] = float(m.group(1))
    if m := _RE_KV_TOKENS.search(text):
        out["kv_cache_size_tokens_log"] = int(_num(m.group(1)))
    if m := _RE_MAX_CONC.search(text):
        out["max_concurrency_tokens_per_request"] = int(_num(m.group(1)))
        out["max_concurrency_x_log"] = float(m.group(2))
    if m := _RE_GRAPH.search(text):
        out["cudagraph_capture_seconds"] = int(m.group(1))
        out["cudagraph_memory_gib"] = float(m.group(2))
    if m := _RE_BREAKDOWN.search(text):
        out["free_memory_on_startup_gib"] = float(m.group(1))
        out["total_device_memory_gib"] = float(m.group(2))
        out["gpu_memory_utilization_echoed"] = float(m.group(3))
        out["memory_budget_gib"] = float(m.group(4))
        out["weights_plus_non_torch_gib"] = float(m.group(5))
        out["peak_activation_gib"] = float(m.group(6))
        out["cudagraph_gib"] = float(m.group(7))
    if m := _RE_KV_IN_USE.search(text):
        out["kv_cache_in_use_gib"] = float(m.group(1))
    if m := _RE_INIT_TOOK.search(text):
        out["init_engine_seconds"] = float(m.group(1))
    if m := _RE_MODEL_LOAD.search(text):
        out["model_loading_gib"] = float(m.group(1))
        out["model_loading_seconds"] = float(m.group(2))
    out["parsed_fields"] = sorted(k for k in out if k != "parsed_fields")
    return out


def classify_failure(text: str) -> tuple[str, str | None]:
    """Map a failed launch onto a named boundary.

    A KV-cache-too-small refusal is a *measured capacity limit*: the engine
    computed that the pool it could allocate cannot hold one sequence of the
    requested length. It is recorded with the two numbers it printed, because
    those are the boundary.
    """
    if m := _RE_KV_TOO_SMALL.search(text):
        est = _RE_EST_MAX_LEN.search(text)
        msg = (
            f"max_model_len={int(_num(m.group(1)))} needs {m.group(2)} GiB of KV cache but only "
            f"{m.group(3)} GiB is available"
        )
        if est:
            msg += f"; engine's own estimated maximum model length {int(_num(est.group(1)))}"
        return "kv_cache_too_small", msg
    if m := _RE_KV_TOO_SMALL_ALT.search(text):
        return (
            "kv_cache_too_small",
            f"max_model_len={int(_num(m.group(1)))} exceeds KV cache capacity "
            f"{int(_num(m.group(2)))} tokens",
        )
    if m := _RE_BUDGET_EXCEEDS_FREE.search(text):
        return (
            "budget_exceeds_free",
            f"requested gpu_memory_utilization {m.group(3)} = {m.group(4)} GiB but only "
            f"{m.group(1)} GiB of {m.group(2)} GiB was free before allocation",
        )
    if _RE_NO_CACHE_BLOCKS.search(text):
        return "oom_no_cache_blocks", "No available memory for the cache blocks"
    if m := _RE_NO_BACKEND.search(text):
        return "backend_unsupported", f"no attention backend accepts {m.group(1)[:400]}"
    if _RE_OOM.search(text):
        line = next(
            (ln.strip() for ln in text.splitlines() if _RE_OOM.search(ln)), "CUDA out of memory"
        )
        return "oom_profiling", line[:600]
    tail = "\n".join(text.strip().splitlines()[-12:])
    return "launch_failed", tail[:1200]


def parse_prometheus(text: str) -> dict[str, float]:
    """Minimal Prometheus text-format parser; label sets are kept in the key."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.rpartition(" ")
        if not key:
            continue
        try:
            out[key] = float(value)
        except ValueError:
            continue
    return out


_RE_CACHE_INFO = re.compile(r'^vllm:cache_config_info\{(.*)\}$')


def cache_config_from_prometheus(raw: dict[str, float]) -> dict[str, Any]:
    """Decode ``vllm:cache_config_info``'s label set into typed values.

    vLLM exposes the engine's resolved cache configuration as *labels* on a gauge
    whose value is always 1.0, so the payload is in the key. This is the
    authoritative statement of how many blocks were actually allocated.
    """
    for key in raw:
        m = _RE_CACHE_INFO.match(key)
        if not m:
            continue
        labels: dict[str, Any] = {}
        for part in re.findall(r'(\w+)="([^"]*)"', m.group(1)):
            name, value = part
            if value in ("None", ""):
                labels[name] = None
            elif value in ("True", "False"):
                labels[name] = value == "True"
            else:
                try:
                    labels[name] = int(value)
                except ValueError:
                    try:
                        labels[name] = float(value)
                    except ValueError:
                        labels[name] = value
        return labels
    return {}


def nvidia_smi_used_mib() -> list[int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(float(x)) for x in out.split() if x.strip()]


# --- the probe ---------------------------------------------------------------


class EngineProbe:
    """Launch, wait, scrape, tear down. Every outcome is a recorded outcome."""

    def __init__(
        self,
        sif: str,
        data_dir: str,
        log_dir: Path,
        startup_timeout_s: float = 900.0,
    ) -> None:
        self.sif = sif
        self.data_dir = data_dir
        self.log_dir = Path(log_dir)
        self.startup_timeout_s = startup_timeout_s
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.process: subprocess.Popen[bytes] | None = None
        self.log_path: Path | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    def start(self, spec: EngineSpec, tag: str) -> ProbeResult:
        self._port = spec.port
        argv = container_argv(self.sif, spec, self.data_dir)
        cmd = shlex.join(argv)
        self.log_path = self.log_dir / f"kve_engine_{tag}.log"
        result = ProbeResult(
            status="launch_failed", launch_command=cmd, engine_log_path=str(self.log_path)
        )

        t0 = time.perf_counter()
        with self.log_path.open("wb") as fh:
            self.process = subprocess.Popen(
                argv, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True
            )
            deadline = t0 + self.startup_timeout_s
            healthy = False
            while time.perf_counter() < deadline:
                if self.process.poll() is not None:
                    break
                try:
                    r = httpx.get(f"{self.base_url}/health", timeout=5.0)
                    if r.status_code == 200:
                        healthy = True
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(2.0)

        text = self.log_path.read_text(errors="replace")
        result.memory_breakdown = parse_engine_log(text)

        if not healthy:
            self.stop()
            if self.process is not None and self.process.poll() is None:
                result.status, result.error = "timeout", (
                    f"engine did not become healthy within {self.startup_timeout_s:.0f}s"
                )
            else:
                result.status, result.error = classify_failure(text)
            result.log_tail = "\n".join(text.splitlines()[-40:])
            return result

        result.status = "ok"
        result.startup_seconds = time.perf_counter() - t0
        result.nvidia_smi_used_mib = nvidia_smi_used_mib()
        try:
            r = httpx.get(f"{self.base_url}/metrics", timeout=30.0)
            r.raise_for_status()
            result.raw_prometheus = parse_prometheus(r.text)
            result.cache_config = cache_config_from_prometheus(result.raw_prometheus)
        except httpx.HTTPError as exc:
            # The engine is up but did not answer /metrics: record that, do not
            # invent a cache configuration.
            result.error = f"metrics scrape failed: {exc}"
        return result

    def scrape(self) -> dict[str, float]:
        try:
            r = httpx.get(f"{self.base_url}/metrics", timeout=30.0)
            r.raise_for_status()
        except httpx.HTTPError:
            return {}
        return parse_prometheus(r.text)

    def wait_for_gpu_memory_release(
        self, threshold_mib: int = 2048, timeout_s: float = 180.0
    ) -> dict[str, Any]:
        """Block until the GPUs are actually idle again, and report what happened.

        Killing the process group returns before the driver has finished
        reclaiming device memory, and the 32B TP=2 sweep (job 5213931) found out
        the expensive way: the first launch succeeded, then every repetition
        after it failed with `budget_exceeds_free`, reporting only 2.78 GiB free
        of 39.67 on a card that should have been empty. The previous engine's
        31 GiB of weights were still resident.

        This did not bite the 7B TP=1 sweeps because 14.58 GiB of weights leaves
        enough slack that a late reclaim still fits under the budget. That is
        precisely the kind of bug that hides until the configuration that cannot
        absorb it, so the wait is unconditional rather than applied only to the
        large model.
        """
        deadline = time.perf_counter() + timeout_s
        last: list[int] = []
        while time.perf_counter() < deadline:
            last = nvidia_smi_used_mib()
            if not last or max(last) <= threshold_mib:
                return {
                    "released": True,
                    "waited_s": round(timeout_s - (deadline - time.perf_counter()), 2),
                    "used_mib": last,
                }
            time.sleep(2.0)
        log.warning("gpu_memory_not_released", used_mib=last, timeout_s=timeout_s)
        return {"released": False, "waited_s": timeout_s, "used_mib": last}

    def stop(self) -> None:
        proc = self.process
        if proc is None:
            return
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                proc.terminate()
            try:
                proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    proc.kill()
                proc.wait(timeout=60)
        self.process = None

    def __enter__(self) -> EngineProbe:
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def sum_metric(raw: dict[str, float], name: str) -> float | None:
    """Sum every series whose metric name matches, or None if there is none.

    None, never 0.0. The sibling project spent a debugging cycle on a prefix-cache
    hit rate that read as a measured zero when the counter had simply been
    renamed; an absent metric must stay visibly absent.
    """
    vals = [v for k, v in raw.items() if k == name or k.startswith(name + "{")]
    return sum(vals) if vals else None


def jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))
