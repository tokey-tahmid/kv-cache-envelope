"""YAML-driven sweep over engine configurations, writing one artifact per run.

A "run" here is one engine launch. That is the unit of the measurement, because
the quantity being measured — how much KV cache the engine could allocate — is
decided once, at init, and never changes afterwards.

Every launch produces an artifact, including the ones that fail. A
`kv_cache_too_small` refusal at max_model_len=32768 is not a gap in the sweep, it
is the capacity boundary the project exists to find, and an OOM at
gpu_memory_utilization=0.98 is the other one. Dropping either would leave the
envelope looking open-ended when it is not.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
import yaml

from .calculator import (
    ModelKVShape,
    kv_bytes,
    kv_bytes_per_token,
    max_concurrent_sequences,
    resolve_dtype_bytes,
    shape_from_config,
)
from .engine import EngineProbe, EngineSpec, jsonable
from .hf_config import load_local
from .provenance import build_provenance, write_artifact

log = structlog.get_logger(__name__)

# Axes expanded as a cartesian product. Anything not listed here is a scalar
# taken from the `engine:` block.
PRODUCT_AXES = (
    "max_model_len",
    "max_num_seqs",
    "gpu_memory_utilization",
    "block_size",
    "kv_cache_dtype",
    "enforce_eager",
    "tensor_parallel_size",
)


@dataclass
class SweepConfig:
    name: str
    repetitions: int
    warmup_repetitions: int
    model_id: str
    model_path: str
    model_revision: str
    engine_defaults: dict[str, Any]
    axes: dict[str, list[Any]]
    occupancy: dict[str, Any] | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> SweepConfig:
        data = yaml.safe_load(Path(path).read_text())
        model_id = data["model"]
        m = data["models"][model_id]
        return cls(
            name=data["name"],
            repetitions=int(data.get("repetitions", 3)),
            warmup_repetitions=int(data.get("warmup_repetitions", 1)),
            model_id=model_id,
            model_path=m["path"],
            model_revision=m["revision"],
            engine_defaults=dict(data.get("engine", {})),
            axes={k: list(v) for k, v in (data.get("axes") or {}).items()},
            occupancy=data.get("occupancy"),
        )

    def expand(self) -> list[dict[str, Any]]:
        keys = [k for k in PRODUCT_AXES if k in self.axes]
        if not keys:
            return [dict(self.engine_defaults)]
        combos = itertools.product(*(self.axes[k] for k in keys))
        out = []
        for combo in combos:
            cfg = dict(self.engine_defaults)
            cfg.update(dict(zip(keys, combo, strict=True)))
            out.append(cfg)
        return out


def load_shape(model_path: str, model_id: str) -> ModelKVShape:
    loaded = load_local(Path(model_path), model_id)
    return shape_from_config(
        loaded.config,
        model_id=model_id,
        revision=loaded.revision,
        config_path=loaded.path,
    )


def prediction_block(
    shape: ModelKVShape, spec: EngineSpec, measured_tokens: int | None
) -> dict[str, Any]:
    """What the analytical model says, alongside what the engine did.

    Computed at measurement time and stored in the artifact so that the
    predicted-vs-measured comparison is reproducible from the artifact alone,
    without re-deriving it from a config that may have moved.
    """
    try:
        dtype_bytes = resolve_dtype_bytes(spec.kv_cache_dtype, shape)
    except ValueError as exc:
        return {"error": str(exc)}
    per_token = kv_bytes_per_token(shape, dtype_bytes=dtype_bytes)
    out: dict[str, Any] = {
        "kv_bytes_per_token": per_token,
        "kv_dtype_bytes": dtype_bytes,
        "num_hidden_layers": shape.num_hidden_layers,
        "num_key_value_heads": shape.num_key_value_heads,
        "num_attention_heads": shape.num_attention_heads,
        "head_dim": shape.head_dim,
        "attention_kind": shape.attention_kind,
        "gqa_ratio": shape.gqa_ratio,
        "kv_heads_source": shape.kv_heads_source,
        "head_dim_source": shape.head_dim_source,
        "sliding_window": shape.sliding_window,
        "predicted_bytes_one_full_context": kv_bytes(
            shape,
            seq_len=spec.max_model_len,
            batch=1,
            dtype_bytes=dtype_bytes,
            block_size=spec.block_size,
        ),
    }
    if measured_tokens is not None:
        # The engine states its pool in tokens; converting with the analytical
        # per-token size gives the pool in bytes on the model's own terms, which
        # is what the residual is computed against.
        pool_bytes = measured_tokens * per_token
        out["measured_pool_bytes_from_tokens"] = pool_bytes
        out["max_concurrent_at_max_model_len"] = max_concurrent_sequences(
            shape,
            pool_bytes,
            seq_len=spec.max_model_len,
            dtype_bytes=dtype_bytes,
            block_size=spec.block_size,
        )
    return out


def run_sweep(
    cfg: SweepConfig,
    *,
    repo_root: Path,
    results_raw: Path,
    sif: str,
    data_dir: str,
    log_dir: Path,
    lockfile: dict[str, Any],
    slice_index: int | None = None,
    slice_count: int | None = None,
    dry_run: bool = False,
) -> int:
    shape = load_shape(cfg.model_path, cfg.model_id)
    if shape.revision != cfg.model_revision:
        raise ValueError(
            f"model revision mismatch: snapshot says {shape.revision!r}, config says "
            f"{cfg.model_revision!r}. Identifiers are copied from ground truth, never typed."
        )
    backend_version = lockfile["images"]["vllm"]["version"]
    image_digest = lockfile["images"]["vllm"]["digest"]

    combos = cfg.expand()
    if slice_index is not None and slice_count:
        combos = [c for i, c in enumerate(combos) if i % slice_count == slice_index]
    total_reps = cfg.warmup_repetitions + cfg.repetitions
    log.info(
        "sweep_plan",
        sweep=cfg.name,
        configurations=len(combos),
        reps_each=total_reps,
        launches=len(combos) * total_reps,
    )
    if dry_run:
        for c in combos:
            log.info("planned", **c)
        return 0

    written = 0
    for ci, combo in enumerate(combos):
        for rep in range(total_reps):
            warmup = rep < cfg.warmup_repetitions
            spec = EngineSpec(
                model_path=cfg.model_path,
                model_id=cfg.model_id,
                model_revision=cfg.model_revision,
                **{k: v for k, v in combo.items() if k in EngineSpec.__dataclass_fields__},
            )
            tag = f"{cfg.name}_c{ci}_r{rep}_{int(time.time())}"
            log.info(
                "launch",
                sweep=cfg.name,
                config_index=ci,
                rep=rep,
                warmup=warmup,
                **{k: getattr(spec, k) for k in ("max_model_len", "max_num_seqs",
                                                 "gpu_memory_utilization", "block_size",
                                                 "kv_cache_dtype")},
            )
            probe = EngineProbe(sif=sif, data_dir=data_dir, log_dir=log_dir)
            occupancy_summary: dict[str, Any] | None = None
            try:
                result = probe.start(spec, tag)
                if result.status == "ok" and cfg.occupancy:
                    occupancy_summary = _run_occupancy_phases(probe, cfg, spec)
            finally:
                probe.stop()

            measured_tokens = result.cache_config.get("kv_cache_size_tokens")
            if not isinstance(measured_tokens, int):
                measured_tokens = result.memory_breakdown.get("kv_cache_size_tokens_log")
            measured_tokens = measured_tokens if isinstance(measured_tokens, int) else None

            measurements = {
                "status": result.status,
                "error": result.error,
                "startup_seconds": result.startup_seconds,
                "engine_launch_command": result.launch_command,
                "engine_log_path": result.engine_log_path,
                "cache_config": jsonable(result.cache_config),
                "memory_breakdown": jsonable(result.memory_breakdown),
                "raw_prometheus": result.raw_prometheus,
                "nvidia_smi_used_mib": result.nvidia_smi_used_mib,
                "log_tail": result.log_tail,
                "prediction": jsonable(prediction_block(shape, spec, measured_tokens)),
                "model_kv_shape": jsonable(shape.as_dict()),
                "occupancy": jsonable(occupancy_summary) if occupancy_summary else None,
            }
            prov = build_provenance(
                repo_root=repo_root,
                backend_name="vllm",
                backend_version=backend_version,
                backend_image_digest=image_digest,
                model_id=cfg.model_id,
                model_revision=cfg.model_revision,
                config={
                    "sweep_name": cfg.name,
                    "config_index": ci,
                    "engine": spec.as_dict(),
                    "occupancy": cfg.occupancy,
                },
                repetition_index=rep,
                warmup=warmup,
                status=result.status,  # type: ignore[arg-type]
                error=result.error,
            )
            write_artifact(results_raw, prov, measurements)
            written += 1
    return written


def _run_occupancy_phases(
    probe: EngineProbe, cfg: SweepConfig, spec: EngineSpec
) -> dict[str, Any]:
    """Run the configured occupancy phases against a live engine."""
    import asyncio

    from .occupancy import run_occupancy

    assert cfg.occupancy is not None
    phases = cfg.occupancy.get("phases") or []
    out: dict[str, Any] = {"phases": []}
    for phase in phases:
        # A phase asking for more context than the engine was configured for is
        # not measurable; record the skip with its reason rather than silently
        # shrinking the request.
        in_tok = int(phase["input_tokens"])
        out_tok = int(phase["output_tokens"])
        if in_tok + out_tok > spec.max_model_len:
            out["phases"].append(
                {
                    "phase": phase,
                    "status": "skipped",
                    "reason": f"input+output {in_tok + out_tok} exceeds "
                    f"max_model_len {spec.max_model_len}",
                }
            )
            continue
        conc = int(phase["concurrency"])
        nreq = int(phase.get("num_requests", conc))
        res = asyncio.run(
            run_occupancy(
                probe.base_url,
                spec.model_id,
                concurrency=conc,
                num_requests=nreq,
                input_tokens=in_tok,
                output_tokens=out_tok,
                seed=int(phase.get("seed", 1234)),
                sample_interval_s=float(cfg.occupancy.get("sample_interval_s", 0.5)),
                request_timeout_s=float(cfg.occupancy.get("request_timeout_s", 1800.0)),
            )
        )
        summary = res.summarise()
        summary["phase"] = phase
        summary["status"] = "ok" if res.requests_failed == 0 else "partial"
        # Keep the raw sample series: peak occupancy and preemption deltas are
        # derived from it, and a surprise in the curve's shape must stay
        # recoverable without re-running the sweep.
        summary["samples"] = res.samples
        out["phases"].append(summary)
    return out
