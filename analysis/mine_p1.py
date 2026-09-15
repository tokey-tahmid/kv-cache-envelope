#!/usr/bin/env python3
"""Mine the sibling project's committed artifacts for KV cache facts.

    python analysis/mine_p1.py

``../llm-inference-bench/results/raw/`` holds 4,573 artifacts from eleven sweeps,
and every vLLM one of them carries ``vllm:cache_config_info`` in its stored
Prometheus exposition. That gauge's *labels* are the engine's resolved cache
configuration: `num_gpu_blocks`, `block_size`, `kv_cache_size_tokens`,
`kv_cache_max_concurrency`. Two of its sweeps were designed with this project in
mind: `paged-attention` (block_size x kv_cache_dtype) and `memory-utilisation`
(gpu_memory_utilization under deliberate oversubscription).

So the PagedAttention deliverable is already paid for in node-hours that were
spent. Mining it first is not a shortcut; re-measuring it would be the waste.

**This script never writes into the sibling project and never reads its results
as if they were ours.** Output lands in ``results/mined_p1/``, deliberately not
``results/raw/``, which is reserved for this project's own runs. The manifest
records every source filename and the git sha that produced them, so any number
traced back from a figure lands on a specific committed artifact in a specific
repository.

Two correctness details that decide whether the mined numbers mean anything:

* **Preemption counters are cumulative for a server's lifetime**, and one server
  serves several load phases. The per-phase figure is the difference between
  consecutive scrapes within a server group, ordered by time. Using the raw value
  would attribute every earlier phase's preemptions to the last one. The sibling
  project learned this the expensive way on its prefix-cache hit rate.
* **The first phase of a group has no predecessor** and gets None, not zero.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kv_cache_envelope.engine import parse_engine_log  # noqa: E402

P1 = REPO.parent / "llm-inference-bench"
P1_RAW = P1 / "results" / "raw"
DEST = REPO / "results" / "mined_p1"

_CACHE_INFO = re.compile(r'^vllm:cache_config_info\{(.*)\}$')
_LABEL = re.compile(r'(\w+)="([^"]*)"')


def decode_labels(blob: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in _LABEL.findall(blob):
        if value in ("None", ""):
            out[name] = None
        elif value in ("True", "False"):
            out[name] = value == "True"
        else:
            try:
                out[name] = int(value)
            except ValueError:
                try:
                    out[name] = float(value)
                except ValueError:
                    out[name] = value
    return out


def cache_config(raw: dict[str, Any]) -> dict[str, Any]:
    for key in raw:
        if m := _CACHE_INFO.match(key):
            return decode_labels(m.group(1))
    return {}


def metric(raw: dict[str, Any], name: str) -> float | None:
    """Sum matching series, or None. Never 0.0 for an absent metric."""
    vals = [v for k, v in raw.items() if k == name or k.startswith(name + "{")]
    return sum(vals) if vals else None


def p1_git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(P1), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def extract(path: Path) -> dict[str, Any] | None:
    d = json.loads(path.read_text())
    prov = d.get("provenance", {})
    if prov.get("backend_name") != "vllm":
        return None
    cfg = prov.get("full_config_dict", {})
    meas = d.get("measurements", {})
    tel = meas.get("engine_telemetry", {}) or {}
    raw = tel.get("raw_prometheus", {}) or {}
    cc = cache_config(raw)
    if not cc:
        return None

    engine = cfg.get("engine", {}) or {}
    load = cfg.get("load", {}) or {}
    workload = cfg.get("workload", {}) or {}
    return {
        "source_artifact": path.name,
        "sweep_name": cfg.get("sweep_name"),
        "group_id": cfg.get("group_id"),
        "phase_id": cfg.get("phase_id"),
        "phase_label": cfg.get("phase_label"),
        "utc_timestamp": prov.get("utc_timestamp"),
        "status": prov.get("status"),
        "warmup": prov.get("warmup"),
        "repetition_index": prov.get("repetition_index"),
        "model_id": prov.get("model_id"),
        "model_revision": prov.get("model_revision"),
        "backend_version": prov.get("backend_version"),
        "backend_image_digest": prov.get("backend_image_digest"),
        "gpu_model": prov.get("gpu_model"),
        "slurm_job_id": prov.get("slurm_job_id"),
        "hostname": prov.get("hostname"),
        # requested engine configuration
        "tensor_parallel_size": engine.get("tensor_parallel_size"),
        "max_model_len": engine.get("max_model_len"),
        "max_num_seqs": engine.get("max_num_seqs"),
        "gpu_memory_utilization": engine.get("gpu_memory_utilization"),
        "block_size_requested": engine.get("block_size"),
        "kv_cache_dtype": engine.get("kv_cache_dtype"),
        "enable_prefix_caching": engine.get("enable_prefix_caching"),
        # what the engine actually resolved (authoritative)
        "num_gpu_blocks": cc.get("num_gpu_blocks"),
        "block_size_resolved": cc.get("block_size"),
        "kv_cache_size_tokens": cc.get("kv_cache_size_tokens"),
        "kv_cache_max_concurrency": cc.get("kv_cache_max_concurrency"),
        "cache_dtype_resolved": cc.get("cache_dtype"),
        "gpu_memory_utilization_echoed": cc.get("gpu_memory_utilization"),
        # load and observed behaviour
        "concurrency": load.get("concurrency"),
        "load_mode": load.get("mode"),
        "input_len": (workload.get("input_len") or {}).get("mean"),
        "output_len": (workload.get("output_len") or {}).get("mean"),
        "num_requests": workload.get("num_requests"),
        "preemptions_cumulative": metric(raw, "vllm:num_preemptions_total"),
        "kv_cache_usage_frac_at_scrape": metric(raw, "vllm:kv_cache_usage_perc"),
        "kv_cache_memory_usage_gb": metric(raw, "vllm:kv_cache_memory_usage_gb"),
        "output_tokens_per_s": (meas.get("output_tokens_per_s")),
        "requests_ok": meas.get("requests_ok"),
        "requests_failed": meas.get("requests_failed"),
        "engine_log_path": meas.get("engine_log_path"),
    }


def add_preemption_deltas(records: list[dict[str, Any]]) -> None:
    """Per-phase preemptions, by differencing within a server group over time.

    One vLLM server serves many phases and the counter never resets, so the raw
    value at the end of a phase includes every phase before it in the same group.
    The first phase of a group has no predecessor and is left None rather than
    being called zero. A negative delta means the server restarted between
    scrapes; it is discarded rather than reported.
    """
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[(r.get("slurm_job_id"), r.get("group_id"))].append(r)

    for rows in groups.values():
        rows.sort(key=lambda r: (r.get("utc_timestamp") or "", r.get("source_artifact") or ""))
        prev: float | None = None
        for r in rows:
            cur = r.get("preemptions_cumulative")
            if cur is None:
                r["preemptions_phase"] = None
                r["preemptions_phase_note"] = "counter absent from this scrape"
                continue
            if prev is None:
                r["preemptions_phase"] = None
                r["preemptions_phase_note"] = "first phase in server group, no predecessor"
            elif cur < prev:
                r["preemptions_phase"] = None
                r["preemptions_phase_note"] = (
                    f"counter went backwards ({prev} -> {cur}); server restarted, discarded"
                )
            else:
                r["preemptions_phase"] = cur - prev
                r["preemptions_phase_note"] = None
            prev = cur


def attach_memory_breakdown(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Join each record to its engine startup log's memory decomposition.

    The log is the only place vLLM states weights-plus-non-torch, peak
    activation and CUDA graph memory, which is exactly the residual this project
    has to explain. Logs live on project storage rather than in either git repo,
    so a missing log is recorded as missing rather than filled in.
    """
    cache: dict[str, dict[str, Any]] = {}
    found = missing = 0
    for r in records:
        p = r.get("engine_log_path")
        if not p:
            r["memory_breakdown"] = None
            missing += 1
            continue
        if p not in cache:
            path = Path(p)
            if path.exists():
                cache[p] = parse_engine_log(path.read_text(errors="replace"))
            else:
                cache[p] = {}
        bd = cache[p]
        r["memory_breakdown"] = bd or None
        if bd:
            found += 1
        else:
            missing += 1
    return {"records_with_memory_breakdown": found, "records_without": missing}


def main() -> int:
    if not P1_RAW.is_dir():
        print(f"sibling project raw artifacts not found at {P1_RAW}", file=sys.stderr)
        return 2

    paths = sorted(P1_RAW.glob("*.json"))
    records: list[dict[str, Any]] = []
    skipped = 0
    for p in paths:
        try:
            rec = extract(p)
        except (OSError, ValueError, KeyError) as exc:
            print(f"  unreadable {p.name}: {exc}", file=sys.stderr)
            skipped += 1
            continue
        if rec is None:
            skipped += 1
            continue
        records.append(rec)

    add_preemption_deltas(records)
    log_stats = attach_memory_breakdown(records)

    DEST.mkdir(parents=True, exist_ok=True)
    out = DEST / "p1_kv_records.json"
    out.write_text(json.dumps(records, indent=1, sort_keys=True) + "\n")

    by_sweep: dict[str, int] = defaultdict(int)
    for r in records:
        by_sweep[str(r.get("sweep_name"))] += 1

    manifest = {
        "mined_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_repo": str(P1),
        "source_repo_git_sha": p1_git_sha(),
        "source_artifacts_scanned": len(paths),
        "records_extracted": len(records),
        "skipped_no_vllm_cache_config": skipped,
        "records_by_sweep": dict(sorted(by_sweep.items())),
        "engine_log_join": log_stats,
        "source_artifact_names": sorted(r["source_artifact"] for r in records),
        "note": (
            "Derived from the sibling project's committed raw artifacts. Kept out of "
            "results/raw/, which holds only this project's own runs. Every record names "
            "the artifact it came from."
        ),
    }
    (DEST / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"scanned {len(paths)} artifacts, extracted {len(records)} vLLM cache configs")
    for k, v in sorted(by_sweep.items()):
        print(f"  {v:5d}  {k}")
    print(f"  engine log join: {log_stats}")
    print(f"wrote {out} and {DEST / 'MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
