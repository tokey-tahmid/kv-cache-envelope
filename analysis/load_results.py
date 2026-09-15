"""Load this project's raw artifacts and the mined sibling-project records.

Two sources, kept deliberately separate everywhere they are used:

* ``results/raw/`` — this project's own engine launches. Append-only, one
  artifact per launch, failures included.
* ``results/mined_p1/`` — derived from the sibling project's committed
  artifacts. Never mixed into the former, and every record names the artifact it
  came from.

Figures label which source each series came from, because "measured here" and
"measured next door on a different driver" are different claims.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "results" / "raw"
MINED = REPO / "results" / "mined_p1" / "p1_kv_records.json"

GIB = 1024**3


def load_raw(raw_dir: Path = RAW) -> pd.DataFrame:
    """Flatten this project's artifacts into one row per engine launch."""
    rows: list[dict[str, Any]] = []
    for p in sorted(raw_dir.glob("*.json")):
        d = json.loads(p.read_text())
        prov = d["provenance"]
        meas = d["measurements"]
        cfg = prov["full_config_dict"]
        engine = cfg.get("engine", {})
        cc = meas.get("cache_config") or {}
        bd = meas.get("memory_breakdown") or {}
        pred = meas.get("prediction") or {}
        shape = meas.get("model_kv_shape") or {}
        row: dict[str, Any] = {
            "artifact": p.name,
            "sweep_name": cfg.get("sweep_name"),
            "config_index": cfg.get("config_index"),
            "status": prov["status"],
            "error": prov.get("error"),
            "warmup": prov.get("warmup"),
            "repetition_index": prov.get("repetition_index"),
            "utc_timestamp": prov.get("utc_timestamp"),
            "hostname": prov.get("hostname"),
            "slurm_job_id": prov.get("slurm_job_id"),
            "gpu_model": prov.get("gpu_model"),
            "gpu_memory_gb": prov.get("gpu_memory_gb"),
            "driver_version": prov.get("driver_version"),
            "cuda_version": prov.get("cuda_version"),
            "backend_version": prov.get("backend_version"),
            "backend_image_digest": prov.get("backend_image_digest"),
            "model_id": prov.get("model_id"),
            "model_revision": prov.get("model_revision"),
            "git_sha": prov.get("git_sha"),
            "startup_seconds": meas.get("startup_seconds"),
        }
        for k in (
            "max_model_len",
            "max_num_seqs",
            "gpu_memory_utilization",
            "block_size",
            "kv_cache_dtype",
            "tensor_parallel_size",
            "enforce_eager",
        ):
            row[k] = engine.get(k)
        row["num_gpu_blocks"] = cc.get("num_gpu_blocks")
        row["block_size_resolved"] = cc.get("block_size")
        row["kv_cache_size_tokens"] = cc.get("kv_cache_size_tokens")
        row["kv_cache_max_concurrency"] = cc.get("kv_cache_max_concurrency")
        row["cache_dtype_resolved"] = cc.get("cache_dtype")
        for k, v in bd.items():
            if k != "parsed_fields":
                row[f"log_{k}"] = v
        row["kv_bytes_per_token"] = pred.get("kv_bytes_per_token")
        row["kv_dtype_bytes"] = pred.get("kv_dtype_bytes")
        row["max_concurrent_at_max_model_len"] = pred.get("max_concurrent_at_max_model_len")
        row["attention_kind"] = shape.get("attention_kind")
        row["gqa_ratio"] = shape.get("gqa_ratio")
        row["num_key_value_heads"] = shape.get("num_key_value_heads")
        row["num_attention_heads"] = shape.get("num_attention_heads")
        row["num_hidden_layers"] = shape.get("num_hidden_layers")
        row["head_dim"] = shape.get("head_dim")
        occ = meas.get("occupancy")
        row["occupancy_phases"] = json.dumps(occ["phases"]) if occ else None
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Analytical pool size implied by the engine's own token count. If this
    # tracks the engine's reported GiB, the per-token formula is exact and the
    # whole residual lives in the memory budget rather than in the KV arithmetic.
    df["analytical_pool_gib"] = (
        df["kv_cache_size_tokens"] * df["kv_bytes_per_token"] / GIB
    )
    return df


def measured(df: pd.DataFrame) -> pd.DataFrame:
    """Successful, non-warmup launches. Failures stay in the frame for the
    boundary analysis; this view is for the fits and the residual."""
    if df.empty:
        return df
    return df[(df["status"] == "ok") & (~df["warmup"].astype(bool))]


def load_mined_p1(path: Path = MINED) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    records = json.loads(path.read_text())
    rows = []
    for r in records:
        row = {k: v for k, v in r.items() if k != "memory_breakdown"}
        bd = r.get("memory_breakdown") or {}
        for k, v in bd.items():
            if k != "parsed_fields":
                row[f"log_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def occupancy_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Explode the stored occupancy phases into one row per phase."""
    out: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        phases_json = r.get("occupancy_phases")
        # pandas turns the None on non-occupancy rows into NaN, which is a float
        # and is truthy, so a bare falsiness check lets it through and json.loads
        # raises. Check for null explicitly.
        if phases_json is None or pd.isna(phases_json) or not isinstance(phases_json, str):
            continue
        for ph in json.loads(phases_json):
            phase = ph.get("phase") or {}
            out.append(
                {
                    "artifact": r["artifact"],
                    "sweep_name": r["sweep_name"],
                    "warmup": r["warmup"],
                    "repetition_index": r["repetition_index"],
                    "max_model_len": r["max_model_len"],
                    "max_num_seqs": r["max_num_seqs"],
                    "gpu_memory_utilization": r["gpu_memory_utilization"],
                    "block_size": r["block_size"],
                    "kv_cache_dtype": r["kv_cache_dtype"],
                    "kv_cache_size_tokens": r["kv_cache_size_tokens"],
                    "kv_bytes_per_token": r["kv_bytes_per_token"],
                    "phase_status": ph.get("status"),
                    "phase_reason": ph.get("reason"),
                    "concurrency": phase.get("concurrency"),
                    "input_tokens": phase.get("input_tokens"),
                    "output_tokens": phase.get("output_tokens"),
                    "num_requests": phase.get("num_requests"),
                    "requests_ok": ph.get("requests_ok"),
                    "requests_failed": ph.get("requests_failed"),
                    "wall_seconds": ph.get("wall_seconds"),
                    "kv_usage_peak": ph.get("kv_cache_usage_frac_peak"),
                    "kv_usage_median": ph.get("kv_cache_usage_frac_median"),
                    "running_peak": ph.get("num_running_peak"),
                    "waiting_peak": ph.get("num_waiting_peak"),
                    "waiting_capacity_peak": ph.get("waiting_capacity_peak"),
                    "preemptions_delta": ph.get("preemptions_delta"),
                    "prompt_length_exact": ph.get("prompt_length_exact"),
                    "output_length_exact": ph.get("output_length_exact"),
                }
            )
    return pd.DataFrame(out)


def hardware_caption(df: pd.DataFrame) -> str:
    """One caption line naming the hardware and software that produced a frame."""
    if df.empty:
        return "no data"

    def one(col: str) -> str:
        vals = sorted({str(v) for v in df[col].dropna().unique()})
        return "/".join(vals) if vals else "unknown"

    return (
        f"{one('gpu_model')} (TP={one('tensor_parallel_size')}), "
        f"driver {one('driver_version')}, CUDA {one('cuda_version')}, "
        f"vLLM {one('backend_version')}, {one('model_id')} @ {one('model_revision')[:12]}"
    )
