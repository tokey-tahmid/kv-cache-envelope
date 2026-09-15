#!/usr/bin/env python3
"""Generate every table in the writeup from ground truth.

    python analysis/make_tables.py

Tables land in ``results/tables/`` as markdown and are included in the README and
notes by reference. Nothing is transcribed: if a number appears in prose it was
produced here, from either a committed config bundle or a committed raw artifact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kv_cache_envelope.calculator import (  # noqa: E402
    UnsupportedArchitecture,
    kv_bytes,
    kv_bytes_per_token,
    max_concurrent_sequences,
    shape_from_config,
)
from kv_cache_envelope.hf_config import read_all_bundles  # noqa: E402
from load_results import GIB, load_raw, measured  # noqa: E402

TABLES = REPO / "results" / "tables"
BUNDLES = REPO / "configs" / "model_configs"


def _md(df: pd.DataFrame) -> str:
    """Render a DataFrame as a markdown table.

    Hand-rolled rather than `DataFrame.to_markdown`, which needs `tabulate`.
    That package is not in the shared harness venv, and the venv is shared with
    the sibling project: adding a dependency to it to save fifteen lines here
    would change the environment every one of that project's artifacts was
    measured in. Fifteen lines is cheaper than that.
    """
    if df.empty:
        return "_(no rows)_"
    cols = [str(c) for c in df.columns]

    def cell(v: object) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        if isinstance(v, float):
            return f"{v:g}"
        return str(v).replace("|", "\\|").replace("\n", " ")

    rows = [[cell(v) for v in rec] for rec in df.itertuples(index=False, name=None)]
    widths = [
        max(len(cols[i]), *(len(r[i]) for r in rows)) if rows else len(cols[i])
        for i in range(len(cols))
    ]
    out = ["| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths, strict=True)) + " |"]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        out.append("| " + " | ".join(v.ljust(w) for v, w in zip(r, widths, strict=True)) + " |")
    return "\n".join(out)


def table_calculator_validation() -> str:
    """Every fetched config, its attention shape, and the cost of the GQA mistake."""
    rows = []
    refused = []
    for b in read_all_bundles(BUNDLES):
        try:
            s = shape_from_config(b.config, model_id=b.model_id, revision=b.revision)
        except UnsupportedArchitecture as exc:
            refused.append({"model": b.model_id, "revision": b.revision, "refused because": str(exc)})
            continue
        try:
            per_token = kv_bytes_per_token(s, kv_cache_dtype="auto")
        except ValueError as exc:
            refused.append({"model": b.model_id, "revision": b.revision, "refused because": str(exc)})
            continue
        naive = 2 * s.num_hidden_layers * s.num_attention_heads * s.head_dim * 2
        rows.append(
            {
                "model": s.model_id,
                "revision": (s.revision or "")[:12],
                "kind": s.attention_kind,
                "layers": s.num_hidden_layers,
                "q heads": s.num_attention_heads,
                "kv heads": s.num_key_value_heads,
                "head_dim": s.head_dim,
                "GQA ratio": f"{s.gqa_ratio:g}x",
                "kv KiB/token": round(per_token / 1024, 2),
                "GiB @ 8k ctx": round(
                    kv_bytes(s, seq_len=8192, kv_cache_dtype="auto") / GIB, 3
                ),
                "kv heads from": s.kv_heads_source,
                "head_dim from": s.head_dim_source,
                "if q heads used": f"{naive / per_token:g}x too large",
            }
        )
    df = pd.DataFrame(rows).sort_values(["kind", "model"])
    out = [
        "# Calculator validation against real model configs",
        "",
        "Every row is computed by `kv_cache_envelope.calculator` from the committed",
        "`config.json` bundle named by its revision. The last column is the error the",
        "`num_attention_heads` mistake would have produced for that model.",
        "",
        _md(df),
        "",
        f"**{len(df)} models, {df['GQA ratio'].nunique()} distinct GQA ratios, "
        f"{df['kind'].nunique()} attention kinds.**",
        "",
    ]
    if refused:
        out += [
            "## Configs the calculator refuses to answer for",
            "",
            "Refusing beats returning a plausible wrong number. These architectures do not",
            "cache per-head K and V, so `2*L*H_kv*D` does not describe them.",
            "",
            _md(pd.DataFrame(refused)),
            "",
        ]
    return "\n".join(out)


def table_measured_envelope(df: pd.DataFrame) -> str:
    if df.empty:
        return "# Measured envelope\n\nNot measured.\n"
    d = df.dropna(subset=["kv_cache_size_tokens"]).copy()
    if d.empty:
        return "# Measured envelope\n\nNot measured.\n"
    d["analytical GiB"] = d["kv_cache_size_tokens"] * d["kv_bytes_per_token"] / GIB
    g = (
        d.groupby(
            [
                "sweep_name",
                "max_model_len",
                "max_num_seqs",
                "gpu_memory_utilization",
                "block_size_resolved",
                "cache_dtype_resolved",
                "tensor_parallel_size",
                "model_id",
            ],
            dropna=False,
        )
        .agg(
            n=("kv_cache_size_tokens", "size"),
            tokens_median=("kv_cache_size_tokens", "median"),
            tokens_min=("kv_cache_size_tokens", "min"),
            tokens_max=("kv_cache_size_tokens", "max"),
            blocks=("num_gpu_blocks", "median"),
            pool_gib=("log_kv_cache_in_use_gib", "median"),
            analytical_gib=("analytical GiB", "median"),
            weights_gib=("log_weights_plus_non_torch_gib", "median"),
            activation_gib=("log_peak_activation_gib", "median"),
            cudagraph_gib=("log_cudagraph_gib", "median"),
            usable_gib=("log_total_device_memory_gib", "median"),
            max_conc=("kv_cache_max_concurrency", "median"),
        )
        .reset_index()
    )
    g["residual_gib"] = g["pool_gib"] - g["analytical_gib"]
    g["residual_pct"] = g["residual_gib"] / g["pool_gib"] * 100
    for c in ("analytical_gib", "residual_gib"):
        g[c] = g[c].round(3)
    g["residual_pct"] = g["residual_pct"].round(3)
    return "\n".join(
        [
            "# Measured KV cache envelope",
            "",
            "One row per configuration, median over non-warmup repetitions with min/max on the",
            "token count. `analytical_gib` is the pool implied by the engine's own token count",
            "and the analytical per-token size; `residual_gib` is how far that sits from the",
            "pool size the engine reports.",
            "",
            _md(g),
            "",
        ]
    )


def table_boundaries(raw: pd.DataFrame) -> str:
    if raw.empty:
        return "# Measured boundaries\n\nNot measured.\n"
    f = raw[raw["status"] != "ok"].copy()
    if f.empty:
        return "# Measured boundaries\n\nNo failures recorded.\n"
    g = (
        f.groupby(
            [
                "sweep_name",
                "model_id",
                "max_model_len",
                "gpu_memory_utilization",
                "block_size",
                "kv_cache_dtype",
                "status",
            ],
            dropna=False,
        )
        .agg(n=("status", "size"), example_error=("error", "first"))
        .reset_index()
    )
    g["example_error"] = g["example_error"].astype(str).str.replace("\n", " ").str.slice(0, 180)
    return "\n".join(
        [
            "# Measured failure boundaries",
            "",
            "Every failed launch is a recorded artifact. These are not gaps in the sweep; the",
            "failure point is the capacity limit the project exists to find.",
            "",
            _md(g),
            "",
        ]
    )


def table_capacity_planning(df: pd.DataFrame) -> str:
    """The README headline: given this card and context L, how many sequences?"""
    if df.empty:
        return "# Capacity planning\n\nNot measured.\n"
    # Anchor on the measured pool at the utilisation a deployment would actually
    # use, for each model and KV dtype measured.
    d = df.dropna(subset=["kv_cache_size_tokens", "kv_bytes_per_token"])
    d = d[d["gpu_memory_utilization"] == 0.90]
    if d.empty:
        return "# Capacity planning\n\nNot measured at gpu_memory_utilization=0.90.\n"

    bundles = {b.model_id: b for b in read_all_bundles(BUNDLES)}
    rows = []
    keys = ["model_id", "cache_dtype_resolved", "tensor_parallel_size"]
    for (model_id, dtype, tp), sub in d.groupby(keys, dropna=False):
        pool_tokens = float(sub["kv_cache_size_tokens"].median())
        per_token = float(sub["kv_bytes_per_token"].median())
        b = bundles.get(model_id)
        if b is None:
            continue
        shape = shape_from_config(b.config, model_id=model_id, revision=b.revision)
        pool_bytes = int(pool_tokens * per_token)
        row: dict[str, object] = {
            "model": model_id,
            "TP": int(tp) if pd.notna(tp) else None,
            "kv dtype": dtype,
            "kv KiB/token": round(per_token / 1024, 2),
            "measured pool (tokens)": int(pool_tokens),
            "measured pool (GiB)": round(pool_bytes / GIB, 2),
        }
        for ctx in (1024, 2048, 4096, 8192, 16384, 32768):
            if shape.max_position_embeddings and ctx > shape.max_position_embeddings:
                row[f"seqs @ {ctx}"] = "beyond model max"
                continue
            row[f"seqs @ {ctx}"] = max_concurrent_sequences(
                shape, pool_bytes, seq_len=ctx, dtype_bytes=int(per_token
                    // (2 * shape.num_hidden_layers * shape.num_key_value_heads * shape.head_dim)),
                block_size=16,
            )
        rows.append(row)
    if not rows:
        return "# Capacity planning\n\nNot measured.\n"
    return "\n".join(
        [
            "# Capacity planning: sequences servable at a given context length",
            "",
            "Concurrent sequences that fit in the **measured** KV pool at",
            "`gpu_memory_utilization=0.90`, each holding a full context of the stated length.",
            "The pool size is measured; the division is the analytical per-token size, which",
            "the residual analysis shows to be exact. Floor, not round: a partially resident",
            "sequence is a preemption waiting to happen, not a servable sequence.",
            "",
            _md(pd.DataFrame(rows)),
            "",
        ]
    )


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    df = measured(raw) if not raw.empty else pd.DataFrame()

    outputs = {
        "calculator_validation.md": table_calculator_validation(),
        "measured_envelope.md": table_measured_envelope(df),
        "boundaries.md": table_boundaries(raw),
        "capacity_planning.md": table_capacity_planning(df),
    }
    for name, text in outputs.items():
        (TABLES / name).write_text(text + "\n")
        print(f"wrote {TABLES / name} ({len(text.splitlines())} lines)")

    summary = {
        "raw_artifacts": int(len(raw)),
        "successful_launches": int((raw["status"] == "ok").sum()) if not raw.empty else 0,
        "recorded_failures": int((raw["status"] != "ok").sum()) if not raw.empty else 0,
        "statuses": (
            {str(k): int(v) for k, v in raw["status"].value_counts().items()}
            if not raw.empty
            else {}
        ),
        "sweeps": (
            {str(k): int(v) for k, v in raw["sweep_name"].value_counts().items()}
            if not raw.empty
            else {}
        ),
    }
    (TABLES / "inventory.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
