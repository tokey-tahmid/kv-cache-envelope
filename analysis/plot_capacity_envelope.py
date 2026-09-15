#!/usr/bin/env python3
"""The capacity-planning figure and the measured boundary map.

    python analysis/plot_capacity_envelope.py

Panel 1 answers the question the README is built around: given one 40 GB A100 and
a context length L, how many sequences can be served at once. The measured points
are the engine's own `kv_cache_max_concurrency`; the line is the analytical
prediction, pool tokens divided by L. They should lie on top of each other,
because the pool is measured and the division is exact.

Panel 2 is the boundary map: pool capacity against `gpu_memory_utilization`, with
the regions where the engine refuses to start shaded and labelled with *which*
refusal it is. There are four distinct ones and they are not interchangeable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_results import hardware_caption, load_raw, measured  # noqa: E402
from style import (  # noqa: E402
    STATUS_CRITICAL,
    SURFACE,
    TEXT_MUTED,
    TEXT_SECONDARY,
    apply_style,
    series_style,
)

REPO = Path(__file__).resolve().parents[1]
FIGS = REPO / "results" / "figures"

# Distinct colours for distinct boundaries. These are failures, so they share the
# reserved status hue family rather than a series slot.
BOUNDARY_COLOURS = {
    "kv_cache_too_small": "#b5651d",
    "budget_exceeds_free": STATUS_CRITICAL,
    "oom_no_cache_blocks": "#8c2f39",
    "oom_profiling": "#6a1b9a",
    "oom": "#8c2f39",
    "backend_unsupported": "#5c5c5c",
    "launch_failed": "#9e9e9e",
}


def panel_concurrency_vs_context(ax: plt.Axes, df: pd.DataFrame) -> str:
    d = df[df["sweep_name"] == "envelope-len"].dropna(
        subset=["max_model_len", "kv_cache_max_concurrency", "kv_cache_size_tokens"]
    )
    if d.empty:
        ax.text(0.5, 0.5, "not measured", ha="center", va="center", transform=ax.transAxes)
        return "not measured"
    g = (
        d.groupby("max_model_len")["kv_cache_max_concurrency"]
        .agg(["median", "min", "max", "size"])
        .reset_index()
        .sort_values("max_model_len")
    )
    st = series_style(0)
    ax.fill_between(g["max_model_len"], g["min"], g["max"], color=st["color"], alpha=0.15,
                    linewidth=0)
    ax.plot(g["max_model_len"], g["median"], label="measured (engine's own figure)",
            markerfacecolor=SURFACE, markeredgewidth=1.3, markeredgecolor=st["color"], **st)

    pool = float(d["kv_cache_size_tokens"].median())
    xs = np.array(sorted(d["max_model_len"].unique()), dtype=float)
    ax.plot(xs, pool / xs, color=TEXT_MUTED, linestyle="--", linewidth=1.1,
            label=f"analytical: {int(pool):,} pool tokens / L")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("context length L (tokens)")
    ax.set_ylabel("concurrent sequences of length L")
    ax.set_title("Capacity falls as 1/L, and the pool never moves")
    ax.legend(frameon=False, fontsize=8)

    spread = (g["max"] - g["min"]).max()
    tok = d["kv_cache_size_tokens"]
    return (
        f"pool is {int(tok.median()):,} tokens at every L from {int(g['max_model_len'].min())} "
        f"to {int(g['max_model_len'].max())} (distinct values: {tok.nunique()}); "
        f"max spread in concurrency across N={int(g['size'].median())} reps is {spread:g}"
    )


def panel_boundary_map(ax: plt.Axes, raw: pd.DataFrame) -> str:
    """Pool against utilisation, with each measured refusal region named."""
    d = raw[raw["max_model_len"] == 32768]
    d = d[d["model_id"] == "Qwen/Qwen2.5-7B-Instruct"]
    d = d[d["tensor_parallel_size"] == 1]
    if d.empty:
        ax.text(0.5, 0.5, "not measured", ha="center", va="center", transform=ax.transAxes)
        return "not measured"
    ok = d[(d["status"] == "ok") & (~d["warmup"].astype(bool))]
    bad = d[(d["status"] != "ok") & (~d["warmup"].astype(bool))]

    findings = []
    if not ok.empty:
        g = (
            ok.groupby("gpu_memory_utilization")["kv_cache_size_tokens"]
            .agg(["median", "min", "max", "size"])
            .reset_index()
            .sort_values("gpu_memory_utilization")
        )
        st = series_style(0)
        ax.fill_between(g["gpu_memory_utilization"], g["min"], g["max"], color=st["color"],
                        alpha=0.15, linewidth=0)
        ax.plot(g["gpu_memory_utilization"], g["median"], label="KV pool (tokens)",
                markerfacecolor=SURFACE, markeredgewidth=1.3, markeredgecolor=st["color"], **st)
        x = g["gpu_memory_utilization"].to_numpy(dtype=float)
        y = g["median"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        bpt = float(ok["kv_bytes_per_token"].median())
        usable = float(ok["log_total_device_memory_gib"].median())
        slope_gib = slope * bpt / 1024**3
        ax.plot(x, slope * x + intercept, color=TEXT_MUTED, linestyle="--", linewidth=1.0,
                label=f"linear fit, slope {slope_gib:.2f} GiB per unit gmu")
        findings.append(
            f"pool is linear in gmu with slope {slope_gib:.2f} GiB, against a measured usable "
            f"device memory of {usable:.2f} GiB ({abs(slope_gib - usable) / usable * 100:.2f}% apart)"
        )
        findings.append(
            f"zero-crossing of the fit at gmu {-intercept / slope:.3f}: the fixed overhead "
            f"consumes that fraction of the card before any cache exists"
        )

    seen: set[str] = set()
    for _, r in bad.iterrows():
        status = str(r["status"])
        colour = BOUNDARY_COLOURS.get(status, "#9e9e9e")
        ax.axvline(
            float(r["gpu_memory_utilization"]),
            color=colour,
            linewidth=6,
            alpha=0.22,
            label=status if status not in seen else None,
        )
        seen.add(status)
    for status in sorted(seen):
        sub = bad[bad["status"] == status]["gpu_memory_utilization"]
        findings.append(f"{status} at gmu {sorted(set(sub.round(3)))}")

    ax.set_xlabel("gpu_memory_utilization")
    ax.set_ylabel("KV pool capacity (tokens)")
    ax.set_title("Boundary map at max_model_len=32768")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    return "; ".join(findings)


def panel_dtype_and_blocks(ax: plt.Axes, df: pd.DataFrame) -> str:
    """fp8 against bf16, and the block-granularity rounding loss."""
    d = df[df["sweep_name"] == "envelope-block"].dropna(
        subset=["block_size_resolved", "kv_cache_size_tokens"]
    )
    if d.empty:
        ax.text(0.5, 0.5, "not measured", ha="center", va="center", transform=ax.transAxes)
        return "not measured"
    findings = []
    for i, (dtype, sub) in enumerate(sorted(d.groupby("cache_dtype_resolved"))):
        g = (
            sub.groupby("block_size_resolved")["kv_cache_size_tokens"]
            .agg(["median", "min", "max"])
            .reset_index()
            .sort_values("block_size_resolved")
        )
        st = series_style(i)
        ax.fill_between(g["block_size_resolved"], g["min"], g["max"], color=st["color"],
                        alpha=0.15, linewidth=0)
        ax.plot(g["block_size_resolved"], g["median"], label=f"kv dtype {dtype}",
                markerfacecolor=SURFACE, markeredgewidth=1.3, markeredgecolor=st["color"], **st)
        lo, hi = g["median"].min(), g["median"].max()
        findings.append(
            f"{dtype}: {int(hi):,} to {int(lo):,} tokens across block sizes "
            f"{sorted(g['block_size_resolved'].astype(int))}, a spread of "
            f"{(hi - lo) / hi * 100:.3f}%"
        )
    ratios = (
        d.groupby(["cache_dtype_resolved", "block_size_resolved"])["kv_cache_size_tokens"]
        .median()
        .unstack(0)
    )
    if {"auto", "fp8"} <= set(ratios.columns):
        r = (ratios["fp8"] / ratios["auto"]).dropna()
        findings.append(f"fp8/bf16 token ratio {r.min():.4f} to {r.max():.4f}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("block_size (tokens per KV block)")
    ax.set_ylabel("KV pool capacity (tokens)")
    ax.set_title("Granularity is free; dtype is not")
    ax.legend(frameon=False, fontsize=8)
    return "; ".join(findings)


def main() -> int:
    apply_style()
    FIGS.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    if raw.empty:
        print("no artifacts under results/raw/ yet; nothing to plot")
        return 0
    df = measured(raw)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    findings = [
        panel_concurrency_vs_context(axes[0], df),
        panel_boundary_map(axes[1], raw),
        panel_dtype_and_blocks(axes[2], df),
    ]
    fig.text(
        0.0,
        -0.04,
        " · ".join(
            [
                hardware_caption(df),
                "median with min/max over N=3 non-warmup repetitions",
                f"{len(df)} successful launches, "
                f"{int((raw['status'] != 'ok').sum())} recorded failures, all committed",
            ]
        ),
        ha="left",
        va="top",
        fontsize=7,
        color=TEXT_MUTED,
        wrap=True,
    )
    fig.suptitle("KV cache capacity envelope and its measured boundaries", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    out = FIGS / "capacity_envelope.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    for f in findings:
        print(f"  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
