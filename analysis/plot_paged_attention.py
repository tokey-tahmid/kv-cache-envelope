#!/usr/bin/env python3
"""PagedAttention behaviour: block granularity, KV dtype, utilisation, preemption.

    python analysis/plot_paged_attention.py

Draws on two datasets and never merges them into one series:

* ``results/mined_p1/`` — the sibling project's `paged-attention` and
  `memory-utilisation` sweeps, which already swept block_size x kv_cache_dtype
  and gpu_memory_utilization *under load*, on driver 595.71.05.
* ``results/raw/`` — this project's own launches, which widen the block axis and
  measure the pool directly rather than inferring it.

They are plotted as distinct series because they were taken on different driver
versions and are different claims.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_results import hardware_caption, load_mined_p1, load_raw, measured  # noqa: E402
from style import (  # noqa: E402
    SURFACE,
    TEXT_MUTED,
    TEXT_SECONDARY,
    apply_style,
    series_style,
)

REPO = Path(__file__).resolve().parents[1]
FIGS = REPO / "results" / "figures"


def _agg(df: pd.DataFrame, by: list[str], col: str) -> pd.DataFrame:
    g = df.dropna(subset=[col]).groupby(by, dropna=False)[col]
    return pd.DataFrame(
        {"median": g.median(), "min": g.min(), "max": g.max(), "n": g.size()}
    ).reset_index()


def panel_block_size(ax: plt.Axes, p2: pd.DataFrame, p1: pd.DataFrame) -> str:
    """Pool capacity in tokens against block granularity.

    The interesting prediction is a null one: the pool is sized in *bytes* by
    memory profiling and only then divided into blocks, so changing the block
    size should move `num_gpu_blocks` inversely and leave the token capacity
    almost unchanged, the "almost" being one partial block of rounding.
    """
    findings = []
    idx = 0
    for src, df, marker_note in (("this project", p2, ""), ("mined from P1", p1, " (P1)")):
        if df.empty or "block_size_resolved" not in df:
            continue
        d = df.dropna(subset=["block_size_resolved", "kv_cache_size_tokens"])
        if d.empty:
            continue
        for dtype, sub in d.groupby("cache_dtype_resolved"):
            a = _agg(sub, ["block_size_resolved"], "kv_cache_size_tokens")
            if a.empty:
                continue
            st = series_style(idx)
            ax.fill_between(a["block_size_resolved"], a["min"], a["max"],
                            color=st["color"], alpha=0.15, linewidth=0)
            ax.plot(a["block_size_resolved"], a["median"], label=f"{dtype}{marker_note}",
                    markerfacecolor=SURFACE, markeredgewidth=1.3,
                    markeredgecolor=st["color"], **st)
            spread = (a["max"].max() - a["min"].min()) / a["median"].median() * 100
            findings.append(
                f"{src} {dtype}: token capacity varies {spread:.2f}% across "
                f"block sizes {sorted(a['block_size_resolved'].astype(int))}"
            )
            idx += 1
    ax.set_xscale("log", base=2)
    ax.set_xlabel("block_size (tokens per KV block)")
    ax.set_ylabel("KV pool capacity (tokens)")
    ax.set_title("Block granularity does not change pool capacity")
    ax.legend(frameon=False, fontsize=8)
    return "; ".join(findings) if findings else "not measured"


def panel_gmu_pool(ax: plt.Axes, p2: pd.DataFrame, p1: pd.DataFrame) -> str:
    """Pool capacity against gpu_memory_utilization, with the fitted slope.

    The slope of this line is the usable device memory: raising the utilisation
    fraction by dx hands the pool dx x (card) more bytes, because every other
    term in the budget is fixed. Recovering the card size from the slope is a
    strong check that nothing else is moving with the knob.
    """
    findings = []
    idx = 0
    for src, df in (("this project", p2), ("mined from P1", p1)):
        if df.empty:
            continue
        d = df.dropna(subset=["gpu_memory_utilization", "kv_cache_size_tokens"])
        d = d[d["cache_dtype_resolved"].isin(["auto", None]) | d["cache_dtype_resolved"].isna()]
        if d.empty:
            continue
        a = _agg(d, ["gpu_memory_utilization"], "kv_cache_size_tokens")
        if len(a) < 2:
            continue
        st = series_style(idx)
        ax.fill_between(a["gpu_memory_utilization"], a["min"], a["max"],
                        color=st["color"], alpha=0.15, linewidth=0)
        ax.plot(a["gpu_memory_utilization"], a["median"], label=src,
                markerfacecolor=SURFACE, markeredgewidth=1.3,
                markeredgecolor=st["color"], **st)
        bpt = d["kv_bytes_per_token"].dropna()
        slope_tokens = np.polyfit(a["gpu_memory_utilization"], a["median"], 1)[0]
        if not bpt.empty:
            slope_gib = slope_tokens * float(bpt.median()) / 1024**3
            findings.append(
                f"{src}: slope {slope_gib:.2f} GiB of pool per unit gpu_memory_utilization"
            )
        idx += 1
    ax.set_xlabel("gpu_memory_utilization")
    ax.set_ylabel("KV pool capacity (tokens)")
    ax.set_title("Pool is linear in the utilisation fraction")
    ax.legend(frameon=False, fontsize=8)
    return "; ".join(findings) if findings else "not measured"


def panel_preemption(ax: plt.Axes, p1: pd.DataFrame) -> str:
    """Preemptions per load phase against utilisation, from the mined sweeps.

    The counter is cumulative for a server's lifetime and one server serves many
    phases, so these are per-phase differences taken within a server group in
    time order. A phase with no predecessor in its group has no defined delta and
    is absent here rather than plotted as zero.
    """
    if p1.empty or "preemptions_phase" not in p1:
        ax.text(0.5, 0.5, "not measured", ha="center", va="center", transform=ax.transAxes)
        return "not measured"
    d = p1.dropna(subset=["preemptions_phase", "gpu_memory_utilization", "concurrency"])
    d = d[d["sweep_name"] == "memory-utilisation"]
    if d.empty:
        ax.text(0.5, 0.5, "not measured", ha="center", va="center", transform=ax.transAxes)
        return "not measured"
    findings = []
    for idx, (conc, sub) in enumerate(sorted(d.groupby("concurrency"))):
        a = _agg(sub, ["gpu_memory_utilization"], "preemptions_phase")
        st = series_style(idx)
        ax.fill_between(a["gpu_memory_utilization"], a["min"], a["max"],
                        color=st["color"], alpha=0.15, linewidth=0)
        ax.plot(a["gpu_memory_utilization"], a["median"], label=f"concurrency {int(conc)}",
                markerfacecolor=SURFACE, markeredgewidth=1.3,
                markeredgecolor=st["color"], **st)
        nz = a[a["median"] > 0]
        if nz.empty:
            findings.append(f"c={int(conc)}: no preemption at any utilisation measured")
        else:
            findings.append(
                f"c={int(conc)}: preemption at gmu<={nz['gpu_memory_utilization'].max():g}, "
                f"peak {nz['median'].max():.0f} per phase"
            )
    ax.set_xlabel("gpu_memory_utilization")
    ax.set_ylabel("preemptions per load phase")
    ax.set_title("Preemption under oversubscription (mined from P1)")
    ax.legend(frameon=False, fontsize=8)
    return "; ".join(findings)


def panel_occupancy(ax: plt.Axes, occ: pd.DataFrame) -> str:
    """This project's own oversubscription sweep: preemptions against concurrency.

    The vertical marks are the analytically predicted capacity, pool tokens
    divided by the full context length. The claim being tested is that
    preemption begins there and not somewhere else.
    """
    if occ.empty:
        ax.text(0.5, 0.5, "not measured", ha="center", va="center", transform=ax.transAxes)
        return "not measured"
    d = occ[(~occ["warmup"].astype(bool)) & (occ["phase_status"] != "skipped")]
    d = d.dropna(subset=["preemptions_delta", "concurrency"])
    if d.empty:
        ax.text(0.5, 0.5, "not measured", ha="center", va="center", transform=ax.transAxes)
        return "not measured"
    findings = []
    for idx, (gmu, sub) in enumerate(sorted(d.groupby("gpu_memory_utilization"))):
        a = _agg(sub, ["concurrency"], "preemptions_delta").sort_values("concurrency")
        st = series_style(idx % 4)
        ax.fill_between(a["concurrency"], a["min"], a["max"], color=st["color"],
                        alpha=0.15, linewidth=0)
        ax.plot(a["concurrency"], a["median"], label=f"gmu {gmu:g}",
                markerfacecolor=SURFACE, markeredgewidth=1.3,
                markeredgecolor=st["color"], **st)
        cap = (sub["kv_cache_size_tokens"] / sub["max_model_len"]).median()
        ax.axvline(cap, color=st["color"], linestyle=":", linewidth=1.0, alpha=0.8)
        onset = a[a["median"] > 0]["concurrency"]
        onset_txt = f"{int(onset.min())}" if not onset.empty else "none observed"
        findings.append(
            f"gmu {gmu:g}: predicted capacity {cap:.0f} sequences, "
            f"first preemption at concurrency {onset_txt}"
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("concurrent sequences offered")
    ax.set_ylabel("preemptions in the phase")
    ax.set_title("Preemption onset vs predicted capacity (dotted)")
    ax.legend(frameon=False, fontsize=8)
    return "; ".join(findings)


def main() -> int:
    apply_style()
    FIGS.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    p2 = measured(raw) if not raw.empty else pd.DataFrame()
    p1 = load_mined_p1()
    if not p1.empty:
        p1 = p1[(p1["status"] == "ok") & (~p1["warmup"].astype(bool))]
        p1 = p1[p1["tensor_parallel_size"] == 1]
        p1 = p1[p1["model_id"] == "Qwen/Qwen2.5-7B-Instruct"]
        # The analytical per-token size for Qwen2.5-7B in bf16, so the mined
        # token counts can be read in bytes on the same axes as ours.
        p1 = p1.assign(
            kv_bytes_per_token=np.where(
                p1["cache_dtype_resolved"].astype(str).str.startswith("fp8"), 28672, 57344
            )
        )

    from load_results import occupancy_rows

    occ = occupancy_rows(raw) if not raw.empty else pd.DataFrame()

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    findings = [
        panel_block_size(axes[0][0], p2, p1),
        panel_gmu_pool(axes[0][1], p2, p1),
        panel_preemption(axes[1][0], p1),
        panel_occupancy(axes[1][1], occ),
    ]
    caption = hardware_caption(p2) if not p2.empty else "this project: not measured"
    p1_note = (
        f"P1 series: {len(p1)} mined artifacts, driver "
        f"{'/'.join(sorted({str(v) for v in p1['gpu_model'].dropna().unique()}))}"
        if not p1.empty
        else "P1 series: not mined"
    )
    fig.text(0.0, -0.02, " · ".join([caption, p1_note,
             "median with min/max across repetitions; warmup excluded"]),
             ha="left", va="top", fontsize=7, color=TEXT_MUTED, wrap=True)
    fig.suptitle("PagedAttention: granularity, capacity and preemption", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    out = FIGS / "paged_attention.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    for f in findings:
        print(f"  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
