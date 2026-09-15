#!/usr/bin/env python3
"""Predicted vs measured KV cache, and the decomposition of the gap.

    python analysis/plot_predicted_vs_measured.py

There are two different "predicted vs measured" questions hiding in this
project, and separating them is most of the analysis:

**1. Is the per-token formula right?**  Compare the KV pool implied by the
engine's own token count (`kv_cache_size_tokens` x the analytical
`2*L*H_kv*D*dtype_bytes`) against the pool size the engine reports in GiB. These
should agree to within printing precision, and if they do, the arithmetic of the
cache is exactly understood.

**2. Is the naive capacity estimate right?**  The estimate an engineer actually
makes on a whiteboard is `gpu_memory_utilization x card_size - weights`. That one
is wrong, and by how much and why is the deliverable. Everything between the two
is a named, measured term:

* the card's nominal size is not its usable size (ECC and driver reserve),
* "weights" as reported by the loader is not "weights plus non-torch": NCCL,
  cuBLAS workspaces and the CUDA context are real and are charged to the budget,
* the profile run's peak activation memory is subtracted before the pool is sized,
* CUDA graph capture takes its bite *after* profiling,
* and the pool is then rounded down to a whole number of blocks.

Every one of those terms is read from the engine's own startup accounting rather
than attributed by us, so the residual is decomposed, not hand-waved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_results import GIB, hardware_caption, load_raw, measured  # noqa: E402
from style import (  # noqa: E402
    SERIES,
    STATUS_CRITICAL,
    SURFACE,
    TEXT_MUTED,
    TEXT_SECONDARY,
    apply_style,
    series_style,
)

REPO = Path(__file__).resolve().parents[1]
FIGS = REPO / "results" / "figures"

# Terms of the budget, in the order the engine spends them. Colours are assigned
# once here so the same term is the same colour in every panel.
TERMS = [
    ("log_weights_plus_non_torch_gib", "weights + non-torch", "#4a4a4a"),
    ("log_peak_activation_gib", "peak activation (profile run)", SERIES[1]),
    ("log_cudagraph_gib", "CUDA graph capture", SERIES[3]),
    ("log_kv_cache_in_use_gib", "KV cache pool", SERIES[0]),
]


def panel_formula_exactness(ax: plt.Axes, df: pd.DataFrame) -> str:
    """Analytical pool bytes against the engine's reported pool bytes."""
    d = df.dropna(subset=["analytical_pool_gib", "log_kv_cache_in_use_gib"])
    if d.empty:
        ax.text(0.5, 0.5, "not measured", ha="center", va="center", transform=ax.transAxes)
        return "not measured"
    x = d["log_kv_cache_in_use_gib"].to_numpy(dtype=float)
    y = d["analytical_pool_gib"].to_numpy(dtype=float)
    lim = [0, max(x.max(), y.max()) * 1.08]
    ax.plot(lim, lim, color=TEXT_MUTED, linewidth=0.9, linestyle="--", zorder=1)
    for i, (dtype, sub) in enumerate(d.groupby("cache_dtype_resolved")):
        st = series_style(i)
        ax.plot(
            sub["log_kv_cache_in_use_gib"],
            sub["analytical_pool_gib"],
            linestyle="none",
            label=f"kv dtype {dtype}",
            markerfacecolor=SURFACE,
            markeredgewidth=1.3,
            markeredgecolor=st["color"],
            marker=st["marker"],
            color=st["color"],
        )
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("engine-reported KV pool (GiB)")
    ax.set_ylabel("analytical pool from token count (GiB)")
    ax.set_title("1. Is the per-token formula right?")
    ax.legend(frameon=False, loc="upper left")

    rel = np.abs(y - x) / x
    worst = rel.max() * 100
    ax.annotate(
        f"max |error| {worst:.2f}%  (N={len(d)} launches)",
        xy=(0.98, 0.04),
        xycoords="axes fraction",
        ha="right",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    return f"max relative error {worst:.3f}% over {len(d)} launches"


def panel_budget_breakdown(ax: plt.Axes, df: pd.DataFrame) -> str:
    """Where gpu_memory_utilization x card actually goes, as measured."""
    cols = [c for c, _, _ in TERMS] + ["log_total_device_memory_gib", "log_memory_budget_gib"]
    d = df.dropna(subset=cols)
    if d.empty:
        ax.text(0.5, 0.5, "not measured", ha="center", va="center", transform=ax.transAxes)
        return "not measured"
    g = d.groupby("gpu_memory_utilization")[cols].median().sort_index()
    x = np.arange(len(g))
    bottom = np.zeros(len(g))
    for col, label, colour in TERMS:
        vals = g[col].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, color=colour, label=label, width=0.72, linewidth=0)
        bottom += vals

    nominal = g["log_total_device_memory_gib"].to_numpy(dtype=float)
    budget = g["log_memory_budget_gib"].to_numpy(dtype=float)
    ax.plot(x, budget, color=STATUS_CRITICAL, linewidth=1.4, marker="_", markersize=18,
            linestyle="none", label="requested budget (gmu x usable)")
    ax.plot(x, nominal, color=TEXT_MUTED, linewidth=1.0, linestyle=":",
            label="usable device memory")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in g.index], rotation=0)
    ax.set_xlabel("gpu_memory_utilization")
    ax.set_ylabel("GPU memory (GiB)")
    ax.set_title("2. Where the budget actually goes")
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)

    overshoot = (bottom - budget)
    return (
        f"allocated minus requested budget ranges "
        f"{overshoot.min():+.2f} to {overshoot.max():+.2f} GiB across "
        f"{len(g)} utilisation settings"
    )


def panel_naive_vs_measured(ax: plt.Axes, df: pd.DataFrame) -> str:
    """The whiteboard estimate against the pool the engine actually got."""
    need = [
        "gpu_memory_utilization",
        "log_total_device_memory_gib",
        "log_model_loading_gib",
        "log_kv_cache_in_use_gib",
    ]
    d = df.dropna(subset=need)
    if d.empty:
        ax.text(0.5, 0.5, "not measured", ha="center", va="center", transform=ax.transAxes)
        return "not measured"

    # The estimate an engineer makes without the engine's accounting: nominal
    # card size (40 GiB, the number on the datasheet) times the utilisation
    # fraction, minus the weights as the loader reports them.
    nominal_datasheet = 40.0
    d = d.assign(
        naive_gib=d["gpu_memory_utilization"] * nominal_datasheet - d["log_model_loading_gib"],
        informed_gib=d["gpu_memory_utilization"] * d["log_total_device_memory_gib"]
        - d["log_model_loading_gib"],
    )
    g = d.groupby("gpu_memory_utilization")[
        ["naive_gib", "informed_gib", "log_kv_cache_in_use_gib"]
    ].agg(["median", "min", "max"]).sort_index()
    x = g.index.to_numpy(dtype=float)

    for i, (col, label) in enumerate(
        [
            ("naive_gib", "naive: gmu x 40 GiB - weights"),
            ("informed_gib", "gmu x usable - weights"),
            ("log_kv_cache_in_use_gib", "measured KV pool"),
        ]
    ):
        st = series_style(i)
        ax.fill_between(x, g[(col, "min")], g[(col, "max")], color=st["color"], alpha=0.15,
                        linewidth=0)
        ax.plot(x, g[(col, "median")], label=label, markerfacecolor=SURFACE,
                markeredgewidth=1.3, markeredgecolor=st["color"], **st)
    ax.set_xlabel("gpu_memory_utilization")
    ax.set_ylabel("KV cache pool (GiB)")
    ax.set_title("3. Naive estimate vs what the engine got")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    naive = g[("naive_gib", "median")].to_numpy(dtype=float)
    meas = g[("log_kv_cache_in_use_gib", "median")].to_numpy(dtype=float)
    err = (naive - meas) / meas * 100
    return (
        f"naive estimate overstates the pool by {err.min():.1f}% to {err.max():.1f}% "
        f"across gmu {x.min():g}-{x.max():g}"
    )


def main() -> int:
    apply_style()
    FIGS.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    if raw.empty:
        print("no artifacts under results/raw/ yet; nothing to plot")
        return 0
    df = measured(raw)
    if df.empty:
        print("no successful non-warmup launches yet; nothing to plot")
        return 0

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    findings = [
        panel_formula_exactness(axes[0], df),
        panel_budget_breakdown(axes[1], df),
        panel_naive_vs_measured(axes[2], df),
    ]
    reps = int(df.groupby(["sweep_name", "config_index"]).size().median())
    fig.text(
        0.0,
        -0.04,
        " · ".join(
            [
                hardware_caption(df),
                f"N={reps} non-warmup repetitions per configuration, median with min/max",
                f"{len(df)} successful launches, "
                f"{int((raw['status'] != 'ok').sum())} recorded failures",
            ]
        ),
        ha="left",
        va="top",
        fontsize=7,
        color=TEXT_MUTED,
        wrap=True,
    )
    fig.suptitle(
        "KV cache: predicted analytically, measured on the engine, residual decomposed",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    out = FIGS / "predicted_vs_measured.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    for f in findings:
        print(f"  {f}")

    # Also emit the same numbers as a table, so no figure has to be read off.
    d = df.dropna(subset=["analytical_pool_gib", "log_kv_cache_in_use_gib"])
    if not d.empty:
        d = d.assign(
            residual_gib=d["log_kv_cache_in_use_gib"] - d["analytical_pool_gib"],
            residual_pct=(d["log_kv_cache_in_use_gib"] - d["analytical_pool_gib"])
            / d["log_kv_cache_in_use_gib"]
            * 100,
        )
        cols = [
            "sweep_name",
            "max_model_len",
            "max_num_seqs",
            "gpu_memory_utilization",
            "block_size_resolved",
            "cache_dtype_resolved",
            "kv_cache_size_tokens",
            "num_gpu_blocks",
            "analytical_pool_gib",
            "log_kv_cache_in_use_gib",
            "residual_gib",
            "residual_pct",
            "log_weights_plus_non_torch_gib",
            "log_peak_activation_gib",
            "log_cudagraph_gib",
            "log_total_device_memory_gib",
        ]
        tbl = (
            d[cols]
            .groupby(cols[:6], dropna=False)
            .median(numeric_only=True)
            .reset_index()
            .sort_values(cols[:6])
        )
        out_csv = FIGS / "predicted_vs_measured.csv"
        tbl.to_csv(out_csv, index=False)
        print(f"wrote {out_csv} ({len(tbl)} configurations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
