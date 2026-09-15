"""Shared figure style.

These figures go in a portfolio and should read like SC-submission plots, so the
rules from ../CLAUDE.md are enforced here rather than remembered per script:
300 dpi, axis units labelled, hardware and model in the caption, repetition count
and spread always shown.

Colour follows a validated categorical palette used **unchanged and in its
documented order**. Two consequences are baked into the helpers below:

* Series are assigned slots in fixed order and never cycled. Past four series,
  facet rather than inventing a fifth hue.
* Three slots in this palette fall below 3:1 contrast on a light surface, which
  triggers the relief rule: those series must carry a visible direct label, not
  colour alone. :func:`label_line_ends` exists so that is the default path, and
  every multi-series chart also gets a legend, so identity is never colour-only.

Frontier and scaling curves are drawn as **connected lines with markers**, not
free scatter. That is partly a reading decision (the connection is the sweep) and
partly a palette one: the four-slot ordering validates on the adjacent pairlist
that lines use, whereas an unconnected four-series scatter would put yellow and
orange on screen as an all-pairs comparison, which fails the normal-vision floor.
Marker shape is carried as a redundant channel regardless.
"""

from __future__ import annotations

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical slots, in fixed order. Never cycle, never generate a fifth.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

# Redundant encoding, so identity survives greyscale printing and CVD.
MARKERS = ("o", "s", "^", "D")

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#6f6e6a"
GRID = "#dcdbd6"
SURFACE = "#ffffff"

# Status colours are reserved and never reused as a series hue.
STATUS_CRITICAL = "#e34948"


def apply_style() -> None:
    """Install the house style. Call once at the top of any plotting script."""
    mpl.rcParams.update(
        {
            "figure.dpi": 150,          # on-screen; savefig overrides to 300
            "savefig.dpi": 300,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.labelcolor": TEXT_PRIMARY,
            "text.color": TEXT_PRIMARY,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            # Recessive grid and axes: the data carries the ink.
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 1.0,
            "axes.axisbelow": True,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.6,
            "lines.markersize": 5.5,
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
        }
    )


def series_style(index: int) -> dict[str, Any]:
    """Colour and marker for series ``index``.

    Raises past the palette rather than cycling: a repeated hue silently merges
    two configurations into one apparent series, which is worse than a crash.
    """
    if index >= len(SERIES):
        raise ValueError(
            f"series index {index} exceeds the {len(SERIES)}-slot categorical palette. "
            "Facet into small multiples or fold into 'Other'; do not cycle hues."
        )
    return {"color": SERIES[index], "marker": MARKERS[index]}


def plot_with_spread(
    ax: plt.Axes,
    x: list[float],
    median: list[float],
    lo: list[float],
    hi: list[float],
    *,
    index: int,
    label: str,
    reps: list[list[float]] | None = None,
) -> None:
    """Draw one series as median with a min/max band, never a bare point.

    A point estimate with no error indication does not go in a figure. When the
    individual repetitions are available they are drawn too, so the reader sees
    the actual spread rather than only its summary: with N=3 an error bar is a
    very lossy description of three numbers.
    """
    style = series_style(index)
    ax.fill_between(x, lo, hi, color=style["color"], alpha=0.15, linewidth=0)
    ax.plot(
        x,
        median,
        label=label,
        markerfacecolor=SURFACE,
        markeredgewidth=1.4,
        markeredgecolor=style["color"],
        **style,
    )
    if reps:
        for xi, values in zip(x, reps, strict=True):
            ax.plot(
                [xi] * len(values),
                values,
                linestyle="none",
                marker="_",
                markersize=7,
                color=style["color"],
                alpha=0.55,
            )


def label_line_ends(ax: plt.Axes, x: float, y: float, text: str, index: int) -> None:
    """Direct-label a series at its end.

    Not decoration: several slots in this palette sit below 3:1 contrast on a
    light surface, and the relief rule requires those series to be identifiable
    without relying on colour.
    """
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=8,
        color=TEXT_SECONDARY,
    )


def provenance_caption(
    fig: plt.Figure,
    *,
    hardware: str,
    model: str,
    backend: str,
    repetitions: int,
    extra: str = "",
) -> None:
    """Stamp the caption every figure in this project must carry.

    Hardware, model and backend version are not optional context: a throughput
    number is meaningless without them, and a figure that travels into a slide
    deck without them invites exactly the wrong comparison.
    """
    bits = [hardware, model, backend, f"N={repetitions} repetitions, median with min/max"]
    if extra:
        bits.append(extra)
    fig.text(
        0.0,
        -0.02,
        " · ".join(bits),
        ha="left",
        va="top",
        fontsize=7,
        color=TEXT_MUTED,
        wrap=True,
    )


def hardware_string(row: Any) -> str:
    """Build the hardware half of a caption from a loaded results row."""
    count = row.get("gpu_count")
    model = row.get("gpu_model", "unknown GPU")
    driver = row.get("driver_version")
    parts = [f"{count}x {model}" if count else str(model)]
    if driver:
        parts.append(f"driver {driver}")
    return ", ".join(parts)


def backend_string(row: Any) -> str:
    name = row.get("backend", "?")
    version = row.get("backend_version")
    digest = row.get("image_digest")
    out = f"{name} {version}" if version else str(name)
    if digest:
        out += f" ({str(digest)[:19]})"
    return out
