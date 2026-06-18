"""Shared paper-style plotting tokens and helpers."""
from __future__ import annotations

import csv
import math
import re
import textwrap
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


FONT_FAMILY = ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
MONO_FONT_FAMILY = ["Consolas", "DejaVu Sans Mono", "SF Mono", "Menlo", "monospace"]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
    "neutral_xlight": "#F4F5F7",
    "neutral_light": "#E2E5EA",
    "neutral_base": "#C5CAD3",
    "neutral_mid": "#7A828F",
    "neutral_dark": "#464C55",
}

COLOR_FAMILIES = {
    "blue": {
        "open": TOKENS["panel"],
        "xlight": "#EAF1FE",
        "light": "#CEDFFE",
        "base": "#A3BEFA",
        "mid": "#5477C4",
        "dark": "#2E4780",
    },
    "gold": {
        "open": TOKENS["panel"],
        "xlight": "#FFF4C2",
        "light": "#FFEA8F",
        "base": "#FFE15B",
        "mid": "#B8A037",
        "dark": "#736422",
    },
    "orange": {
        "open": TOKENS["panel"],
        "xlight": "#FFEDDE",
        "light": "#FFBDA1",
        "base": "#F0986E",
        "mid": "#CC6F47",
        "dark": "#804126",
    },
    "olive": {
        "open": TOKENS["panel"],
        "xlight": "#D8ECBD",
        "light": "#BEEB96",
        "base": "#A3D576",
        "mid": "#71B436",
        "dark": "#386411",
    },
    "pink": {
        "open": TOKENS["panel"],
        "xlight": "#FCDAD6",
        "light": "#F5BACC",
        "base": "#F390CA",
        "mid": "#BD569B",
        "dark": "#8A3A6F",
    },
}

MODE_ORDER = [
    "textual_only",
    "E",
    "O",
    "A",
    "B",
    "C",
    "D",
    "FMP_single",
    "FMP_multi",
    "FMP_single_plus_step",
]

MODE_LABELS = {
    "textual_only": "Textual-only",
    "E": "E semantic-only",
    "O": "O joint",
    "A": "A joint",
    "B": "B joint",
    "C": "C joint",
    "D": "D joint",
    "FMP_single": "FMP single",
    "FMP_multi": "FMP multi",
    "FMP_single_plus_step": "FMP single + step",
    "textual_semantic_joint_legacy": "legacy textual+semantic",
    "O_repo_legacy": "legacy O_repo",
    "O_fair_legacy": "legacy O_fair",
    "Random_Linf": "Random L-inf",
}

MODE_COLORS = {
    "textual_only": COLOR_FAMILIES["blue"]["mid"],
    "E": COLOR_FAMILIES["olive"]["mid"],
    "O": TOKENS["neutral_dark"],
    "A": COLOR_FAMILIES["pink"]["mid"],
    "B": COLOR_FAMILIES["blue"]["dark"],
    "C": COLOR_FAMILIES["olive"]["dark"],
    "D": COLOR_FAMILIES["gold"]["mid"],
    "FMP_single": COLOR_FAMILIES["orange"]["mid"],
    "FMP_multi": COLOR_FAMILIES["orange"]["dark"],
    "FMP_single_plus_step": COLOR_FAMILIES["pink"]["dark"],
    "textual_semantic_joint_legacy": TOKENS["neutral_mid"],
    "O_repo_legacy": TOKENS["neutral_mid"],
    "O_fair_legacy": TOKENS["neutral_mid"],
    "Random_Linf": TOKENS["neutral_base"],
}

MODE_LINESTYLES = {
    "textual_only": (0, (4, 2)),
    "E": (0, (1.3, 1.8)),
    "O": "-",
    "A": (0, (5, 2)),
    "B": (0, (3, 1.5)),
    "C": (0, (1.2, 1.6)),
    "D": (0, (5, 2, 1.5, 2)),
    "FMP_single": "-",
    "FMP_multi": (0, (3, 1.5)),
    "FMP_single_plus_step": (0, (5, 2, 1.5, 2)),
    "textual_semantic_joint_legacy": (0, (4, 2)),
    "O_repo_legacy": (0, (4, 2)),
    "O_fair_legacy": (0, (1.2, 1.6)),
    "Random_Linf": (0, (2, 2)),
}

METHOD_FAMILY = {
    "textual_only": "blue",
    "E": "olive",
    "O": "gold",
    "A": "pink",
    "B": "blue",
    "C": "olive",
    "D": "gold",
    "FMP_single": "orange",
    "FMP_multi": "orange",
    "FMP_single_plus_step": "pink",
}


def use_paper_style() -> None:
    """Apply a deterministic, publication-oriented Matplotlib theme."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.monospace": MONO_FONT_FAMILY,
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.titlecolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "axes.linewidth": 0.8,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.65,
            "grid.linestyle": "-",
            "legend.frameon": False,
            "savefig.facecolor": TOKENS["surface"],
            "savefig.edgecolor": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def label_mode(mode: str) -> str:
    return MODE_LABELS.get(str(mode), str(mode))


def mode_sort_key(mode: str) -> tuple[int, str]:
    mode = str(mode)
    if mode in MODE_ORDER:
        return MODE_ORDER.index(mode), mode
    return 100, mode


def ordered_modes(modes: Iterable[str], *, include_legacy: bool = True) -> list[str]:
    unique = []
    for mode in modes:
        if not mode:
            continue
        if not include_legacy and str(mode).endswith("_legacy"):
            continue
        if mode not in unique:
            unique.append(str(mode))
    return sorted(unique, key=mode_sort_key)


def color_for_mode(mode: str) -> str:
    return MODE_COLORS.get(str(mode), TOKENS["neutral_dark"])


def linestyle_for_mode(mode: str):
    return MODE_LINESTYLES.get(str(mode), "-")


def add_chart_header(
    fig,
    ax,
    title: str,
    subtitle: str,
    *,
    title_width: int = 84,
    subtitle_width: int = 118,
    top: float = 0.985,
) -> None:
    """Add a left-aligned figure header and reserve top space."""
    title = textwrap.fill(str(title).strip(), width=title_width, break_long_words=False)
    subtitle = textwrap.fill(str(subtitle).strip(), width=subtitle_width, break_long_words=False)
    if not title or not subtitle:
        raise ValueError("Every shipped chart needs a non-empty title and subtitle.")
    title_lines = title.count("\n") + 1
    subtitle_lines = subtitle.count("\n") + 1
    ax.set_title("")
    left = ax.get_position().x0
    fig.text(
        left,
        top,
        title,
        ha="left",
        va="top",
        fontsize=11.2,
        fontweight="semibold",
        color=TOKENS["ink"],
        linespacing=1.08,
    )
    fig.text(
        left,
        top - 0.055 - 0.034 * (title_lines - 1),
        subtitle,
        ha="left",
        va="top",
        fontsize=8.2,
        color=TOKENS["muted"],
        linespacing=1.18,
    )
    fig.subplots_adjust(top=max(0.54, 0.83 - 0.035 * (title_lines + subtitle_lines - 2)))


def add_center_header(
    fig,
    title: str,
    subtitle: str,
    *,
    title_size: float = 10.8,
    subtitle_size: float = 7.7,
) -> None:
    fig.text(
        0.5,
        0.975,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.5,
        0.918,
        subtitle,
        ha="center",
        va="top",
        fontsize=subtitle_size,
        color=TOKENS["muted"],
    )


def format_axis(ax, *, x_grid: bool = False, y_grid: bool = True) -> None:
    ax.set_facecolor(TOKENS["panel"])
    if y_grid:
        ax.grid(True, axis="y", alpha=0.95)
    else:
        ax.grid(False, axis="y")
    if x_grid:
        ax.grid(True, axis="x", alpha=0.65)
    else:
        ax.grid(False, axis="x")
    ax.tick_params(axis="both", labelsize=7.4, length=2.6, width=0.7, pad=1.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(TOKENS["axis"])
        ax.spines[spine].set_linewidth(0.8)


def format_integer_x(ax, nbins: int = 5) -> None:
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=nbins))


def axis_loss_formatter(value, _pos=None) -> str:
    if not math.isfinite(float(value)):
        return ""
    value = float(value)
    sign = "-" if value < 0 else ""
    abs_value = abs(value)
    if abs_value >= 1000000:
        return f"{sign}{abs_value / 1000000:.1f}M"
    if abs_value >= 100000:
        return f"{sign}{abs_value / 1000:.0f}k"
    if abs_value >= 1000:
        return f"{sign}{abs_value / 1000:.1f}k"
    if abs_value >= 10:
        return f"{value:.0f}"
    if abs_value >= 1:
        return f"{value:.2g}"
    return f"{value:.2f}"


def numeric_formatter() -> mticker.FuncFormatter:
    return mticker.FuncFormatter(axis_loss_formatter)


def bottom_legend(fig, handles: Sequence, *, ncol: int = 4, y: float = 0.004, fontsize: float = 7.6) -> None:
    handles = [h for h in handles if h is not None]
    if not handles:
        return
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=max(1, min(ncol, len(handles))),
        frameon=False,
        fontsize=fontsize,
        handlelength=2.8,
        columnspacing=1.7,
        handletextpad=0.55,
    )


def mode_legend_handles(modes: Iterable[str]) -> list[Line2D]:
    handles = []
    for mode in ordered_modes(modes):
        handles.append(
            Line2D(
                [0],
                [0],
                color=color_for_mode(mode),
                linestyle=linestyle_for_mode(mode),
                marker="o",
                markersize=3.6,
                linewidth=1.15,
                label=label_mode(mode),
            )
        )
    return handles


def patch_handle(label: str, color: str, edge: str | None = None) -> Patch:
    return Patch(facecolor=color, edgecolor=edge or TOKENS["neutral_dark"], linewidth=0.8, label=label)


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value)).strip("_")
    return value or "figure"


def save_figure(fig, out_base: Path, *, formats: Sequence[str] = ("png", "svg"), dpi: int = 300) -> list[Path]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        fmt = fmt.lower().strip(".")
        if not fmt:
            continue
        path = out_base.with_suffix(f".{fmt}")
        save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.035}
        if fmt in {"png", "jpg", "jpeg", "tif", "tiff"}:
            save_kwargs["dpi"] = dpi
        fig.savefig(path, **save_kwargs)
        paths.append(path)
    return paths


def write_manifest(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in records for key in row.keys()}) or ["figure_id"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def add_manifest_records(
    records: list[dict],
    *,
    figure_id: str,
    paths: Sequence[Path],
    source: str,
    family: str,
    description: str,
    expected_effect: str,
) -> None:
    for path in paths:
        records.append(
            {
                "figure_id": figure_id,
                "path": str(path),
                "source": source,
                "family": family,
                "description": description,
                "expected_effect": expected_effect,
            }
        )
