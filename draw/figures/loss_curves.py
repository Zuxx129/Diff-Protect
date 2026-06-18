#!/usr/bin/env python3
"""Paper-style loss and diagnostic curves for SD3 experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

from .io import find_loss_files, load_loss_data, normalize_array, smooth_array
from .style import (
    TOKENS,
    add_center_header,
    add_manifest_records,
    axis_loss_formatter,
    bottom_legend,
    color_for_mode,
    format_axis,
    format_integer_x,
    label_mode,
    linestyle_for_mode,
    mode_sort_key,
    numeric_formatter,
    save_figure,
    slugify,
    use_paper_style,
)


ABCD_MODES = {"O", "A", "B", "C", "D"}

COMPONENT_STYLES = {
    "total": {
        "label": "Total loss",
        "color": TOKENS["neutral_dark"],
        "linestyle": "-",
        "linewidth": 1.15,
    },
    "loss_total": {
        "label": "Total loss",
        "color": TOKENS["neutral_dark"],
        "linestyle": "-",
        "linewidth": 1.15,
    },
    "textual": {
        "label": "Textual loss",
        "color": "#5477C4",
        "linestyle": (0, (4, 2)),
        "linewidth": 1.15,
    },
    "loss_textual_mse": {
        "label": "Textual loss",
        "color": "#5477C4",
        "linestyle": (0, (4, 2)),
        "linewidth": 1.15,
    },
    "mmdit": {
        "label": "MMDiT loss",
        "color": "#CC6F47",
        "linestyle": (0, (5, 2, 1.5, 2)),
        "linewidth": 1.15,
    },
    "loss_o": {
        "label": "Semantic loss",
        "color": "#71B436",
        "linestyle": (0, (3, 1.5)),
        "linewidth": 1.15,
    },
    "loss_fmp": {
        "label": "FMP loss",
        "color": "#CC6F47",
        "linestyle": "-",
        "linewidth": 1.15,
    },
    "loss_mechanism": {
        "label": "Mechanism loss",
        "color": "#BD569B",
        "linestyle": (0, (1.2, 1.6)),
        "linewidth": 1.15,
    },
    "loss_step": {
        "label": "Step loss",
        "color": "#B8A037",
        "linestyle": (0, (5, 2, 1.5, 2)),
        "linewidth": 1.15,
    },
    "prediction_error": {
        "label": "Prediction error",
        "color": "#BD569B",
        "linestyle": (0, (3, 1.5)),
        "linewidth": 1.05,
    },
    "u_norm": {
        "label": "Target velocity norm",
        "color": "#71B436",
        "linestyle": (0, (1.2, 1.6)),
        "linewidth": 1.05,
    },
    "v_pred_norm": {
        "label": "Predicted velocity norm",
        "color": "#5477C4",
        "linestyle": (0, (5, 2)),
        "linewidth": 1.05,
    },
    "transition_sep": {
        "label": "Transition separation",
        "color": "#B8A037",
        "linestyle": (0, (4, 2)),
        "linewidth": 1.05,
    },
    "grad_total_l2": {
        "label": "Total gradient L2",
        "color": TOKENS["neutral_dark"],
        "linestyle": (0, (3, 1.5)),
        "linewidth": 1.05,
    },
    "grad_fmp_l2": {
        "label": "FMP gradient L2",
        "color": "#CC6F47",
        "linestyle": (0, (3, 1.5)),
        "linewidth": 1.05,
    },
    "grad_step_l2": {
        "label": "Step gradient L2",
        "color": "#B8A037",
        "linestyle": (0, (1.2, 1.6)),
        "linewidth": 1.05,
    },
}

COMPONENT_ORDER = [
    "total",
    "loss_total",
    "textual",
    "loss_textual_mse",
    "mmdit",
    "loss_o",
    "loss_fmp",
    "loss_mechanism",
    "loss_step",
    "prediction_error",
    "transition_sep",
    "u_norm",
    "v_pred_norm",
    "grad_total_l2",
    "grad_fmp_l2",
    "grad_step_l2",
]


def _mode_curve_sort(item) -> tuple:
    cfg = item[0]
    mode = str(cfg.get("mode", ""))
    return (
        *mode_sort_key(mode),
        int(cfg.get("epsilon", -1)) if cfg.get("epsilon") is not None else -1,
        int(cfg.get("steps", -1)) if cfg.get("steps") is not None else -1,
        str(cfg.get("seed", "")),
    )


def _subtitle_from_configs(configs: list[dict], normalize_flag: bool) -> str:
    parts = []
    eps = sorted({cfg.get("epsilon") for cfg in configs if cfg.get("epsilon") is not None})
    steps = sorted({cfg.get("steps") for cfg in configs if cfg.get("steps") is not None})
    seeds = sorted({cfg.get("seed") for cfg in configs if cfg.get("seed") is not None})
    if len(eps) == 1:
        parts.append(f"epsilon={eps[0]}")
    if len(steps) == 1:
        parts.append(f"{steps[0]} PGD steps")
    if len(seeds) == 1:
        parts.append(f"seed={seeds[0]}")
    parts.append("normalized loss values" if normalize_flag else "raw loss values")
    return "; ".join(parts)


def _series_for(data: dict, key: str, smooth_window: int | None, normalize: bool) -> np.ndarray:
    arr = np.asarray(data.get(key, np.array([], dtype=float)), dtype=float)
    arr = smooth_array(arr, smooth_window)
    if normalize:
        arr = normalize_array(arr)
    return arr


def _has_abcd_components(cfg: dict, data: dict) -> bool:
    if cfg.get("mode") not in ABCD_MODES:
        return False
    if "total" not in data or "textual" not in data or "mmdit" not in data:
        return False
    return bool(np.any(data["textual"]) or np.any(data["mmdit"]))


def load_grouped_loss_entries(
    root: str | Path,
    image: str | None,
    *,
    modes: list[str] | None = None,
    gmodes: list[str] | None = None,
) -> dict[str, list[tuple[dict, dict]]]:
    grouped: dict[str, list[tuple[dict, dict]]] = {}
    for cfg, image_id, loss_path in find_loss_files(root, image, representative=True):
        if modes and cfg.get("mode") not in modes:
            continue
        if gmodes and cfg.get("g_mode", "+") not in gmodes:
            continue
        try:
            data = load_loss_data(loss_path)
        except Exception as exc:
            print(f"Skip {loss_path}: {exc}")
            continue
        grouped.setdefault(image_id, []).append((cfg, data))
        total = data.get("total", data.get("loss_total", np.array([])))
        print(f"  [{cfg.get('mode')}] {image_id}: {len(total)} steps from {loss_path}")
    return grouped


def plot_modes(
    grouped: dict[str, list[tuple[dict, dict]]],
    output_dir: Path,
    *,
    formats: list[str],
    dpi: int,
    smooth: int | None = None,
    normalize: bool = False,
    log_scale: bool = False,
    manifest: list[dict] | None = None,
    source: str = "loss.npz",
) -> list[Path]:
    saved: list[Path] = []
    for image, curves in grouped.items():
        curves = sorted(curves, key=_mode_curve_sort)
        fig, ax = plt.subplots(figsize=(8.4, 4.35))
        handles = []
        used_modes = []
        for cfg, data in curves:
            mode = str(cfg.get("mode", ""))
            y = data.get("total", data.get("loss_total", np.array([])))
            y = smooth_array(y, smooth)
            if normalize:
                y = normalize_array(y)
            if len(y) == 0:
                continue
            x = np.arange(len(y))
            line, = ax.plot(
                x,
                y,
                color=color_for_mode(mode),
                linestyle=linestyle_for_mode(mode),
                linewidth=1.2,
                marker=None,
                label=label_mode(mode),
                alpha=0.96,
            )
            handles.append(line)
            used_modes.append(mode)

        if not handles:
            plt.close(fig)
            continue
        ax.set_xlabel("PGD step", fontsize=8.4, color=TOKENS["ink"])
        ax.set_ylabel("Normalized loss" if normalize else "Loss", fontsize=8.4, color=TOKENS["ink"])
        if log_scale and not normalize:
            ax.set_yscale("symlog")
        format_axis(ax, x_grid=True, y_grid=True)
        format_integer_x(ax)
        if not normalize:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(axis_loss_formatter))
        add_center_header(
            fig,
            f"{image} attack loss by mode",
            _subtitle_from_configs([cfg for cfg, _ in curves], normalize),
        )
        bottom_legend(fig, handles, ncol=min(max(len(handles), 1), 5), y=0.004)
        fig.subplots_adjust(left=0.095, right=0.985, top=0.80, bottom=0.25)
        paths = save_figure(fig, output_dir / f"{image}_loss_modes", formats=formats, dpi=dpi)
        plt.close(fig)
        saved.extend(paths)
        if manifest is not None:
            add_manifest_records(
                manifest,
                figure_id=f"{image}_loss_modes",
                paths=paths,
                source=source,
                family="Trend",
                description="Total PGD loss curves by method.",
                expected_effect="Optimized objectives should rise or converge; flat curves with near-zero gradients indicate implementation or startup issues.",
            )
    return saved


def plot_abcd_component_panel(
    image: str,
    curves: list[tuple[dict, dict]],
    output_dir: Path,
    *,
    formats: list[str],
    dpi: int,
    smooth: int | None = None,
    normalize: bool = False,
    manifest: list[dict] | None = None,
    source: str = "loss.npz",
) -> list[Path]:
    curves = sorted(curves, key=_mode_curve_sort)
    n_modes = len(curves)
    fig_width = max(3.0 * n_modes, 6.4)
    fig, axes = plt.subplots(1, n_modes, figsize=(fig_width, 2.75), squeeze=False, sharex=True)
    axes = axes[0]
    for idx, (cfg, data) in enumerate(curves):
        mode = str(cfg.get("mode", "?"))
        ax = axes[idx]
        ax2 = ax.twinx()
        total = _series_for(data, "total", smooth, normalize)
        textual = _series_for(data, "textual", smooth, normalize)
        mmdit = _series_for(data, "mmdit", smooth, normalize)
        x = np.arange(len(total))
        ax.plot(x, total, **{k: v for k, v in COMPONENT_STYLES["total"].items() if k != "label"})
        ax.plot(x, textual, **{k: v for k, v in COMPONENT_STYLES["textual"].items() if k != "label"})
        ax2.plot(x, mmdit, **{k: v for k, v in COMPONENT_STYLES["mmdit"].items() if k != "label"})
        ax.set_title(
            f"({chr(97 + idx)}) Mode {mode}",
            fontsize=8.6,
            fontweight="semibold",
            color=TOKENS["ink"],
            pad=5,
        )
        format_axis(ax, x_grid=True, y_grid=True)
        format_integer_x(ax)
        ax2.grid(False)
        ax2.tick_params(axis="y", labelsize=7.0, length=2.5, width=0.7, colors="#804126", pad=2.0)
        ax2.spines["right"].set_color(TOKENS["axis"])
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        if not normalize:
            ax.yaxis.set_major_formatter(numeric_formatter())
            ax2.yaxis.set_major_formatter(numeric_formatter())
        if len(total) > 0:
            ax.set_xlim(0, len(total) - 1)
        if idx == 0:
            ax.set_ylabel("Total / textual loss", fontsize=8.0, color=TOKENS["ink"])
        if idx == n_modes - 1:
            ax2.set_ylabel("MMDiT loss", fontsize=8.0, color="#804126")
    fig.supxlabel("PGD step", fontsize=8.4, color=TOKENS["ink"], y=0.085)
    add_center_header(
        fig,
        f"{image} loss components by mode",
        _subtitle_from_configs([cfg for cfg, _ in curves], normalize),
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=COMPONENT_STYLES[key]["color"],
            linestyle=COMPONENT_STYLES[key]["linestyle"],
            linewidth=COMPONENT_STYLES[key]["linewidth"],
            label=COMPONENT_STYLES[key]["label"],
        )
        for key in ["total", "textual", "mmdit"]
    ]
    bottom_legend(fig, handles, ncol=3, y=0.004)
    fig.subplots_adjust(left=0.045, right=0.975, top=0.79, bottom=0.24, wspace=0.56)
    paths = save_figure(fig, output_dir / f"{image}_loss_components", formats=formats, dpi=dpi)
    plt.close(fig)
    if manifest is not None:
        add_manifest_records(
            manifest,
            figure_id=f"{image}_loss_components",
            paths=paths,
            source=source,
            family="Small Multiples / Trend",
            description="O/A/B/C/D total, textual, and MMDiT loss components.",
            expected_effect="Textual and MMDiT components should reveal whether the joint objective is dominated by textual target-pull or structural terms.",
        )
    return paths


def plot_single_component_chart(
    image: str,
    cfg: dict,
    data: dict,
    output_dir: Path,
    *,
    formats: list[str],
    dpi: int,
    smooth: int | None = None,
    normalize: bool = False,
    log_scale: bool = False,
    manifest: list[dict] | None = None,
    source: str = "loss.npz",
) -> list[Path]:
    mode = str(cfg.get("mode", "unknown"))
    fig, ax = plt.subplots(figsize=(7.4, 4.25))
    handles = []
    for key in COMPONENT_ORDER:
        if key not in data or len(data[key]) == 0:
            continue
        if key == "loss_total" and "total" in data:
            continue
        if key == "textual" and "loss_textual_mse" in data:
            continue
        arr = _series_for(data, key, smooth, normalize)
        if len(arr) == 0:
            continue
        style = COMPONENT_STYLES.get(
            key,
            {"label": key, "color": TOKENS["neutral_dark"], "linestyle": "-", "linewidth": 1.05},
        )
        line, = ax.plot(
            np.arange(len(arr)),
            arr,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=style["label"],
            alpha=0.96,
        )
        handles.append(line)
    if not handles:
        plt.close(fig)
        return []
    ax.set_xlabel("PGD step", fontsize=8.4, color=TOKENS["ink"])
    ax.set_ylabel("Normalized value" if normalize else "Value", fontsize=8.4, color=TOKENS["ink"])
    if log_scale and not normalize:
        ax.set_yscale("symlog")
    format_axis(ax, x_grid=True, y_grid=True)
    format_integer_x(ax)
    if not normalize:
        ax.yaxis.set_major_formatter(numeric_formatter())
    add_center_header(
        fig,
        f"{image} - {label_mode(mode)} loss components",
        _subtitle_from_configs([cfg], normalize),
    )
    bottom_legend(fig, handles, ncol=min(max(len(handles), 1), 4), y=0.004)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.80, bottom=0.27)
    safe = slugify(f"{image}_{mode}_{cfg.get('g_mode', '')}_{cfg.get('opt_direction', '')}_{cfg.get('seed', '')}")
    paths = save_figure(fig, output_dir / f"{safe}_loss_components", formats=formats, dpi=dpi)
    plt.close(fig)
    if manifest is not None:
        add_manifest_records(
            manifest,
            figure_id=f"{safe}_loss_components",
            paths=paths,
            source=source,
            family="Trend",
            description=f"{label_mode(mode)} step-level loss and diagnostic curves.",
            expected_effect="FMP loss and gradients should stay finite and non-zero; step ablation fields appear only when the optional transition loss is enabled.",
        )
    return paths


def plot_components(
    grouped: dict[str, list[tuple[dict, dict]]],
    output_dir: Path,
    *,
    formats: list[str],
    dpi: int,
    smooth: int | None = None,
    normalize: bool = False,
    log_scale: bool = False,
    manifest: list[dict] | None = None,
    source: str = "loss.npz",
) -> list[Path]:
    saved: list[Path] = []
    for image, curves in grouped.items():
        curves = sorted(curves, key=_mode_curve_sort)
        abcd_curves = [(cfg, data) for cfg, data in curves if _has_abcd_components(cfg, data)]
        other_curves = [(cfg, data) for cfg, data in curves if (cfg, data) not in abcd_curves]
        if abcd_curves:
            saved.extend(
                plot_abcd_component_panel(
                    image,
                    abcd_curves,
                    output_dir,
                    formats=formats,
                    dpi=dpi,
                    smooth=smooth,
                    normalize=normalize,
                    manifest=manifest,
                    source=source,
                )
            )
        for cfg, data in other_curves:
            saved.extend(
                plot_single_component_chart(
                    image,
                    cfg,
                    data,
                    output_dir,
                    formats=formats,
                    dpi=dpi,
                    smooth=smooth,
                    normalize=normalize,
                    log_scale=log_scale,
                    manifest=manifest,
                    source=source,
                )
            )
    return saved


def generate_loss_figures(
    *,
    root: str | Path,
    output: str | Path,
    image: str | None = None,
    modes: list[str] | None = None,
    gmodes: list[str] | None = None,
    all_images: bool = False,
    components: bool = True,
    formats: list[str] | None = None,
    dpi: int = 300,
    smooth: int | None = None,
    normalize: bool = False,
    log_scale: bool = False,
    manifest: list[dict] | None = None,
) -> list[Path]:
    if not all_images and not image:
        raise ValueError("Specify image or all_images=True")
    use_paper_style()
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = formats or ["png", "svg"]
    grouped = load_grouped_loss_entries(root, None if all_images else image, modes=modes, gmodes=gmodes)
    if not grouped:
        print(f"No loss files found under {root}")
        return []
    source = str(root)
    saved = plot_modes(
        grouped,
        output_dir,
        formats=formats,
        dpi=dpi,
        smooth=smooth,
        normalize=normalize,
        log_scale=log_scale,
        manifest=manifest,
        source=source,
    )
    if components:
        saved.extend(
            plot_components(
                grouped,
                output_dir,
                formats=formats,
                dpi=dpi,
                smooth=smooth,
                normalize=normalize,
                log_scale=log_scale,
                manifest=manifest,
                source=source,
            )
        )
    for path in saved:
        print(f"Saved: {path}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SD3 loss curves for Diff-Protect experiments")
    parser.add_argument("--root", default="out_sd3")
    parser.add_argument("--image", default=None)
    parser.add_argument("--all", action="store_true", dest="all_images")
    parser.add_argument("--modes", nargs="*", default=None)
    parser.add_argument("--gmodes", nargs="*", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--smooth", type=int, default=None)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--components", action="store_true")
    parser.add_argument("--log", action="store_true", help="Use symlog scale for non-normalized non-twin-axis charts")
    parser.add_argument("--formats", default="png,svg", help="Comma-separated export formats")
    args = parser.parse_args()
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
    output_dir = Path(args.output or Path(args.root) / "figures")
    generate_loss_figures(
        root=args.root,
        output=output_dir,
        image=args.image,
        modes=args.modes,
        gmodes=args.gmodes,
        all_images=args.all_images,
        components=args.components,
        formats=formats,
        dpi=args.dpi,
        smooth=args.smooth,
        normalize=args.normalize,
        log_scale=args.log,
    )


if __name__ == "__main__":
    main()
