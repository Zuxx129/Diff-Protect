"""Mechanism and optimization diagnostic figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .io import add_derived_metrics, filter_rows, grouped_mean_ci, read_csv_rows, to_float
from .style import (
    TOKENS,
    add_chart_header,
    add_manifest_records,
    bottom_legend,
    color_for_mode,
    format_axis,
    label_mode,
    linestyle_for_mode,
    mode_legend_handles,
    numeric_formatter,
    ordered_modes,
    save_figure,
    slugify,
    use_paper_style,
)


DIAGNOSTIC_FIELDS = [
    "loss_total_final",
    "loss_textual_mse_final",
    "loss_o_final",
    "loss_mechanism_final",
    "loss_fmp_final",
    "loss_prediction_error_final",
    "loss_u_norm_final",
    "loss_v_pred_norm_final",
    "loss_grad_total_l2_final",
    "loss_grad_fmp_l2_final",
    "loss_linf_final",
    "loss_step_final",
    "loss_transition_sep_final",
    "loss_grad_step_l2_final",
    "loss_attn_injection_l2_final",
    "loss_feature_cos_final",
    "loss_velocity_div_final",
    "loss_modality_ratio_dev_final",
    "loss_cross_modal_cka_final",
]

FIELD_LABELS = {
    "loss_total_final": "total",
    "loss_textual_mse_final": "textual",
    "loss_o_final": "semantic",
    "loss_mechanism_final": "mechanism",
    "loss_fmp_final": "FMP",
    "loss_prediction_error_final": "pred. error",
    "loss_u_norm_final": "u norm",
    "loss_v_pred_norm_final": "v norm",
    "loss_grad_total_l2_final": "grad total",
    "loss_grad_fmp_l2_final": "grad FMP",
    "loss_linf_final": "L-inf",
    "loss_step_final": "step loss",
    "loss_transition_sep_final": "transition",
    "loss_grad_step_l2_final": "grad step",
    "loss_attn_injection_l2_final": "attn inj.",
    "loss_feature_cos_final": "feature cos",
    "loss_velocity_div_final": "velocity div.",
    "loss_modality_ratio_dev_final": "modality ratio",
    "loss_cross_modal_cka_final": "cross CKA",
}

CURVE_FIELDS = [
    "loss_total_final",
    "loss_textual_mse_final",
    "loss_o_final",
    "loss_mechanism_final",
    "loss_fmp_final",
    "loss_grad_total_l2_final",
    "loss_grad_fmp_l2_final",
    "loss_prediction_error_final",
]


def _has_metric(rows: list[dict], metric: str) -> bool:
    return any(to_float(row.get(metric)) is not None for row in rows)


def _field_label(field: str) -> str:
    return FIELD_LABELS.get(field, field.replace("_final", "").replace("loss_", "").replace("_", " "))


def _means_by_mode(rows: list[dict], fields: list[str]) -> tuple[list[str], list[str], np.ndarray]:
    modes = ordered_modes([str(r.get("mode", "")) for r in rows], include_legacy=False)
    fields = [field for field in fields if _has_metric(rows, field)]
    matrix = np.full((len(modes), len(fields)), np.nan, dtype=float)
    for i, mode in enumerate(modes):
        mode_rows = [row for row in rows if row.get("mode") == mode]
        for j, field in enumerate(fields):
            vals = [to_float(row.get(field)) for row in mode_rows]
            vals = [v for v in vals if v is not None]
            if vals:
                matrix[i, j] = float(np.mean(vals))
    return modes, fields, matrix


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    out = np.full_like(matrix, np.nan, dtype=float)
    for col in range(matrix.shape[1]):
        values = matrix[:, col]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        lo = float(finite.min())
        hi = float(finite.max())
        if hi <= lo:
            out[np.isfinite(values), col] = 0.5
        else:
            out[:, col] = (values - lo) / (hi - lo)
    return out


def plot_diagnostic_heatmap(
    rows: list[dict],
    out_dir: Path,
    *,
    formats: list[str],
    dpi: int,
    manifest: list[dict] | None = None,
    source: str = "full_metrics.csv",
) -> list[Path]:
    modes, fields, raw = _means_by_mode(rows, DIAGNOSTIC_FIELDS)
    if raw.size == 0 or not fields:
        return []
    norm = _normalize_columns(raw)
    fig_width = max(8.4, 0.54 * len(fields) + 3.2)
    fig_height = max(3.8, 0.36 * len(modes) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cmap = LinearSegmentedColormap.from_list(
        "diag_blue",
        [TOKENS["panel"], "#EAF1FE", "#CEDFFE", "#A3BEFA", "#5477C4"],
    )
    im = ax.imshow(norm, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(fields)), [_field_label(field) for field in fields], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(modes)), [label_mode(mode) for mode in modes])
    ax.tick_params(axis="both", labelsize=7.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(len(fields) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(modes) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=TOKENS["panel"], linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7.0, colors=TOKENS["muted"])
    cbar.set_label("Column-normalized mean", fontsize=7.5, color=TOKENS["muted"])
    add_chart_header(
        fig,
        ax,
        "Optimization and mechanism diagnostics",
        "Each column is min-max normalized across methods; use this figure to diagnose which internal objectives moved, not as external success evidence.",
    )
    fig.subplots_adjust(left=0.16, right=0.955, top=0.78, bottom=0.27)
    figure_id = "diagnostic_loss_field_heatmap"
    paths = save_figure(fig, out_dir / figure_id, formats=formats, dpi=dpi)
    plt.close(fig)
    if manifest is not None:
        add_manifest_records(
            manifest,
            figure_id=figure_id,
            paths=paths,
            source=source,
            family="Matrix & Cohort",
            description="Column-normalized final loss, gradient, and mechanism diagnostics by method.",
            expected_effect="FMP methods should show finite FMP loss and gradients; O/A/B/C/D should expose whether textual or MMDiT terms dominate.",
        )
    return paths


def plot_diagnostic_curve(
    rows: list[dict],
    field: str,
    out_dir: Path,
    *,
    formats: list[str],
    dpi: int,
    manifest: list[dict] | None = None,
    source: str = "full_metrics.csv",
) -> list[Path]:
    if not _has_metric(rows, field):
        return []
    stats = grouped_mean_ci(rows, ["mode", "epsilon"], field)
    if not stats:
        return []
    modes = ordered_modes([str(s["mode"]) for s in stats], include_legacy=False)
    fig, ax = plt.subplots(figsize=(8.0, 4.35))
    for mode in modes:
        part = sorted([s for s in stats if s["mode"] == mode], key=lambda s: float(s["epsilon"]))
        if not part:
            continue
        ax.errorbar(
            [float(p["epsilon"]) for p in part],
            [float(p["mean"]) for p in part],
            yerr=[float(p["ci95"]) for p in part],
            color=color_for_mode(mode),
            linestyle=linestyle_for_mode(mode),
            marker="o",
            markersize=3.3,
            linewidth=1.05,
            capsize=2.3,
            label=label_mode(mode),
        )
    ax.set_xlabel("Epsilon", fontsize=8.5, color=TOKENS["ink"])
    ax.set_ylabel(_field_label(field), fontsize=8.5, color=TOKENS["ink"])
    ax.yaxis.set_major_formatter(numeric_formatter())
    format_axis(ax, x_grid=True, y_grid=True)
    add_chart_header(
        fig,
        ax,
        f"Diagnostic field vs epsilon: {_field_label(field)}",
        "Mean final logged diagnostic over available paired rows; this checks optimization behavior and should be interpreted with external edit metrics.",
    )
    bottom_legend(fig, mode_legend_handles(modes), ncol=5, y=0.004)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.79, bottom=0.24)
    figure_id = f"diagnostic_vs_epsilon_{slugify(field)}"
    paths = save_figure(fig, out_dir / figure_id, formats=formats, dpi=dpi)
    plt.close(fig)
    if manifest is not None:
        add_manifest_records(
            manifest,
            figure_id=figure_id,
            paths=paths,
            source=source,
            family="Trend",
            description=f"Final {_field_label(field)} as epsilon changes.",
            expected_effect="Fields should move consistently with the intended objective; external metrics decide whether that movement is useful.",
        )
    return paths


def generate_diagnostic_figures(
    *,
    metrics_path: str | Path,
    out_dir: str | Path,
    formats: list[str] | None = None,
    dpi: int = 300,
    manifest: list[dict] | None = None,
) -> list[Path]:
    use_paper_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = formats or ["png", "svg"]
    rows = add_derived_metrics(read_csv_rows(metrics_path))
    if not rows:
        print(f"No metrics rows found for diagnostics: {metrics_path}")
        return []
    saved = plot_diagnostic_heatmap(rows, out_dir, formats=formats, dpi=dpi, manifest=manifest, source=str(metrics_path))
    for field in CURVE_FIELDS:
        saved.extend(plot_diagnostic_curve(rows, field, out_dir, formats=formats, dpi=dpi, manifest=manifest, source=str(metrics_path)))
    for path in saved:
        print(f"Saved: {path}")
    return saved
