"""Ablation-specific figures for SD3 v3.1 experiments."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .io import add_derived_metrics, filter_rows, grouped_mean_ci, read_csv_rows, to_float
from .metric_comparison import METRIC_SPECS
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


FMP_MODES = ["FMP_single", "FMP_multi"]
STEP_MODES = ["FMP_single", "FMP_single_plus_step"]
ABLATION_METRICS = ["l2_edit_rmse", "lpips_edit", "source_similarity_drop", "clip_prompt_drop"]
STEP_DIAGNOSTICS = ["loss_step_final", "loss_transition_sep_final", "loss_grad_step_l2_final"]


def _has_metric(rows: list[dict], metric: str) -> bool:
    return any(to_float(row.get(metric)) is not None for row in rows)


def _metric_label(metric: str) -> str:
    return METRIC_SPECS.get(metric, {}).get("label", metric.replace("_", " "))


def plot_ablation_line(
    rows: list[dict],
    metric: str,
    axis_key: str,
    modes: list[str],
    out_dir: Path,
    *,
    formats: list[str],
    dpi: int,
    figure_prefix: str,
    title: str,
    subtitle: str,
    expected_effect: str,
    manifest: list[dict] | None = None,
    source: str = "full_metrics.csv",
) -> list[Path]:
    if not _has_metric(rows, metric):
        return []
    plot_rows = filter_rows(rows, modes=modes, paired=True)
    stats = grouped_mean_ci(plot_rows, ["mode", axis_key], metric)
    stats = [s for s in stats if to_float(s.get(axis_key)) is not None]
    if not stats:
        return []
    used_modes = ordered_modes([str(s["mode"]) for s in stats], include_legacy=False)
    fig, ax = plt.subplots(figsize=(7.6, 4.35))
    for mode in used_modes:
        part = sorted([s for s in stats if s["mode"] == mode], key=lambda s: float(s[axis_key]))
        if not part:
            continue
        ax.errorbar(
            [float(p[axis_key]) for p in part],
            [float(p["mean"]) for p in part],
            yerr=[float(p["ci95"]) for p in part],
            color=color_for_mode(mode),
            linestyle=linestyle_for_mode(mode),
            marker="o",
            markersize=3.6,
            linewidth=1.15,
            capsize=2.5,
            label=label_mode(mode),
        )
    ax.set_xlabel("Epsilon" if axis_key == "epsilon" else "SDEdit noise level sigma", fontsize=8.5, color=TOKENS["ink"])
    ax.set_ylabel(_metric_label(metric), fontsize=8.5, color=TOKENS["ink"])
    ax.yaxis.set_major_formatter(numeric_formatter())
    format_axis(ax, x_grid=True, y_grid=True)
    add_chart_header(fig, ax, title, subtitle)
    bottom_legend(fig, mode_legend_handles(used_modes), ncol=3, y=0.004)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.79, bottom=0.24)
    figure_id = f"{figure_prefix}_{axis_key}_{slugify(metric)}"
    paths = save_figure(fig, out_dir / figure_id, formats=formats, dpi=dpi)
    plt.close(fig)
    if manifest is not None:
        add_manifest_records(
            manifest,
            figure_id=figure_id,
            paths=paths,
            source=source,
            family="Trend",
            description=f"{figure_prefix} ablation for {_metric_label(metric)} across {axis_key}.",
            expected_effect=expected_effect,
        )
    return paths


def plot_step_diagnostic(
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
    plot_rows = filter_rows(rows, modes=STEP_MODES, paired=True)
    stats = grouped_mean_ci(plot_rows, ["mode"], field)
    if not stats:
        return []
    modes = ordered_modes([str(s["mode"]) for s in stats], include_legacy=False)
    stat_by_mode = {str(s["mode"]): s for s in stats}
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    x = range(len(modes))
    values = [float(stat_by_mode[m]["mean"]) for m in modes]
    ci = [float(stat_by_mode[m]["ci95"]) for m in modes]
    bars = ax.bar(
        list(x),
        values,
        yerr=ci,
        color=[color_for_mode(m) for m in modes],
        edgecolor=[color_for_mode(m) for m in modes],
        linewidth=0.9,
        capsize=3,
    )
    for bar in bars:
        bar.set_alpha(0.86)
    ax.set_xticks(list(x), [label_mode(m) for m in modes], rotation=15, ha="right")
    ax.set_ylabel(field.replace("_final", "").replace("loss_", "").replace("_", " "), fontsize=8.5, color=TOKENS["ink"])
    ax.yaxis.set_major_formatter(numeric_formatter())
    format_axis(ax, x_grid=False, y_grid=True)
    add_chart_header(
        fig,
        ax,
        "Optional transition-step ablation diagnostic",
        "This figure appears only when FMP_single_plus_step was run and logged step-specific fields.",
    )
    fig.subplots_adjust(left=0.14, right=0.975, top=0.76, bottom=0.24)
    figure_id = f"step_ablation_{slugify(field)}"
    paths = save_figure(fig, out_dir / figure_id, formats=formats, dpi=dpi)
    plt.close(fig)
    if manifest is not None:
        add_manifest_records(
            manifest,
            figure_id=figure_id,
            paths=paths,
            source=source,
            family="Comparison & Ranking",
            description=f"Step ablation diagnostic: {field}.",
            expected_effect="If L_step is helpful, transition separation or edit metrics may improve; if not, keep FMP as the main method.",
        )
    return paths


def generate_ablation_figures(
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
        print(f"No metrics rows found for ablations: {metrics_path}")
        return []
    saved: list[Path] = []
    for metric in ABLATION_METRICS:
        saved.extend(
            plot_ablation_line(
                rows,
                metric,
                "epsilon",
                FMP_MODES,
                out_dir,
                formats=formats,
                dpi=dpi,
                figure_prefix="fmp_single_multi",
                title=f"FMP single vs multi across epsilon: {_metric_label(metric)}",
                subtitle="Comparison isolates whether multi-sigma expectation improves robustness over a single-sigma proxy.",
                expected_effect="FMP_multi should reduce single-sigma overfitting and keep comparable or better edit disruption across budgets.",
                manifest=manifest,
                source=str(metrics_path),
            )
        )
        saved.extend(
            plot_ablation_line(
                rows,
                metric,
                "sigma",
                FMP_MODES,
                out_dir,
                formats=formats,
                dpi=dpi,
                figure_prefix="fmp_single_multi",
                title=f"FMP single vs multi across SDEdit sigma: {_metric_label(metric)}",
                subtitle="Eval-sigma curves test whether the optimization target transfers beyond one noising level.",
                expected_effect="FMP_multi is expected to be smoother across eval sigmas; FMP_single may peak near its optimized sigma.",
                manifest=manifest,
                source=str(metrics_path),
            )
        )
    for metric in ABLATION_METRICS:
        saved.extend(
            plot_ablation_line(
                rows,
                metric,
                "epsilon",
                STEP_MODES,
                out_dir,
                formats=formats,
                dpi=dpi,
                figure_prefix="step_ablation",
                title=f"Optional step loss ablation: {_metric_label(metric)}",
                subtitle="This optional comparison is interpreted only if FMP_single_plus_step data exists.",
                expected_effect="L_step should only be kept if it improves edit disruption enough to justify its extra compute.",
                manifest=manifest,
                source=str(metrics_path),
            )
        )
    for field in STEP_DIAGNOSTICS:
        saved.extend(plot_step_diagnostic(rows, field, out_dir, formats=formats, dpi=dpi, manifest=manifest, source=str(metrics_path)))
    for path in saved:
        print(f"Saved: {path}")
    return saved

