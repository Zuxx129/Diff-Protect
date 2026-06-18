"""Main metric comparison figures for SD3 v3.1 experiments."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .io import (
    add_derived_metrics,
    default_condition,
    describe_condition,
    filter_rows,
    grouped_mean_ci,
    read_csv_rows,
    to_float,
)
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


MAIN_METHODS = ["textual_only", "E", "O", "A", "B", "C", "D", "FMP_single", "FMP_multi"]

METRIC_SPECS = {
    "l2_edit_rmse": {
        "label": "Edit RMSE",
        "title": "Edit output divergence by method",
        "subtitle": "Paired clean/adv SDEdit RMSE; higher values indicate stronger edit disruption.",
        "expected": "FMP should exceed textual_only, E, and O/A/B/C/D if flow-matching prediction error transfers to edited outputs.",
    },
    "lpips_edit": {
        "label": "Edit LPIPS",
        "title": "Perceptual edit divergence by method",
        "subtitle": "Paired clean/adv SDEdit LPIPS; higher values indicate stronger perceptual disruption.",
        "expected": "FMP_multi is expected to be at least as strong as FMP_single and more stable across seeds or sigmas.",
    },
    "clip_prompt_drop": {
        "label": "Prompt CLIP drop",
        "title": "Prompt alignment drop by method",
        "subtitle": "Computed as clean edit CLIPScore minus protected edit CLIPScore; higher values mean prompt alignment is reduced.",
        "expected": "Effective protection should reduce prompt alignment without violating perturbation constraints.",
    },
    "source_similarity_drop": {
        "label": "Source similarity drop",
        "title": "Source similarity drop by method",
        "subtitle": "Computed as clean source similarity minus protected source similarity; higher values indicate loss of source identity or structure.",
        "expected": "FMP should increase source-similarity drop when the protected edit moves away from the clean edit trajectory.",
    },
    "linf_perturb_tensor_-1_1": {
        "label": "Perturbation L-infinity",
        "title": "Perturbation budget check",
        "subtitle": "Tensor-space L-infinity perturbation; values must stay within the configured epsilon budget.",
        "expected": "All methods should satisfy the same L-infinity constraint, so differences should mostly reflect attack objective quality.",
    },
    "l2_perturb_rmse": {
        "label": "Perturbation RMSE",
        "title": "Perturbation magnitude by method",
        "subtitle": "Pixel-space RMSE between original and protected images; lower values are visually preferable.",
        "expected": "Useful methods should keep perturbation RMSE low while increasing edit disruption.",
    },
    "lpips_perturb": {
        "label": "Perturbation LPIPS",
        "title": "Perturbation perceptual visibility",
        "subtitle": "LPIPS between original and protected images; lower values indicate better perceptual invisibility.",
        "expected": "FMP should not gain edit disruption only by producing visibly worse protected images.",
    },
}

EPSILON_METRICS = [
    "l2_edit_rmse",
    "lpips_edit",
    "clip_prompt_drop",
    "source_similarity_drop",
    "linf_perturb_tensor_-1_1",
    "l2_perturb_rmse",
    "lpips_perturb",
]

SIGMA_METRICS = ["l2_edit_rmse", "lpips_edit", "source_similarity_drop"]
SUMMARY_METRICS = ["l2_edit_rmse", "lpips_edit", "clip_prompt_drop", "source_similarity_drop"]


def _has_metric(rows: Sequence[dict], metric: str) -> bool:
    return any(to_float(row.get(metric)) is not None for row in rows)


def _metric_label(metric: str) -> str:
    return METRIC_SPECS.get(metric, {}).get("label", metric.replace("_", " "))


def _metric_title(metric: str) -> str:
    return METRIC_SPECS.get(metric, {}).get("title", _metric_label(metric))


def _metric_subtitle(metric: str) -> str:
    return METRIC_SPECS.get(metric, {}).get("subtitle", "Mean across available paired runs; whiskers show 95% normal CI.")


def _metric_expected(metric: str) -> str:
    return METRIC_SPECS.get(metric, {}).get("expected", "Compare method ordering and confidence intervals before drawing conclusions.")


def _finite_group_points(rows: list[dict], x_key: str, metric: str) -> list[dict]:
    return [
        rec
        for rec in grouped_mean_ci(rows, ["mode", x_key], metric)
        if to_float(rec.get(x_key)) is not None and math.isfinite(float(rec["mean"]))
    ]


def _condition_rows(rows: list[dict], condition: dict[str, object]) -> list[dict]:
    return filter_rows(
        rows,
        modes=MAIN_METHODS,
        epsilon=condition.get("epsilon") if "epsilon" in condition else None,
        steps=condition.get("steps") if "steps" in condition else None,
        random_start=condition.get("random_start") if "random_start" in condition else None,
        paired=True,
    )


def plot_method_summary(
    rows: list[dict],
    metric: str,
    out_dir: Path,
    *,
    formats: list[str],
    dpi: int,
    condition: dict[str, object] | None = None,
    manifest: list[dict] | None = None,
    source: str = "full_metrics.csv",
) -> list[Path]:
    if not _has_metric(rows, metric):
        return []
    condition = condition or default_condition(rows)
    plot_rows = _condition_rows(rows, condition)
    if not plot_rows:
        plot_rows = filter_rows(rows, modes=MAIN_METHODS, paired=True)
    stats = grouped_mean_ci(plot_rows, ["mode"], metric)
    stats = [s for s in stats if str(s.get("mode")) in MAIN_METHODS]
    if not stats:
        return []
    modes = ordered_modes([str(s["mode"]) for s in stats], include_legacy=False)
    stat_by_mode = {str(s["mode"]): s for s in stats}
    y = np.arange(len(modes))
    means = [float(stat_by_mode[m]["mean"]) for m in modes]
    ci = [float(stat_by_mode[m]["ci95"]) for m in modes]
    fig, ax = plt.subplots(figsize=(7.2, max(3.7, 0.36 * len(modes) + 1.4)))
    for idx, mode in enumerate(modes):
        ax.errorbar(
            means[idx],
            y[idx],
            xerr=ci[idx],
            fmt="o",
            markersize=4.3,
            color=color_for_mode(mode),
            markerfacecolor=color_for_mode(mode),
            markeredgecolor=TOKENS["panel"],
            markeredgewidth=0.8,
            linewidth=1.0,
            capsize=3,
        )
    ax.set_yticks(y, [label_mode(m) for m in modes])
    ax.invert_yaxis()
    ax.set_xlabel(_metric_label(metric), fontsize=8.5, color=TOKENS["ink"])
    format_axis(ax, x_grid=True, y_grid=False)
    ax.xaxis.set_major_formatter(numeric_formatter())
    subtitle = f"{_metric_subtitle(metric)} Filter: {describe_condition(condition)}."
    add_chart_header(fig, ax, _metric_title(metric), subtitle)
    fig.subplots_adjust(left=0.22, right=0.975, top=0.78, bottom=0.12)
    figure_id = f"main_method_comparison_{slugify(metric)}"
    paths = save_figure(fig, out_dir / figure_id, formats=formats, dpi=dpi)
    plt.close(fig)
    if manifest is not None:
        add_manifest_records(
            manifest,
            figure_id=figure_id,
            paths=paths,
            source=source,
            family="Uncertainty & Benchmark",
            description=f"Method-level mean and 95% CI for {_metric_label(metric)}.",
            expected_effect=_metric_expected(metric),
        )
    return paths


def plot_metric_vs_epsilon(
    rows: list[dict],
    metric: str,
    out_dir: Path,
    *,
    formats: list[str],
    dpi: int,
    manifest: list[dict] | None = None,
    source: str = "full_metrics.csv",
) -> list[Path]:
    if not _has_metric(rows, metric):
        return []
    plot_rows = filter_rows(rows, modes=MAIN_METHODS, paired=True)
    stats = _finite_group_points(plot_rows, "epsilon", metric)
    if not stats:
        return []
    modes = ordered_modes([str(s["mode"]) for s in stats], include_legacy=False)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for mode in modes:
        points = sorted([s for s in stats if s["mode"] == mode], key=lambda s: float(s["epsilon"]))
        xs = [float(p["epsilon"]) for p in points]
        ys = [float(p["mean"]) for p in points]
        yerr = [float(p["ci95"]) for p in points]
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            color=color_for_mode(mode),
            linestyle=linestyle_for_mode(mode),
            marker="o",
            markersize=3.6,
            linewidth=1.15,
            capsize=2.5,
            label=label_mode(mode),
        )
    ax.set_xlabel("Epsilon", fontsize=8.5, color=TOKENS["ink"])
    ax.set_ylabel(_metric_label(metric), fontsize=8.5, color=TOKENS["ink"])
    ax.yaxis.set_major_formatter(numeric_formatter())
    format_axis(ax, x_grid=True, y_grid=True)
    add_chart_header(
        fig,
        ax,
        f"{_metric_label(metric)} vs perturbation budget",
        f"{_metric_subtitle(metric)} Curves aggregate available images, seeds, steps, and eval sigmas.",
    )
    bottom_legend(fig, mode_legend_handles(modes), ncol=5, y=0.004)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.79, bottom=0.24)
    figure_id = f"method_vs_epsilon_{slugify(metric)}"
    paths = save_figure(fig, out_dir / figure_id, formats=formats, dpi=dpi)
    plt.close(fig)
    if manifest is not None:
        add_manifest_records(
            manifest,
            figure_id=figure_id,
            paths=paths,
            source=source,
            family="Trend",
            description=f"{_metric_label(metric)} as epsilon changes.",
            expected_effect=_metric_expected(metric),
        )
    return paths


def plot_metric_vs_sigma(
    rows: list[dict],
    metric: str,
    out_dir: Path,
    *,
    formats: list[str],
    dpi: int,
    manifest: list[dict] | None = None,
    source: str = "full_metrics.csv",
) -> list[Path]:
    if not _has_metric(rows, metric):
        return []
    plot_rows = filter_rows(rows, modes=MAIN_METHODS, paired=True)
    stats = _finite_group_points(plot_rows, "sigma", metric)
    if not stats:
        return []
    modes = ordered_modes([str(s["mode"]) for s in stats], include_legacy=False)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for mode in modes:
        points = sorted([s for s in stats if s["mode"] == mode], key=lambda s: float(s["sigma"]))
        xs = [float(p["sigma"]) for p in points]
        ys = [float(p["mean"]) for p in points]
        yerr = [float(p["ci95"]) for p in points]
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            color=color_for_mode(mode),
            linestyle=linestyle_for_mode(mode),
            marker="o",
            markersize=3.6,
            linewidth=1.15,
            capsize=2.5,
            label=label_mode(mode),
        )
    ax.set_xlabel("SDEdit noise level sigma", fontsize=8.5, color=TOKENS["ink"])
    ax.set_ylabel(_metric_label(metric), fontsize=8.5, color=TOKENS["ink"])
    ax.yaxis.set_major_formatter(numeric_formatter())
    format_axis(ax, x_grid=True, y_grid=True)
    add_chart_header(
        fig,
        ax,
        f"{_metric_label(metric)} vs SDEdit noise level",
        f"{_metric_subtitle(metric)} FMP_multi should be smoother across eval sigmas than a single-sigma attack.",
    )
    bottom_legend(fig, mode_legend_handles(modes), ncol=5, y=0.004)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.79, bottom=0.24)
    figure_id = f"method_vs_sigma_{slugify(metric)}"
    paths = save_figure(fig, out_dir / figure_id, formats=formats, dpi=dpi)
    plt.close(fig)
    if manifest is not None:
        add_manifest_records(
            manifest,
            figure_id=figure_id,
            paths=paths,
            source=source,
            family="Trend",
            description=f"{_metric_label(metric)} as paired SDEdit sigma changes.",
            expected_effect=_metric_expected(metric),
        )
    return paths


def plot_tradeoff(
    rows: list[dict],
    out_dir: Path,
    *,
    formats: list[str],
    dpi: int,
    manifest: list[dict] | None = None,
    source: str = "full_metrics.csv",
) -> list[Path]:
    x_metric = "lpips_perturb" if _has_metric(rows, "lpips_perturb") else "l2_perturb_rmse"
    y_metric = "lpips_edit" if _has_metric(rows, "lpips_edit") else "l2_edit_rmse"
    if not _has_metric(rows, x_metric) or not _has_metric(rows, y_metric):
        return []
    plot_rows = filter_rows(rows, modes=MAIN_METHODS, paired=True)
    stats = grouped_mean_ci(plot_rows, ["mode", "epsilon"], y_metric)
    x_stats = grouped_mean_ci(plot_rows, ["mode", "epsilon"], x_metric)
    x_lookup = {(s["mode"], s["epsilon"]): float(s["mean"]) for s in x_stats}
    points = []
    for rec in stats:
        key = (rec["mode"], rec["epsilon"])
        if key in x_lookup:
            points.append({**rec, "x_mean": x_lookup[key]})
    if not points:
        return []
    fig, ax = plt.subplots(figsize=(7.4, 4.65))
    for mode in ordered_modes([str(p["mode"]) for p in points], include_legacy=False):
        part = sorted([p for p in points if p["mode"] == mode], key=lambda p: float(p["epsilon"]))
        xs = [float(p["x_mean"]) for p in part]
        ys = [float(p["mean"]) for p in part]
        ax.plot(
            xs,
            ys,
            color=color_for_mode(mode),
            linestyle=linestyle_for_mode(mode),
            marker="o",
            markersize=3.8,
            linewidth=1.05,
            label=label_mode(mode),
        )
        if xs and ys:
            ax.text(
                xs[-1],
                ys[-1],
                f" {label_mode(mode)}",
                fontsize=6.8,
                va="center",
                color=color_for_mode(mode),
            )
    ax.set_xlabel(_metric_label(x_metric), fontsize=8.5, color=TOKENS["ink"])
    ax.set_ylabel(_metric_label(y_metric), fontsize=8.5, color=TOKENS["ink"])
    ax.xaxis.set_major_formatter(numeric_formatter())
    ax.yaxis.set_major_formatter(numeric_formatter())
    format_axis(ax, x_grid=True, y_grid=True)
    add_chart_header(
        fig,
        ax,
        "Edit disruption versus perturbation visibility",
        "Each point is a method-epsilon mean; desirable points move upward with minimal rightward movement.",
    )
    fig.subplots_adjust(left=0.105, right=0.90, top=0.79, bottom=0.17)
    figure_id = "perturbation_quality_tradeoff"
    paths = save_figure(fig, out_dir / figure_id, formats=formats, dpi=dpi)
    plt.close(fig)
    if manifest is not None:
        add_manifest_records(
            manifest,
            figure_id=figure_id,
            paths=paths,
            source=source,
            family="Relationship",
            description="Tradeoff between perturbation visibility and edit disruption.",
            expected_effect="The strongest method should achieve higher edit LPIPS/RMSE without a disproportionate perturbation-quality penalty.",
        )
    return paths


def plot_paired_differences(
    paired_rows: list[dict],
    metric: str,
    out_dir: Path,
    *,
    formats: list[str],
    dpi: int,
    manifest: list[dict] | None = None,
    source: str = "paired_differences.csv",
) -> list[Path]:
    diff_metric = f"diff_{metric}"
    if not _has_metric(paired_rows, diff_metric):
        return []
    targets = ordered_modes([str(r.get("target_mode", "")) for r in paired_rows], include_legacy=False)
    rows = [r for r in paired_rows if r.get("target_mode") in targets and r.get("baseline_mode") in MAIN_METHODS]
    if not rows:
        return []
    groups = []
    labels = []
    colors = []
    for target in targets:
        for baseline in MAIN_METHODS:
            vals = [to_float(r.get(diff_metric)) for r in rows if r.get("target_mode") == target and r.get("baseline_mode") == baseline]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            groups.append(vals)
            labels.append(f"{label_mode(target)} - {label_mode(baseline)}")
            colors.append(color_for_mode(target))
    if not groups:
        return []
    fig, ax = plt.subplots(figsize=(8.2, max(3.6, 0.32 * len(groups) + 1.4)))
    y = np.arange(len(groups))
    for idx, vals in enumerate(groups):
        avg = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ci = 1.96 * sd / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
        ax.errorbar(avg, y[idx], xerr=ci, fmt="o", markersize=3.8, color=colors[idx], capsize=2.8, linewidth=1.0)
    ax.axvline(0, color=TOKENS["ink"], linestyle=":", linewidth=1.0)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"Delta {_metric_label(metric)}", fontsize=8.5, color=TOKENS["ink"])
    ax.xaxis.set_major_formatter(numeric_formatter())
    format_axis(ax, x_grid=True, y_grid=False)
    add_chart_header(
        fig,
        ax,
        f"Paired FMP lift over baselines: {_metric_label(metric)}",
        "Positive values mean the FMP variant produces stronger edit disruption than the matched baseline under identical image, seed, epsilon, steps, and sigma.",
    )
    fig.subplots_adjust(left=0.38, right=0.975, top=0.78, bottom=0.13)
    figure_id = f"paired_delta_{slugify(metric)}"
    paths = save_figure(fig, out_dir / figure_id, formats=formats, dpi=dpi)
    plt.close(fig)
    if manifest is not None:
        add_manifest_records(
            manifest,
            figure_id=figure_id,
            paths=paths,
            source=source,
            family="Uncertainty & Benchmark",
            description=f"Paired differences for {_metric_label(metric)}.",
            expected_effect="Most FMP-minus-baseline intervals should be positive for a successful protection objective.",
        )
    return paths


def generate_metric_figures(
    *,
    metrics_path: str | Path,
    paired_differences_path: str | Path | None,
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
        print(f"No metrics rows found: {metrics_path}")
        return []
    saved: list[Path] = []
    condition = default_condition(rows)
    for metric in SUMMARY_METRICS:
        saved.extend(
            plot_method_summary(
                rows,
                metric,
                out_dir,
                formats=formats,
                dpi=dpi,
                condition=condition,
                manifest=manifest,
                source=str(metrics_path),
            )
        )
    for metric in EPSILON_METRICS:
        saved.extend(
            plot_metric_vs_epsilon(
                rows,
                metric,
                out_dir,
                formats=formats,
                dpi=dpi,
                manifest=manifest,
                source=str(metrics_path),
            )
        )
    for metric in SIGMA_METRICS:
        saved.extend(
            plot_metric_vs_sigma(
                rows,
                metric,
                out_dir,
                formats=formats,
                dpi=dpi,
                manifest=manifest,
                source=str(metrics_path),
            )
        )
    saved.extend(plot_tradeoff(rows, out_dir, formats=formats, dpi=dpi, manifest=manifest, source=str(metrics_path)))
    paired = add_derived_metrics(read_csv_rows(paired_differences_path))
    for metric in ["l2_edit_rmse", "lpips_edit", "source_similarity_drop", "clip_prompt_drop"]:
        saved.extend(
            plot_paired_differences(
                paired,
                metric,
                out_dir,
                formats=formats,
                dpi=dpi,
                manifest=manifest,
                source=str(paired_differences_path),
            )
        )
    for path in saved:
        print(f"Saved: {path}")
    return saved

