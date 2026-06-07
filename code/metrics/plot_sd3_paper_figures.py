#!/usr/bin/env python3
"""Generate paper-style figures from SD3 metrics/summary CSV files.

The script is intentionally matplotlib-only and works without seaborn. It reads
row-level metrics and/or aggregated summaries and produces common figures:

- success_rate vs epsilon by mode
- LPIPS edit vs epsilon by mode
- delta CLIPScore vs epsilon by mode
- mechanism diagnostic final values by mode
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODE_ORDER = ["Random_Linf", "O_repo", "O_fair", "A", "B", "C", "D"]
MECH_KEYS = [
    "loss_attn_injection_l2_final",
    "loss_feature_cos_final",
    "loss_velocity_div_final",
    "loss_modality_ratio_dev_final",
    "loss_cross_modal_cka_final",
]


def _read_csv(path: Path):
    if not path or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _group_mean(rows, x_key, y_key, mode_key="mode"):
    buckets = defaultdict(list)
    for row in rows:
        x = _to_float(row.get(x_key))
        y = _to_float(row.get(y_key))
        mode = row.get(mode_key, "unknown")
        if x is None or y is None:
            continue
        buckets[(mode, x)].append(y)
    series = defaultdict(list)
    for (mode, x), vals in buckets.items():
        series[mode].append((x, mean(vals)))
    return {m: sorted(v) for m, v in series.items()}


def _plot_lines(series, ylabel, title, out):
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for mode in MODE_ORDER:
        if mode not in series:
            continue
        xs = [p[0] for p in series[mode]]
        ys = [p[1] for p in series[mode]]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=mode)
    for mode, points in series.items():
        if mode in MODE_ORDER:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", linewidth=1.2, label=mode)
    ax.set_xlabel("epsilon")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"saved {out}")


def _plot_mechanisms(rows, out_dir: Path):
    for key in MECH_KEYS:
        series = _group_mean(rows, "epsilon", key)
        if not series:
            continue
        _plot_lines(series, key, f"Mechanism diagnostic: {key}", out_dir / f"mechanism_{key}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="out_sd3/full_metrics.csv")
    parser.add_argument("--summary", default="out_sd3/full_summary.csv")
    parser.add_argument("--out-dir", default="out_sd3/paper_figures")
    args = parser.parse_args()

    metrics = _read_csv(Path(args.metrics))
    summary = _read_csv(Path(args.summary))
    out_dir = Path(args.out_dir)

    # Prefer summary for success rate. Use metrics for raw effect/diagnostic curves.
    if summary:
        series = _group_mean(summary, "epsilon", "success_rate")
        if series:
            _plot_lines(series, "success rate", "Success rate vs epsilon", out_dir / "success_rate_vs_epsilon.png")

    if metrics:
        for key, ylabel, title, name in [
            ("lpips_edit", "LPIPS(clean_edit, adv_edit)", "Edit LPIPS vs epsilon", "lpips_edit_vs_epsilon.png"),
            ("delta_clip_prompt", "Delta CLIPScore", "Prompt CLIPScore drop vs epsilon", "delta_clip_prompt_vs_epsilon.png"),
            ("l2_edit_rmse", "edit RMSE", "Edit RMSE vs epsilon", "edit_rmse_vs_epsilon.png"),
            ("linf_perturb_tensor_-1_1", "L_inf tensor", "Perturbation L_inf vs epsilon", "linf_vs_epsilon.png"),
        ]:
            series = _group_mean(metrics, "epsilon", key)
            if series:
                _plot_lines(series, ylabel, title, out_dir / name)
        _plot_mechanisms(metrics, out_dir)

    print(f"figures written to {out_dir}")


if __name__ == "__main__":
    main()
