#!/usr/bin/env python3
"""Aggregate SD3 v3 metrics into analysis tables."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Tuple


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

PAIR_METRICS = [
    "l2_edit_rmse",
    "psnr_edit",
    "ssim_edit",
    "lpips_edit",
    "delta_clip_prompt",
    "delta_clip_src",
    "linf_perturb_tensor_-1_1",
    "l2_perturb_rmse",
    "psnr_perturb",
    "ssim_perturb",
    "loss_total_final",
    "loss_fmp_final",
    "loss_o_final",
    "loss_textual_mse_final",
    "loss_mechanism_final",
    "loss_step_final",
    "loss_transition_sep_final",
    "loss_grad_step_l2_final",
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()}) or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value):
    try:
        if value is None or value == "":
            return None
        value = float(value)
        if math.isnan(value):
            return None
        return value
    except Exception:
        return None


def _numeric_keys(rows: List[Dict[str, str]]) -> List[str]:
    keys = set()
    for row in rows:
        for key, value in row.items():
            if _to_float(value) is not None:
                keys.add(key)
    return sorted(keys)


def _group_rows(rows: Iterable[Dict[str, str]], keys: Iterable[str]):
    buckets = defaultdict(list)
    keys = [k for k in keys if any(k in row for row in rows)]
    for row in rows:
        buckets[tuple(row.get(k, "") for k in keys)].append(row)
    return keys, buckets


def _summarize(rows: List[Dict[str, str]], group_keys: List[str], metrics: List[str]) -> List[Dict[str, object]]:
    existing_group_keys, buckets = _group_rows(rows, group_keys)
    out = []
    for group_value, group_rows in buckets.items():
        rec: Dict[str, object] = {k: v for k, v in zip(existing_group_keys, group_value)}
        rec["n_rows"] = len(group_rows)
        for metric in metrics:
            vals = [_to_float(row.get(metric)) for row in group_rows]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            rec[f"{metric}_mean"] = mean(vals)
            rec[f"{metric}_std"] = stdev(vals) if len(vals) > 1 else 0.0
            rec[f"{metric}_count"] = len(vals)
        out.append(rec)
    return sorted(out, key=lambda r: tuple(str(r.get(k, "")) for k in existing_group_keys))


def _paired_key(row: Dict[str, str]) -> Tuple[str, ...]:
    keys = ["image_id", "epsilon", "steps", "random_start", "seed", "sigma", "paired"]
    return tuple(row.get(k, "") for k in keys)


def _paired_differences(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    by_key = defaultdict(dict)
    for row in rows:
        mode = row.get("mode", "")
        if mode:
            by_key[_paired_key(row)][mode] = row

    out: List[Dict[str, object]] = []
    for key, by_mode in by_key.items():
        for target_mode in ["FMP_single", "FMP_multi", "FMP_single_plus_step"]:
            target = by_mode.get(target_mode)
            if target is None:
                continue
            for baseline_mode in MODE_ORDER:
                if baseline_mode == target_mode or baseline_mode not in by_mode:
                    continue
                base = by_mode[baseline_mode]
                rec: Dict[str, object] = {
                    "target_mode": target_mode,
                    "baseline_mode": baseline_mode,
                    "image_id": key[0],
                    "epsilon": key[1],
                    "steps": key[2],
                    "random_start": key[3],
                    "seed": key[4],
                    "sigma": key[5],
                    "paired": key[6],
                }
                has_metric = False
                for metric in PAIR_METRICS:
                    a = _to_float(target.get(metric))
                    b = _to_float(base.get(metric))
                    if a is None or b is None:
                        continue
                    rec[f"diff_{metric}"] = a - b
                    has_metric = True
                if has_metric:
                    out.append(rec)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="out_sd3_v3/full_metrics.csv")
    parser.add_argument("--out-dir", default="out_sd3_v3/analysis")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    out_dir = Path(args.out_dir)
    rows = _read_csv(metrics_path)
    if not rows:
        raise SystemExit(f"No metrics rows found: {metrics_path}")

    numeric = _numeric_keys(rows)
    summary_metrics = [m for m in PAIR_METRICS if m in numeric]
    if not summary_metrics:
        summary_metrics = numeric

    _write_csv(rows, out_dir / "full_metrics.csv")
    _write_csv(_summarize(rows, ["mode"], summary_metrics), out_dir / "summary_by_method.csv")
    _write_csv(_paired_differences(rows), out_dir / "paired_differences.csv")
    _write_csv(
        _summarize(rows, ["mode", "epsilon", "steps", "random_start", "sigma"], summary_metrics),
        out_dir / "ablation_summary.csv",
    )
    print(f"analysis tables written to {out_dir}")


if __name__ == "__main__":
    main()
