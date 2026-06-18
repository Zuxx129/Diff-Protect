"""Data loading and experiment-directory parsing for SD3 figures."""
from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable, Sequence

import numpy as np


MODE_RE = (
    r"^(textual_only|FMP_single_plus_step|FMP_single|FMP_multi|"
    r"textual_semantic_joint|O_repo|O_fair|Random_Linf|E|O|A|B|C|D)"
)

SKIP_EXPERIMENT_DIRS = {
    "figures",
    "paper_figures",
    "analysis",
    "experiment_logs",
    "validation_logs",
    "minimal_logs",
    "minimal_rs_false_logs",
    "minimal_rs_true_logs",
    "direction_logs",
    "full_logs",
    "step_ablation_logs",
    "__pycache__",
}


def read_csv_rows(path: Path | str | None) -> list[dict[str, str]]:
    if not path:
        return []
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        if math.isnan(out):
            return None
        return out
    except Exception:
        return None


def to_int(value) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def numeric_values(rows: Iterable[dict], key: str) -> list[float]:
    vals = []
    for row in rows:
        val = to_float(row.get(key))
        if val is not None and math.isfinite(val):
            vals.append(val)
    return vals


def row_has_metric(row: dict, key: str) -> bool:
    return to_float(row.get(key)) is not None


def add_derived_metrics(rows: Sequence[dict]) -> list[dict]:
    out = []
    for row in rows:
        rec = dict(row)
        prompt_delta = to_float(row.get("delta_clip_prompt"))
        if prompt_delta is not None:
            rec["clip_prompt_drop"] = -prompt_delta
        src_delta = to_float(row.get("delta_clip_src"))
        if src_delta is not None:
            rec["source_similarity_drop"] = -src_delta
        diff_prompt_delta = to_float(row.get("diff_delta_clip_prompt"))
        if diff_prompt_delta is not None:
            rec["diff_clip_prompt_drop"] = -diff_prompt_delta
        diff_src_delta = to_float(row.get("diff_delta_clip_src"))
        if diff_src_delta is not None:
            rec["diff_source_similarity_drop"] = -diff_src_delta
        out.append(rec)
    return out


def filter_rows(
    rows: Sequence[dict],
    *,
    modes: Sequence[str] | None = None,
    epsilon: float | None = None,
    steps: float | None = None,
    random_start: str | None = None,
    paired: bool | None = None,
) -> list[dict]:
    allowed = set(modes or [])
    out = []
    for row in rows:
        if allowed and row.get("mode") not in allowed:
            continue
        if epsilon is not None and to_float(row.get("epsilon")) != float(epsilon):
            continue
        if steps is not None and to_float(row.get("steps")) != float(steps):
            continue
        if random_start is not None and str(row.get("random_start", "")).lower() != str(random_start).lower():
            continue
        if paired is not None:
            value = str(row.get("paired", "")).lower()
            actual = value in {"true", "1", "yes"}
            if actual != paired:
                continue
        out.append(row)
    return out


def default_condition(rows: Sequence[dict]) -> dict[str, object]:
    eps = sorted({v for v in (to_float(r.get("epsilon")) for r in rows) if v is not None})
    steps = sorted({v for v in (to_float(r.get("steps")) for r in rows) if v is not None})
    random_values = {str(r.get("random_start", "")).lower() for r in rows if r.get("random_start", "") != ""}
    condition: dict[str, object] = {}
    if eps:
        condition["epsilon"] = eps[-1]
    if steps:
        condition["steps"] = steps[-1]
    if "true" in random_values:
        condition["random_start"] = "true"
    return condition


def describe_condition(condition: dict[str, object]) -> str:
    parts = []
    if "epsilon" in condition:
        parts.append(f"epsilon={condition['epsilon']:g}")
    if "steps" in condition:
        parts.append(f"steps={condition['steps']:g}")
    if "random_start" in condition:
        parts.append(f"random_start={condition['random_start']}")
    return ", ".join(parts) if parts else "all available runs"


def grouped_mean_ci(rows: Sequence[dict], group_keys: Sequence[str], metric: str) -> list[dict[str, object]]:
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        val = to_float(row.get(metric))
        if val is None:
            continue
        group = tuple(row.get(key, "") for key in group_keys)
        buckets[group].append(val)
    out = []
    for group, vals in buckets.items():
        if not vals:
            continue
        n = len(vals)
        avg = mean(vals)
        sd = stdev(vals) if n > 1 else 0.0
        ci = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
        rec = {key: group[i] for i, key in enumerate(group_keys)}
        rec.update({"mean": avg, "std": sd, "ci95": ci, "n": n})
        out.append(rec)
    return out


def parse_exp_dirname(dirname: str) -> dict[str, object]:
    mode_match = re.search(MODE_RE, dirname)
    cfg: dict[str, object] = {"mode": mode_match.group(1) if mode_match else dirname.split("_")[0]}
    if cfg["mode"] == "O_repo":
        cfg["mode"] = "O_repo_legacy"
    elif cfg["mode"] == "O_fair":
        cfg["mode"] = "O_fair_legacy"
    elif cfg["mode"] == "textual_semantic_joint":
        cfg["mode"] = "textual_semantic_joint_legacy"
    patterns = {
        "epsilon": (r"eps(\d+)", int),
        "steps": (r"steps(\d+)", int),
        "g_mode": (r"gmode([+-])", str),
        "opt_direction": (r"opt(maximize|minimize)", str),
        "textual_objective": (r"_text(.+?)(?:_tw|_mw|_seed|$)", str),
        "textual_weight": (r"tw([\d.]+)", float),
        "mmdit_weight": (r"mw([\d.]+)", float),
        "random_start": (r"rs(true|false)", str),
        "seed": (r"seed(\d+)", int),
    }
    for key, (pattern, typ) in patterns.items():
        match = re.search(pattern, dirname)
        if match:
            cfg[key] = typ(match.group(1))
    return cfg


def load_loss_data(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        out = {key: np.asarray(data[key], dtype=float).reshape(-1) for key in data.files}
        if "total" not in out and "loss_total" in out:
            out["total"] = out["loss_total"]
        if "loss_total" not in out and "total" in out:
            out["loss_total"] = out["total"]
        total = out.get("total", np.array([], dtype=float))
        out.setdefault("textual", np.zeros_like(total))
        out.setdefault("mmdit", np.zeros_like(total))
        return out
    arr = np.load(path).astype(float).reshape(-1)
    return {"total": arr, "loss_total": arr, "textual": np.zeros_like(arr), "mmdit": np.zeros_like(arr)}


def _loss_file_priority(cfg: dict, loss_path: Path) -> tuple:
    seed = to_int(cfg.get("seed"))
    return (
        1 if loss_path.suffix == ".npz" else 0,
        to_int(cfg.get("steps")) or -1,
        to_int(cfg.get("epsilon")) or -1,
        1 if seed is None else 0,
        -abs(seed or 0),
        str(loss_path),
    )


def find_loss_files(root: str | Path, image_name: str | None = None, *, representative: bool = True) -> list[tuple[dict, str, Path]]:
    root = Path(root)
    if not root.exists():
        return []
    results: list[tuple[dict, str, Path]] = []
    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name in SKIP_EXPERIMENT_DIRS:
            continue
        cfg = parse_exp_dirname(exp_dir.name)
        patterns = (
            [f"**/{image_name}_loss.npz", f"**/{image_name}_loss.npy"]
            if image_name
            else ["**/*_loss.npz", "**/*_loss.npy"]
        )
        for pattern in patterns:
            for loss_path in sorted(exp_dir.glob(pattern)):
                stem = re.sub(r"_loss\.(npz|npy)$", "", loss_path.name)
                results.append((cfg, stem, loss_path))
    if not representative:
        return results
    seen: dict[tuple, tuple[dict, str, Path]] = {}
    priorities: dict[tuple, tuple] = {}
    for cfg, stem, path in results:
        key = (cfg.get("mode"), stem)
        priority = _loss_file_priority(cfg, path)
        if key not in seen or priority > priorities[key]:
            seen[key] = (cfg, stem, path)
            priorities[key] = priority
    return list(seen.values())


def smooth_array(arr: np.ndarray, window: int | None) -> np.ndarray:
    if window and window > 1 and len(arr) >= window:
        return np.convolve(arr, np.ones(window) / window, mode="valid")
    return arr


def normalize_array(arr: np.ndarray) -> np.ndarray:
    if len(arr) == 0:
        return arr
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def original_for(stem: str, root: Path | str = "test_images/to_protect") -> Path | None:
    root = Path(root)
    if not root.exists():
        return None
    for cand in root.rglob(f"{stem}.*"):
        return cand
    return None


def noise_key(path: Path, prefix: str) -> str | None:
    match = re.search(rf"_sdedit_{prefix}_noise_([0-9.]+)\.png$", path.name)
    return match.group(1) if match else None


def find_image_samples(root: str | Path) -> list[dict[str, object]]:
    root = Path(root)
    if not root.exists():
        return []
    samples: list[dict[str, object]] = []
    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name in SKIP_EXPERIMENT_DIRS:
            continue
        cfg = parse_exp_dirname(exp_dir.name)
        for adv_path in exp_dir.rglob("*_attacked.png"):
            stem = adv_path.name.replace("_attacked.png", "")
            clean = {noise_key(p, "clean"): p for p in adv_path.parent.glob(stem + "_sdedit_clean_noise_*.png")}
            adv = {noise_key(p, "adv"): p for p in adv_path.parent.glob(stem + "_sdedit_adv_noise_*.png")}
            clean.pop(None, None)
            adv.pop(None, None)
            sigmas = sorted(set(clean.keys()) & set(adv.keys()), key=lambda s: float(s))
            if not sigmas:
                samples.append(
                    {
                        **cfg,
                        "image_id": stem,
                        "attacked_path": adv_path,
                        "original_path": original_for(stem),
                        "sigma": "",
                        "clean_edit_path": None,
                        "adv_edit_path": None,
                    }
                )
                continue
            for sigma in sigmas:
                samples.append(
                    {
                        **cfg,
                        "image_id": stem,
                        "attacked_path": adv_path,
                        "original_path": original_for(stem),
                        "sigma": sigma,
                        "clean_edit_path": clean[sigma],
                        "adv_edit_path": adv[sigma],
                    }
                )
    return samples
