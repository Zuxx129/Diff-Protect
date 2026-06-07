#!/usr/bin/env python3
"""Plot Diff-Protect SD3 loss curves.

Supports legacy and v2 directory names:
  A_eps16_steps100_gmode+_tw1.0_mw1.0
  C_eps8_steps2_gmode+_optmaximize_texttoward_target_tw1.0_mw1.0_seed0
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODE_ORDER = ["O", "O_repo", "O_fair", "A", "B", "C", "D"]
MODE_LABELS = {
    "O": "O: legacy baseline",
    "O_repo": "O_repo: repo baseline",
    "O_fair": "O_fair: velocity baseline",
    "A": "A: cross-modal",
    "B": "B: feature shift",
    "C": "C: temporal break",
    "D": "D: modality imbalance",
}
COMPONENT_LABELS = {
    "total": "Total loss",
    "textual": "Textual loss",
    "mmdit": "MMDiT loss",
}


def parse_dirname(dirname: str) -> dict:
    cfg = {"mode": dirname.split("_")[0]}
    patterns = {
        "epsilon": (r"eps(\d+)", int),
        "steps": (r"steps(\d+)", int),
        "g_mode": (r"gmode([+-])", str),
        "opt_direction": (r"opt(maximize|minimize)", str),
        "textual_objective": (r"text([^_]+)", str),
        "textual_weight": (r"tw([\d.]+)", float),
        "mmdit_weight": (r"mw([\d.]+)", float),
        "seed": (r"seed(\d+)", int),
    }
    for key, (pattern, typ) in patterns.items():
        match = re.search(pattern, dirname)
        if match:
            cfg[key] = typ(match.group(1))
    return cfg


def load_loss_data(path: str) -> dict:
    if path.endswith(".npz"):
        data = np.load(path)
        total = np.asarray(data["total"]) if "total" in data else np.array([])
        return {
            "total": total,
            "textual": np.asarray(data["textual"]) if "textual" in data else np.zeros_like(total),
            "mmdit": np.asarray(data["mmdit"]) if "mmdit" in data else np.zeros_like(total),
        }
    arr = np.load(path)
    return {"total": arr, "textual": np.zeros_like(arr), "mmdit": np.zeros_like(arr)}


def find_loss_files(root: str, image_name: str | None = None) -> list:
    results = []
    root_path = Path(root)
    for exp_dir in sorted(root_path.iterdir() if root_path.exists() else []):
        if not exp_dir.is_dir() or exp_dir.name == "figures":
            continue
        cfg = parse_dirname(exp_dir.name)
        patterns = [f"**/{image_name}_loss.npz", f"**/{image_name}_loss.npy"] if image_name else ["**/*_loss.npz", "**/*_loss.npy"]
        for pattern in patterns:
            for loss_path in sorted(exp_dir.glob(pattern)):
                img_name = re.sub(r"_loss\.(npz|npy)$", "", loss_path.name)
                results.append((cfg, img_name, str(loss_path)))
    return results


def smooth(arr: np.ndarray, window: int | None) -> np.ndarray:
    if window and window > 1 and len(arr) >= window:
        return np.convolve(arr, np.ones(window) / window, mode="valid")
    return arr


def normalize(arr: np.ndarray) -> np.ndarray:
    if len(arr) == 0:
        return arr
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def label_for(cfg: dict) -> str:
    mode = cfg.get("mode", "?")
    parts = [MODE_LABELS.get(mode, mode)]
    if "g_mode" in cfg:
        parts.append(f"g={cfg['g_mode']}")
    if "opt_direction" in cfg:
        parts.append(f"opt={cfg['opt_direction']}")
    if "seed" in cfg:
        parts.append(f"seed={cfg['seed']}")
    return " | ".join(parts)


def plot_modes(grouped: dict, output_dir: Path, args) -> None:
    for image, curves in grouped.items():
        curves = sorted(curves, key=lambda item: MODE_ORDER.index(item[0].get("mode", "D")) if item[0].get("mode") in MODE_ORDER else 99)
        fig, ax = plt.subplots(figsize=(9, 5))
        for cfg, data in curves:
            y = smooth(data["total"], args.smooth)
            if args.normalize:
                y = normalize(y)
            ax.plot(np.arange(len(y)), y, linewidth=1.7, label=label_for(cfg))
        ax.set_title(f"{image} — total loss")
        ax.set_xlabel("PGD step")
        ax.set_ylabel("normalized loss" if args.normalize else "loss")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = output_dir / f"{image}_loss_modes.png"
        fig.savefig(out, dpi=args.dpi)
        plt.close(fig)
        print(f"Saved: {out}")


def plot_components(grouped: dict, output_dir: Path, args) -> None:
    for image, curves in grouped.items():
        for cfg, data in curves:
            mode = cfg.get("mode", "unknown")
            fig, ax = plt.subplots(figsize=(8, 5))
            for comp in ["total", "textual", "mmdit"]:
                y = smooth(data[comp], args.smooth)
                if args.normalize:
                    y = normalize(y)
                ax.plot(np.arange(len(y)), y, linewidth=1.6, label=COMPONENT_LABELS[comp])
            ax.set_title(f"{image} — {label_for(cfg)}")
            ax.set_xlabel("PGD step")
            ax.set_ylabel("normalized loss" if args.normalize else "loss")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            safe = re.sub(r"[^A-Za-z0-9_.+-]+", "_", f"{image}_{mode}_{cfg.get('g_mode','')}_{cfg.get('opt_direction','')}")
            out = output_dir / f"{safe}_loss_components.png"
            fig.savefig(out, dpi=args.dpi)
            plt.close(fig)
            print(f"Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="out_sd3")
    parser.add_argument("--image", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--modes", nargs="*", default=None)
    parser.add_argument("--gmodes", nargs="*", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--smooth", type=int, default=None)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--components", action="store_true")
    args = parser.parse_args()

    if not args.all and not args.image:
        raise SystemExit("Specify --all or --image NAME")
    output_dir = Path(args.output or Path(args.root) / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = find_loss_files(args.root, args.image)
    grouped = {}
    for cfg, image, loss_path in entries:
        if args.modes and cfg.get("mode") not in args.modes:
            continue
        if args.gmodes and cfg.get("g_mode", "+") not in args.gmodes:
            continue
        try:
            data = load_loss_data(loss_path)
        except Exception as exc:
            print(f"Skip {loss_path}: {exc}")
            continue
        grouped.setdefault(image, []).append((cfg, data))

    if not grouped:
        print("No loss files found after filtering")
        return
    plot_modes(grouped, output_dir, args)
    if args.components:
        plot_components(grouped, output_dir, args)


if __name__ == "__main__":
    main()
