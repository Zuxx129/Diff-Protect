#!/usr/bin/env python3
"""Generate random L_inf perturbation baselines.

This baseline is not an attack. It creates imperceptibility-matched random
perturbations for thresholding and sanity comparisons.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _iter_images(root: Path):
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        yield from root.rglob(ext)


def _perturb(arr: np.ndarray, eps: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    eps01 = eps / 255.0
    if mode == "uniform":
        noise = rng.uniform(-eps01, eps01, size=arr.shape)
    elif mode == "gaussian":
        noise = rng.normal(0.0, eps01 / 2.0, size=arr.shape)
        noise = np.clip(noise, -eps01, eps01)
    else:
        raise ValueError(f"unknown mode: {mode}")
    return np.clip(arr + noise, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test_images/to_protect")
    parser.add_argument("--output", default="out_sd3/Random_Linf")
    parser.add_argument("--epsilon", type=int, default=8)
    parser.add_argument("--mode", choices=["uniform", "gaussian"], default="uniform")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-exp-num", type=int, default=100)
    args = parser.parse_args()

    inp = Path(args.input)
    out_root = Path(args.output) / f"Random_Linf_eps{args.epsilon}_{args.mode}_seed{args.seed}"
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows = []
    for idx, path in enumerate(sorted(_iter_images(inp))):
        if idx >= args.max_exp_num:
            break
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
        adv = _perturb(arr, args.epsilon, args.mode, rng)
        rel_dir = path.parent.name
        name = path.stem
        save_dir = out_root / rel_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{name}_attacked.png"
        Image.fromarray(np.uint8(np.round(adv * 255.0))).save(out_path)
        rows.append({"source": str(path), "attacked": str(out_path), "epsilon": args.epsilon, "mode": args.mode, "seed": args.seed})
    manifest = out_root / "manifest.json"
    manifest.write_text(json.dumps({"count": len(rows), "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out_root), "count": len(rows), "manifest": str(manifest)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
