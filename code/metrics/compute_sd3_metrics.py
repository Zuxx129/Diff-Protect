#!/usr/bin/env python3
"""Compute paired SD3 experiment metrics.

This script is dependency-light by default. Optional CLIP and LPIPS metrics are
computed only when their packages are installed and explicitly enabled.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def _load_rgb(path: Path, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.BICUBIC)
    return np.asarray(img).astype(np.float32) / 255.0


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = _mse(a, b)
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def _linf(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _ssim_fallback(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception:
        return None
    return float(ssim(a, b, channel_axis=2, data_range=1.0))


def _try_clip_scores(prompt: str, image_path: Path) -> Optional[float]:
    try:
        import torch
        import torch.nn.functional as F
        from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
    except Exception:
        return None
    model_id = "openai/clip-vit-base-patch32"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPImageProcessor.from_pretrained(model_id)
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    text = tokenizer([prompt], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        im = F.normalize(model.get_image_features(pixel_values), dim=-1)
        tx = F.normalize(model.get_text_features(**text), dim=-1)
    return float((im * tx).sum(dim=-1).item())


def _try_clip_image_sim(a_path: Path, b_path: Path) -> Optional[float]:
    try:
        import torch
        import torch.nn.functional as F
        from transformers import CLIPImageProcessor, CLIPModel
    except Exception:
        return None
    model_id = "openai/clip-vit-base-patch32"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPImageProcessor.from_pretrained(model_id)
    images = [Image.open(a_path).convert("RGB"), Image.open(b_path).convert("RGB")]
    pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
    with torch.inference_mode():
        emb = F.normalize(model.get_image_features(pixel_values), dim=-1)
    return float((emb[0] * emb[1]).sum().item())


def _try_lpips(a_path: Path, b_path: Path) -> Optional[float]:
    try:
        import torch
        import lpips
        import torchvision.transforms as T
    except Exception:
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net="alex").to(device).eval()
    tfm = T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    a = tfm(Image.open(a_path).convert("RGB")).unsqueeze(0).to(device)
    b = tfm(Image.open(b_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        return float(loss_fn(a, b).item())


def _parse_exp_dir(path: Path) -> Dict[str, str]:
    name = path.name
    out = {"exp_dir": str(path), "exp_name": name}
    patterns = {
        "mode": r"^(O_repo|O_fair|O|A|B|C|D)",
        "epsilon": r"eps(\d+)",
        "steps": r"steps(\d+)",
        "g_mode": r"gmode([+-])",
        "opt_direction": r"opt(maximize|minimize)",
        "textual_objective": r"text([^_]+)",
        "seed": r"seed(\d+)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, name)
        if m:
            out[key] = m.group(1)
    return out


def _find_pairs(exp_dir: Path):
    attacked = list(exp_dir.rglob("*_attacked.png"))
    for adv_path in attacked:
        stem = adv_path.name.replace("_attacked.png", "")
        original_guess = None
        for cand in Path("test_images/to_protect").rglob(f"{stem}.*"):
            original_guess = cand
            break
        loss_npz = adv_path.with_name(stem + "_loss.npz")
        sdedit = sorted(adv_path.parent.glob(stem + "_sdedit_*.png"))
        yield stem, original_guess, adv_path, loss_npz if loss_npz.exists() else None, sdedit


def compute_metrics(root: Path, prompt: str, enable_clip: bool, enable_lpips: bool) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    exp_dirs = [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
    for exp in sorted(exp_dirs):
        base = _parse_exp_dir(exp)
        for stem, original_path, attacked_path, loss_path, sdedit_paths in _find_pairs(exp):
            row: Dict[str, object] = dict(base)
            row.update({"image_id": stem, "attacked_path": str(attacked_path)})
            if original_path and original_path.exists():
                row["original_path"] = str(original_path)
                a = _load_rgb(attacked_path)
                o = _load_rgb(original_path, size=Image.open(attacked_path).size)
                row["linf_perturb_0_1"] = _linf(a, o)
                row["l2_perturb_rmse"] = _l2(a, o)
                row["psnr_perturb"] = _psnr(a, o)
                ssim = _ssim_fallback(a, o)
                if ssim is not None:
                    row["ssim_perturb"] = ssim
                if enable_lpips:
                    lp = _try_lpips(original_path, attacked_path)
                    if lp is not None:
                        row["lpips_perturb"] = lp
                if enable_clip:
                    sim = _try_clip_image_sim(original_path, attacked_path)
                    if sim is not None:
                        row["clip_img_original_attacked"] = sim
            if loss_path:
                row["loss_path"] = str(loss_path)
                try:
                    data = np.load(loss_path)
                    for key in ["total", "textual", "mmdit"]:
                        if key in data and len(data[key]) > 0:
                            row[f"loss_{key}_first"] = float(data[key][0])
                            row[f"loss_{key}_final"] = float(data[key][-1])
                except Exception as exc:
                    row["loss_error"] = str(exc)
            for sd in sdedit_paths:
                sub = dict(row)
                sub["sdedit_path"] = str(sd)
                sub["sdedit_name"] = sd.name
                if enable_clip:
                    cs = _try_clip_scores(prompt, sd)
                    if cs is not None:
                        sub["clip_text_image"] = cs
                rows.append(sub)
            if not sdedit_paths:
                rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()}) or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="out_sd3")
    parser.add_argument("--prompt", default="a photo")
    parser.add_argument("--out", default=None)
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--lpips", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.out) if args.out else root / "metrics.csv"
    rows = compute_metrics(root, args.prompt, args.clip, args.lpips)
    write_csv(rows, out)
    summary = {"root": str(root), "rows": len(rows), "out": str(out), "clip": args.clip, "lpips": args.lpips}
    (out.parent / "metrics_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
