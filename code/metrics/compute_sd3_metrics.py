#!/usr/bin/env python3
"""Compute SD3 experiment metrics.

The script supports both current adv-only outputs and paired SDEdit outputs:
  *_sdedit_noise_0.3.png
  *_sdedit_clean_noise_0.3.png
  *_sdedit_adv_noise_0.3.png

Optional CLIP and LPIPS metrics are computed only when requested and available.
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


_CLIP_MODEL = None
_CLIP_PROCESSOR = None
_CLIP_TOKENIZER = None
_LPIPS_MODEL = None


def _clip_setup():
    global _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_TOKENIZER
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_TOKENIZER
    import torch
    from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
    model_id = "openai/clip-vit-base-patch32"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _CLIP_MODEL = CLIPModel.from_pretrained(model_id).to(device).eval()
    _CLIP_PROCESSOR = CLIPImageProcessor.from_pretrained(model_id)
    _CLIP_TOKENIZER = CLIPTokenizer.from_pretrained(model_id)
    return _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_TOKENIZER


def _try_clip_scores(prompt: str, image_path: Path) -> Optional[float]:
    try:
        import torch
        import torch.nn.functional as F
        model, processor, tokenizer = _clip_setup()
    except Exception:
        return None
    device = next(model.parameters()).device
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
        model, processor, _ = _clip_setup()
    except Exception:
        return None
    device = next(model.parameters()).device
    images = [Image.open(a_path).convert("RGB"), Image.open(b_path).convert("RGB")]
    pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
    with torch.inference_mode():
        emb = F.normalize(model.get_image_features(pixel_values), dim=-1)
    return float((emb[0] * emb[1]).sum().item())


def _try_lpips(a_path: Path, b_path: Path) -> Optional[float]:
    global _LPIPS_MODEL
    try:
        import torch
        import lpips
        import torchvision.transforms as T
    except Exception:
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _LPIPS_MODEL is None:
        _LPIPS_MODEL = lpips.LPIPS(net="alex").to(device).eval()
    tfm = T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    a = tfm(Image.open(a_path).convert("RGB")).unsqueeze(0).to(device)
    b = tfm(Image.open(b_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        return float(_LPIPS_MODEL(a, b).item())


def _parse_exp_dir(path: Path) -> Dict[str, object]:
    name = path.name
    out: Dict[str, object] = {"exp_dir": str(path), "exp_name": name}
    if name.startswith("Random_Linf"):
        out["mode"] = "Random_Linf"
        m = re.search(r"eps(\d+)", name)
        if m:
            out["epsilon"] = m.group(1)
        m = re.search(r"seed(\d+)", name)
        if m:
            out["seed"] = m.group(1)
        m = re.search(r"Random_Linf_eps\d+_([^_]+)_seed", name)
        if m:
            out["random_mode"] = m.group(1)
        return out
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


def _original_for(stem: str) -> Optional[Path]:
    for root in [Path("test_images/to_protect")]:
        for cand in root.rglob(f"{stem}.*"):
            return cand
    return None


def _noise_key(path: Path, prefix: str) -> Optional[str]:
    m = re.search(rf"_sdedit_{prefix}_noise_([0-9.]+)\.png$", path.name)
    return m.group(1) if m else None


def _find_samples(exp_dir: Path):
    attacked = list(exp_dir.rglob("*_attacked.png"))
    for adv_path in attacked:
        stem = adv_path.name.replace("_attacked.png", "")
        original = _original_for(stem)
        loss_npz = adv_path.with_name(stem + "_loss.npz")
        paired_clean = {_noise_key(p, "clean"): p for p in adv_path.parent.glob(stem + "_sdedit_clean_noise_*.png")}
        paired_adv = {_noise_key(p, "adv"): p for p in adv_path.parent.glob(stem + "_sdedit_adv_noise_*.png")}
        paired_clean.pop(None, None)
        paired_adv.pop(None, None)
        single_adv = sorted(adv_path.parent.glob(stem + "_sdedit_noise_*.png"))
        yield stem, original, adv_path, loss_npz if loss_npz.exists() else None, paired_clean, paired_adv, single_adv


def _add_perturb_metrics(row: Dict[str, object], original_path: Optional[Path], attacked_path: Path, enable_clip: bool, enable_lpips: bool) -> None:
    if not original_path or not original_path.exists():
        return
    row["original_path"] = str(original_path)
    a = _load_rgb(attacked_path)
    o = _load_rgb(original_path, size=Image.open(attacked_path).size)
    row["linf_perturb_0_1"] = _linf(a, o)
    row["linf_perturb_tensor_-1_1"] = 2.0 * row["linf_perturb_0_1"]
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


def _add_loss_metrics(row: Dict[str, object], loss_path: Optional[Path]) -> None:
    if not loss_path:
        return
    row["loss_path"] = str(loss_path)
    try:
        data = np.load(loss_path)
        for key in data.files:
            arr = np.asarray(data[key])
            if arr.size == 0:
                continue
            prefix = f"loss_{key}"
            row[f"{prefix}_first"] = float(arr.reshape(-1)[0])
            row[f"{prefix}_final"] = float(arr.reshape(-1)[-1])
            row[f"{prefix}_delta"] = float(arr.reshape(-1)[-1] - arr.reshape(-1)[0])
    except Exception as exc:
        row["loss_error"] = str(exc)


def _paired_row(base: Dict[str, object], stem: str, original_path: Optional[Path], attacked_path: Path,
                loss_path: Optional[Path], clean_path: Path, adv_edit_path: Path, sigma: str,
                prompt: str, enable_clip: bool, enable_lpips: bool) -> Dict[str, object]:
    row: Dict[str, object] = dict(base)
    row.update({
        "image_id": stem,
        "attacked_path": str(attacked_path),
        "paired": True,
        "sigma": sigma,
        "clean_edit_path": str(clean_path),
        "adv_edit_path": str(adv_edit_path),
    })
    _add_perturb_metrics(row, original_path, attacked_path, enable_clip, enable_lpips)
    _add_loss_metrics(row, loss_path)
    clean = _load_rgb(clean_path)
    adv = _load_rgb(adv_edit_path, size=Image.open(clean_path).size)
    row["l2_edit_rmse"] = _l2(clean, adv)
    row["psnr_edit"] = _psnr(clean, adv)
    row["linf_edit_0_1"] = _linf(clean, adv)
    ssim = _ssim_fallback(clean, adv)
    if ssim is not None:
        row["ssim_edit"] = ssim
    if enable_lpips:
        lp = _try_lpips(clean_path, adv_edit_path)
        if lp is not None:
            row["lpips_edit"] = lp
    if enable_clip:
        c0 = _try_clip_scores(prompt, clean_path)
        c1 = _try_clip_scores(prompt, adv_edit_path)
        if c0 is not None and c1 is not None:
            row["clip_prompt_clean"] = c0
            row["clip_prompt_adv"] = c1
            row["delta_clip_prompt"] = c1 - c0
        if original_path and original_path.exists():
            s0 = _try_clip_image_sim(original_path, clean_path)
            s1 = _try_clip_image_sim(original_path, adv_edit_path)
            if s0 is not None and s1 is not None:
                row["clip_src_clean"] = s0
                row["clip_src_adv"] = s1
                row["delta_clip_src"] = s1 - s0
    return row


def compute_metrics(root: Path, prompt: str, enable_clip: bool, enable_lpips: bool) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    skip_dirs = {"figures", "experiment_logs", "validation_logs", "minimal_logs", "direction_logs", "full_logs"}
    exp_dirs = [p for p in root.iterdir() if p.is_dir() and p.name not in skip_dirs] if root.exists() else []
    for exp in sorted(exp_dirs):
        base = _parse_exp_dir(exp)
        for stem, original_path, attacked_path, loss_path, clean_by_sigma, adv_by_sigma, single_adv in _find_samples(exp):
            paired_sigmas = sorted(set(clean_by_sigma.keys()) & set(adv_by_sigma.keys()))
            if paired_sigmas:
                for sigma in paired_sigmas:
                    rows.append(_paired_row(base, stem, original_path, attacked_path, loss_path, clean_by_sigma[sigma], adv_by_sigma[sigma], sigma, prompt, enable_clip, enable_lpips))
                continue
            base_row: Dict[str, object] = dict(base)
            base_row.update({"image_id": stem, "attacked_path": str(attacked_path), "paired": False})
            _add_perturb_metrics(base_row, original_path, attacked_path, enable_clip, enable_lpips)
            _add_loss_metrics(base_row, loss_path)
            if not single_adv:
                rows.append(base_row)
            for sd in single_adv:
                row = dict(base_row)
                row["adv_edit_path"] = str(sd)
                row["sdedit_name"] = sd.name
                if enable_clip:
                    cs = _try_clip_scores(prompt, sd)
                    if cs is not None:
                        row["clip_text_image"] = cs
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
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "metrics_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
