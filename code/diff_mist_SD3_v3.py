#!/usr/bin/env python3
"""SD3 v3 launcher with isolated objectives and paired SDEdit outputs."""
from __future__ import annotations

import glob
import json
import os
import ssl
import time
from pathlib import Path

import hydra
import numpy as np
import PIL
import torch
import torchvision.transforms as transforms
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from attacks_SD3_v3 import SD3V3LinfPGD
from diff_mist_SD3 import SD3_target_model, identity_loss, load_image_from_path, _denoise_from_noise_level
from utils import cprint, mp

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["TORCH_HOME"] = os.getcwd()
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hub/")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def _device(device):
    if isinstance(device, int):
        return f"cuda:{device}"
    if isinstance(device, str) and device.isdigit():
        return f"cuda:{device}"
    return str(device)


def _seed(seed):
    if seed is None:
        return None
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "yes", "y"}:
            return True
        if value in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _tensor_from_pil(img, input_size, device):
    img = img.convert("RGB")
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    arr = arr[:, :, :3]
    trans = transforms.Compose([transforms.ToTensor()])
    out = torch.zeros([1, 3, input_size, input_size], device=device)
    out[0] = trans(arr).to(device)
    return out


def _prompt_for_path(path):
    if "anime" in path:
        return "an anime picture"
    if "artwork" in path:
        return "an artwork painting"
    if "landscape" in path:
        return "a landscape photo"
    if "portrait" in path:
        return "a portrait photo"
    return "a photo"


def _bool_name(value) -> str:
    return str(_to_bool(value)).lower()


def _jsonable_cfg(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)


def init_v3(args, prompt, device):
    try:
        from diffusers import StableDiffusion3Pipeline
        backend = "diffusers"
    except Exception as diffusers_exc:
        try:
            from modelscope import StableDiffusion3Pipeline
            backend = "modelscope"
        except Exception as modelscope_exc:
            raise ImportError(
                "Unable to import StableDiffusion3Pipeline. Install the v3.1 "
                "runtime with `pip install diffusers transformers accelerate "
                "safetensors sentencepiece protobuf`, or install the legacy "
                "`modelscope` package if you intentionally use that backend."
            ) from diffusers_exc

    cprint(f"Loading SD3 model via {backend}: {args.get('model_name', 'stabilityai/stable-diffusion-3.5-medium')}", "y")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.get("model_name", "stabilityai/stable-diffusion-3.5-medium"),
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(device)
    if hasattr(pipe.transformer, "gradient_checkpointing"):
        pipe.transformer.gradient_checkpointing = False
    if hasattr(pipe.transformer, "disable_gradient_checkpointing"):
        pipe.transformer.disable_gradient_checkpointing()
    net = SD3_target_model(
        pipe,
        condition=prompt,
        mode=args.mode,
        g_mode=args.get("g_mode", "+"),
        textual_weight=float(args.get("textual_weight", 1.0)),
        mmdit_weight=float(args.get("mmdit_weight", 100000.0)),
        device=device,
    )
    net.eval()
    return {
        "net": net,
        "fn": identity_loss(),
        "parameters": {
            "epsilon": args.epsilon / 255.0 * 2.0,
            "alpha": args.alpha / 255.0 * 2.0,
            "steps": int(args.steps),
            "input_size": int(args.input_size),
            "mode": str(args.mode),
            "opt_direction": args.get("opt_direction", "maximize"),
            "textual_weight": float(args.get("textual_weight", 1.0)),
            "mmdit_weight": float(args.get("mmdit_weight", 100000.0)),
            "random_start": _to_bool(args.get("random_start", True)),
            "fmp_sigma_levels": list(args.get("fmp_sigma_levels", [0.1, 0.3, 0.5])),
            "fmp_multi_reduce": args.get("fmp_multi_reduce", "mean"),
            "fmp_target_convention": args.get("fmp_target_convention", "noise_minus_data"),
            "fmp_loss_eta": float(args.get("fmp_loss_eta", 1e-8)),
            "use_step_loss": _to_bool(args.get("use_step_loss", False)),
            "lambda_step": float(args.get("lambda_step", 1.0)),
            "sdedit_steps": int(args.get("sdedit_steps", 28)),
            "debug_grad": _to_bool(args.get("debug_grad", False)),
            "capture_blocks": args.get("capture_blocks", None),
        },
    }


def _run_sdedit_pair(net, x_clean, x_adv, device, levels, steps, seed=None, paired=True):
    if not levels:
        return {}
    pipe = net.pipe
    out = {}
    with torch.no_grad():
        z_adv = pipe.vae.encode(x_adv.to(pipe.vae.dtype)).latent_dist.mean
        z_adv = (z_adv - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        z_adv = z_adv.to(pipe.transformer.dtype)
        z_clean = None
        if paired:
            z_clean = pipe.vae.encode(x_clean.to(pipe.vae.dtype)).latent_dist.mean
            z_clean = (z_clean - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            z_clean = z_clean.to(pipe.transformer.dtype)
    for i, level in enumerate(levels):
        sigma = float(level)
        gen = None
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed) + 1009 * i)
        noise = torch.randn(z_adv.shape, device=z_adv.device, dtype=z_adv.dtype, generator=gen)
        with torch.no_grad():
            adv_img = _denoise_from_noise_level(
                pipe,
                (1.0 - sigma) * z_adv + sigma * noise,
                net.prompt_embeds,
                net.pooled_prompt_embeds,
                sigma,
                int(steps),
                device,
            )
            if paired and z_clean is not None:
                clean_img = _denoise_from_noise_level(
                    pipe,
                    (1.0 - sigma) * z_clean + sigma * noise,
                    net.prompt_embeds,
                    net.pooled_prompt_embeds,
                    sigma,
                    int(steps),
                    device,
                )
                out[f"clean_noise_{sigma}"] = clean_img
                out[f"adv_noise_{sigma}"] = adv_img
            else:
                out[f"noise_{sigma}"] = adv_img
    return out


def infer_v3(
    img,
    cfg,
    tar_img,
    device,
    seed=None,
    run_sdedit=True,
    paired_sdedit=True,
    sdedit_levels=None,
    sdedit_steps=28,
):
    net = cfg["net"]
    p = cfg["parameters"]
    x = _tensor_from_pil(img, p["input_size"], device)
    target = _tensor_from_pil(tar_img, p["input_size"], device) if tar_img is not None else None
    net.target_info = target
    net.mode = p["mode"]
    net._clean_x = x.clone().detach()

    attack = SD3V3LinfPGD(
        net=net,
        fn=cfg["fn"],
        epsilon=p["epsilon"],
        steps=p["steps"],
        eps_iter=p["alpha"],
        clip_min=-1.0,
        clip_max=1.0,
        opt_direction=p["opt_direction"],
        mode=p["mode"],
        textual_weight=p["textual_weight"],
        mmdit_weight=p["mmdit_weight"],
        fmp_sigma_levels=p["fmp_sigma_levels"],
        fmp_multi_reduce=p["fmp_multi_reduce"],
        fmp_target_convention=p["fmp_target_convention"],
        fmp_loss_eta=p["fmp_loss_eta"],
        use_step_loss=p["use_step_loss"],
        lambda_step=p["lambda_step"],
        sdedit_steps=p["sdedit_steps"],
        debug_grad=p["debug_grad"],
        capture_blocks=p["capture_blocks"],
    )
    t0 = time.time()
    adv, history = attack.pgd_sd3(X=x, target_image=target, random_start=p["random_start"])
    print(f"Max perturbation: {torch.abs(adv - x).max().item():.6f}")
    print(f"Attack takes: {time.time() - t0:.2f}s")

    sdedit = {}
    if run_sdedit:
        sdedit = _run_sdedit_pair(
            net,
            x,
            adv,
            device,
            sdedit_levels,
            sdedit_steps,
            seed=seed,
            paired=paired_sdedit,
        )
    image = torch.clamp((adv[0] + 1.0) / 2.0, 0.0, 1.0).detach()
    grid = 255.0 * rearrange(image, "c h w -> h w c").cpu().numpy()
    return grid, sdedit, history


@hydra.main(version_base=None, config_path="../configs/attack", config_name="base_sd3_v3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    args = cfg.attack
    device = _device(args.device)
    seed = _seed(args.get("seed", None))
    prompt = _prompt_for_path(args.img_path)
    random_start_name = _bool_name(args.get("random_start", True))
    name = (
        f"{args.mode}_eps{args.epsilon}_steps{args.steps}"
        f"_opt{args.get('opt_direction', 'maximize')}_rs{random_start_name}"
    )
    if seed is not None:
        name += f"_seed{seed}"
    out_root = os.path.join(args.output_path, name)
    Path(out_root).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out_root, "config.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable_cfg(cfg), f, indent=2, ensure_ascii=False)

    run_cfg = init_v3(args, prompt, device)
    paths = (
        glob.glob(args.img_path + "/*.png")
        + glob.glob(args.img_path + "/*.jpg")
        + glob.glob(args.img_path + "/*.jpeg")
    )
    paths = paths[: args.get("max_exp_num", 100)]
    cprint(f"Found {len(paths)} images to process", "y")

    for path in tqdm(paths):
        parent = Path(path).parent.name
        stem = Path(path).stem
        rel_stem = os.path.join(parent, stem)
        sample_dir = os.path.join(out_root, parent)
        Path(sample_dir).mkdir(parents=True, exist_ok=True)

        target_path = "test_images/target/MIST.png"
        img = load_image_from_path(path, args.input_size)
        tar = load_image_from_path(target_path, args.input_size) if os.path.exists(target_path) else None
        grid, sdedit, history = infer_v3(
            img,
            run_cfg,
            tar,
            device,
            seed=seed,
            run_sdedit=_to_bool(args.get("run_sdedit", True)),
            paired_sdedit=_to_bool(args.get("paired_sdedit", True)),
            sdedit_levels=list(args.get("sdedit_noise_levels", [0.1, 0.3, 0.5])),
            sdedit_steps=args.get("sdedit_steps", 28),
        )
        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(out_root, f"{rel_stem}_attacked.png"))
        for key, pil_img in sdedit.items():
            if isinstance(pil_img, PIL.Image.Image):
                pil_img.save(os.path.join(out_root, f"{rel_stem}_sdedit_{key}.png"))
        if history and len(history.get("loss_total", history.get("total", []))) > 0:
            loss_path = os.path.join(out_root, f"{rel_stem}_loss.npz")
            np.savez(loss_path, **{k: np.asarray(v) for k, v in history.items()})
            cprint(
                f"Loss saved: total={history.get('loss_total', history.get('total'))[-1]:.4f}",
                "g",
            )


if __name__ == "__main__":
    main()
