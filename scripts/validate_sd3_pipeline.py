#!/usr/bin/env python3
"""Validation harness for Diff-Protect SD3.

Stages fail early and identify the broken layer: files, syntax, config keys,
objective semantics, launcher wiring, or optional GPU smoke run.
"""
from __future__ import annotations

import argparse
import json
import py_compile
import re
import subprocess
import sys
from pathlib import Path

REQUIRED_FILES = [
    "configs/attack/base_sd3.yaml",
    "code/attacks_SD3.py",
    "code/diff_mist_SD3.py",
    "code/plot_loss.py",
]
REQUIRED_CONFIG_KEYS = [
    "opt_direction",
    "objective_convention",
    "textual_objective",
    "debug_grad",
    "capture_blocks",
    "seed",
    "run_sdedit",
    "paired_sdedit",
    "sdedit_noise_levels",
    "sdedit_steps",
]


def fail(stage: str, error: str) -> None:
    print(json.dumps({"status": "failed", "stage": stage, "error": error}, ensure_ascii=False, indent=2))
    raise SystemExit(1)


def stage(name: str) -> None:
    print(f"\n[validate] === {name} ===")


def check_files(repo: Path) -> None:
    stage("file-existence")
    missing = [p for p in REQUIRED_FILES if not (repo / p).exists()]
    if missing:
        fail("file-existence", f"missing files: {missing}")
    print("[validate] required files exist")


def check_compile(repo: Path) -> None:
    stage("py-compile")
    for rel in ["code/attacks_SD3.py", "code/diff_mist_SD3.py", "code/plot_loss.py", "scripts/validate_sd3_pipeline.py", "scripts/collect_sd3_data.py"]:
        try:
            py_compile.compile(str(repo / rel), doraise=True)
            print(f"[validate] compiled {rel}")
        except Exception as exc:
            fail("py-compile", f"{rel}: {exc}")


def check_config(repo: Path) -> None:
    stage("config-keys")
    text = (repo / "configs/attack/base_sd3.yaml").read_text(encoding="utf-8")
    missing = [k for k in REQUIRED_CONFIG_KEYS if f"{k}:" not in text]
    if missing:
        fail("config-keys", f"missing keys: {missing}")
    print("[validate] base_sd3.yaml contains objective and eval keys")


def check_attack_semantics(repo: Path) -> None:
    stage("attack-objective-semantics")
    text = (repo / "code/attacks_SD3.py").read_text(encoding="utf-8")
    required = [
        "larger scalar loss = stronger attack objective",
        "detach_features",
        "textual_objective",
        "return 1.0 - cos_sim.mean()",
        "_print_grad_debug",
        "img_to_txt_attn",
    ]
    missing = [s for s in required if s not in text]
    if missing:
        fail("attack-objective-semantics", f"missing snippets: {missing}")
    bad_patterns = [
        r"with torch\.no_grad\(\):\s*# Compute attention weights",
        r"attn_maps\.append\(attn_weights\.detach\(\)\)",
        r"hook\.img_stream_feats\.append\(img_h\.detach\(\)\)",
    ]
    bad = [p for p in bad_patterns if re.search(p, text, flags=re.MULTILINE)]
    if bad:
        fail("attack-objective-semantics", f"old detach/no_grad patterns remain: {bad}")
    print("[validate] attack objective semantics look consistent")


def check_launcher(repo: Path) -> None:
    stage("launcher-wiring")
    text = (repo / "code/diff_mist_SD3.py").read_text(encoding="utf-8")
    required = [
        "args.get('opt_direction'",
        "args.get('textual_objective'",
        "args.get('debug_grad'",
        "args.get('capture_blocks'",
        "paired_sdedit",
        "sdedit_noise_levels",
        "textual_objective=textual_objective",
        "debug_grad=debug_grad",
        "capture_blocks=capture_blocks",
    ]
    missing = [s for s in required if s not in text]
    if missing:
        fail("launcher-wiring", f"missing launcher wiring: {missing}")
    print("[validate] launcher wiring looks consistent")


def run_smoke(repo: Path, args: argparse.Namespace) -> None:
    stage("optional-gpu-smoke")
    cmd = [
        sys.executable,
        "code/diff_mist_SD3.py",
        "attack.mode=O_repo",
        "attack.opt_direction=maximize",
        "attack.textual_objective=toward_target",
        "attack.steps=1",
        f"attack.epsilon={args.epsilon}",
        f"attack.input_size={args.input_size}",
        f"attack.device={args.device}",
        "attack.run_sdedit=False",
        "attack.max_exp_num=1",
        "attack.debug_grad=True",
    ]
    print("[validate] running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout)
    if proc.returncode != 0:
        fail("optional-gpu-smoke", f"smoke command failed with return code {proc.returncode}")
    print("[validate] smoke run passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--epsilon", type=int, default=8)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    for fn in [check_files, check_compile, check_config, check_attack_semantics, check_launcher]:
        fn(repo)
    if args.run_smoke:
        run_smoke(repo, args)
    print(json.dumps({"status": "passed", "repo": str(repo), "smoke": args.run_smoke}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
