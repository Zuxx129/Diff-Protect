#!/usr/bin/env python3
"""Validation harness for the SD3 experiment branch.

This script is deliberately static-first. It verifies repository wiring that can
be checked without a GPU, then optionally runs a tiny GPU smoke test.
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
    "code/diff_mist_SD3_v2.py",
    "code/metrics/compute_sd3_metrics.py",
    "code/metrics/aggregate_sd3_results.py",
    "code/metrics/random_linf_baseline.py",
    "code/metrics/compute_fid_kid.py",
    "code/plot_loss.py",
    "scripts/collect_sd3_data.py",
    "scripts/validate_sd3_pipeline.py",
    "scripts/run_sd3_validation.sh",
    "scripts/run_sd3_minimal_collect.sh",
    "scripts/run_sd3_direction_check.sh",
    "scripts/run_sd3_full_modes.sh",
    "scripts/run_sd3_random_baseline.sh",
    "scripts/run_sd3_ablation_weights.sh",
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
    for rel in REQUIRED_FILES:
        if rel.endswith(".py"):
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
    print("[validate] config contains objective and eval keys")


def check_attack_semantics(repo: Path) -> None:
    stage("attack-objective-semantics")
    text = (repo / "code/attacks_SD3.py").read_text(encoding="utf-8")
    required = [
        "larger scalar loss = stronger attack objective",
        "detach_features",
        "textual_objective",
        "loss = 1.0 - cos_sim.mean()",
        "_trajectory_loss_shared_noise",
        "_compute_clean_features_at_timestep",
        "txt_injection_feats",
        "modality_ratio_dev",
        "cross_modal_cka",
        "grad_mmdit_l2",
        "img_to_txt_attn",
    ]
    missing = [s for s in required if s not in text]
    if missing:
        fail("attack-objective-semantics", f"missing snippets: {missing}")
    old_patterns = [
        r"attn_maps\.append\(attn_weights\.detach\(\)\)",
        r"hook\.img_stream_feats\.append\(img_h\.detach\(\)\)",
    ]
    bad = [p for p in old_patterns if re.search(p, text)]
    if bad:
        fail("attack-objective-semantics", f"old detach patterns remain: {bad}")
    print("[validate] attack objective semantics look consistent")


def check_entrypoints(repo: Path) -> None:
    stage("entrypoints")
    v2 = (repo / "code/diff_mist_SD3_v2.py").read_text(encoding="utf-8")
    collect = (repo / "scripts/collect_sd3_data.py").read_text(encoding="utf-8")
    required_v2 = ["infer_v2", "_run_sdedit_pair", "paired_sdedit", "SD3_Linf_PGD"]
    missing = [s for s in required_v2 if s not in v2]
    if missing:
        fail("entrypoints", f"diff_mist_SD3_v2.py missing implementation markers: {missing}")
    if "code/diff_mist_SD3_v2.py" not in collect:
        fail("entrypoints", "collect_sd3_data.py must call code/diff_mist_SD3_v2.py")
    if "--extra" not in collect:
        fail("entrypoints", "collect_sd3_data.py must support --extra for ablation")
    print("[validate] entrypoints are wired")


def check_experiment_tooling(repo: Path) -> None:
    stage("experiment-tooling")
    checks = {
        "scripts/run_sd3_minimal_collect.sh": ["--paired-sdedit", "aggregate_sd3_results.py"],
        "scripts/run_sd3_direction_check.sh": ["A,B,C,D", "--debug-grad"],
        "scripts/run_sd3_full_modes.sh": ["O_repo,O_fair,A,B,C,D", "--paired-sdedit"],
        "scripts/run_sd3_ablation_weights.sh": ["attack.textual_weight", "attack.mmdit_weight"],
        "scripts/run_sd3_random_baseline.sh": ["random_linf_baseline.py"],
        "code/metrics/compute_sd3_metrics.py": ["delta_clip_prompt", "loss_", "paired_sigmas"],
        "code/metrics/aggregate_sd3_results.py": ["success_rate"],
    }
    missing = []
    for rel, snippets in checks.items():
        text = (repo / rel).read_text(encoding="utf-8")
        for s in snippets:
            if s not in text:
                missing.append(f"{rel}: {s}")
    if missing:
        fail("experiment-tooling", f"missing snippets: {missing}")
    print("[validate] experiment tooling is wired")


def run_smoke(repo: Path, args: argparse.Namespace) -> None:
    stage("optional-gpu-smoke")
    cmd = [
        sys.executable,
        "code/diff_mist_SD3_v2.py",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--epsilon", type=int, default=8)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    for fn in [check_files, check_compile, check_config, check_attack_semantics, check_entrypoints, check_experiment_tooling]:
        fn(repo)
    if args.run_smoke:
        run_smoke(repo, args)
    print(json.dumps({"status": "passed", "repo": str(repo), "smoke": args.run_smoke}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
