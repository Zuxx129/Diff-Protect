#!/usr/bin/env python3
"""Run SD3 v3 experiment grids and collect logs/metadata."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


def _split_csv(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _extra_overrides(value: str) -> List[str]:
    if not value:
        return []
    return _split_csv(value)


def _filter_modes(modes: List[str], only_modes: str, exclude_modes: str) -> List[str]:
    if only_modes:
        allowed = set(_split_csv(only_modes))
        modes = [m for m in modes if m in allowed]
    if exclude_modes:
        denied = set(_split_csv(exclude_modes))
        modes = [m for m in modes if m not in denied]
    return modes


def _bool_string(value: str) -> str:
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return "true"
    if value in {"0", "false", "no", "n"}:
        return "false"
    raise ValueError(f"Expected boolean string, got {value!r}")


def _hydra_list(csv_value: str) -> str:
    return "[" + ",".join(_split_csv(csv_value)) + "]"


def _exp_name(args, mode: str, epsilon: str, steps: str, seed: str) -> str:
    return (
        f"{mode}_eps{epsilon}_steps{steps}_opt{args.opt_direction}"
        f"_rs{_bool_string(args.random_start)}_seed{seed}"
    )


def _effective_use_step_loss(args, mode: str) -> str:
    if mode == "FMP_single_plus_step":
        return "true"
    return _bool_string(args.use_step_loss)


def _expected_output_dir(repo: Path, args, mode: str, epsilon: str, steps: str, seed: str) -> Path:
    out = Path(args.output_path)
    if not out.is_absolute():
        out = repo / out
    return out / _exp_name(args, mode, epsilon, steps, seed)


def _build_command(args, mode: str, epsilon: str, steps: str, seed: str) -> List[str]:
    cmd = [
        sys.executable,
        args.entrypoint,
        f"attack.mode={mode}",
        f"attack.epsilon={epsilon}",
        f"attack.steps={steps}",
        f"attack.seed={seed}",
        f"attack.input_size={args.input_size}",
        f"attack.device={args.device}",
        f"attack.max_exp_num={args.max_exp_num}",
        f"attack.opt_direction={args.opt_direction}",
        f"attack.random_start={_bool_string(args.random_start)}",
        f"attack.textual_weight={args.textual_weight}",
        f"attack.mmdit_weight={args.mmdit_weight}",
        f"attack.fmp_sigma_levels={_hydra_list(args.fmp_sigma_levels)}",
        f"attack.fmp_multi_reduce={args.fmp_multi_reduce}",
        f"attack.fmp_target_convention={args.fmp_target_convention}",
        f"attack.use_step_loss={_effective_use_step_loss(args, mode)}",
        f"attack.lambda_step={args.lambda_step}",
        f"attack.run_sdedit={_bool_string(args.run_sdedit)}",
        f"attack.paired_sdedit={_bool_string(args.paired_sdedit)}",
        f"attack.sdedit_noise_levels={_hydra_list(args.sdedit_noise_levels)}",
        f"attack.sdedit_steps={args.sdedit_steps}",
        f"attack.output_path={args.output_path}",
        f"attack.debug_grad={_bool_string(args.debug_grad)}",
    ]
    cmd.extend(_extra_overrides(args.extra))
    return cmd


def _run_one(repo: Path, cmd: List[str], log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("COMMAND: " + " ".join(cmd) + "\n\n")
        if dry_run:
            f.write("DRY RUN: command not executed\n")
            return 0
        proc = subprocess.run(cmd, cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        f.write(proc.stdout)
        return proc.returncode


def _has_finished_output(path: Path) -> bool:
    return path.exists() and any(path.rglob("*_loss.npz")) and any(path.rglob("*_attacked.png"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--entrypoint", default="code/diff_mist_SD3_v3.py")
    parser.add_argument("--modes", default="textual_only,E,O,FMP_single")
    parser.add_argument("--only-modes", default="", help="Optional comma-separated subset of --modes to run")
    parser.add_argument("--exclude-modes", default="", help="Optional comma-separated modes to remove from --modes")
    parser.add_argument("--epsilons", default="8")
    parser.add_argument("--steps", default="50")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--max-exp-num", type=int, default=1)
    parser.add_argument("--opt-direction", default="maximize")
    parser.add_argument("--random-start", default="true")
    parser.add_argument("--textual-weight", type=float, default=1.0)
    parser.add_argument("--mmdit-weight", type=float, default=100000.0)
    parser.add_argument("--fmp-sigma-levels", default="0.1,0.3,0.5")
    parser.add_argument("--fmp-multi-reduce", default="mean")
    parser.add_argument("--fmp-target-convention", default="noise_minus_data")
    parser.add_argument("--use-step-loss", default="false")
    parser.add_argument("--lambda-step", type=float, default=1.0)
    parser.add_argument("--sdedit-noise-levels", default="0.1,0.3,0.5")
    parser.add_argument("--sdedit-steps", type=int, default=28)
    parser.add_argument("--run-sdedit", default="true")
    parser.add_argument("--paired-sdedit", default="true")
    parser.add_argument("--debug-grad", default="false")
    parser.add_argument("--output-path", default="out_sd3_v3/")
    parser.add_argument("--log-root", default="out_sd3_v3/experiment_logs")
    parser.add_argument("--extra", default="", help="Comma-separated additional Hydra overrides")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--continue-on-fail", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    args = parser.parse_args()

    if args.stop_on_fail and args.continue_on_fail:
        raise ValueError("--stop-on-fail and --continue-on-fail cannot both be set")

    repo = Path(args.repo).resolve()
    entrypoint = repo / args.entrypoint
    if not entrypoint.exists():
        raise FileNotFoundError(f"entrypoint not found: {entrypoint}")

    log_root = Path(args.log_root)
    if not log_root.is_absolute():
        log_root = repo / log_root
    log_root.mkdir(parents=True, exist_ok=True)
    manifest_path = log_root / "manifest.csv"

    modes = _filter_modes(_split_csv(args.modes), args.only_modes, args.exclude_modes)
    epsilons = _split_csv(args.epsilons)
    steps_values = _split_csv(args.steps)
    seeds = _split_csv(args.seeds)
    jobs = list(itertools.product(modes, epsilons, steps_values, seeds))
    jobs = jobs[args.start_index:args.end_index]

    rows: List[Dict[str, object]] = []
    for local_idx, (mode, eps, steps, seed) in enumerate(jobs, start=args.start_index):
        expected_dir = _expected_output_dir(repo, args, mode, eps, steps, seed)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        tag = f"{stamp}_idx{local_idx}_{mode}_eps{eps}_steps{steps}_rs{_bool_string(args.random_start)}_seed{seed}"
        log_path = log_root / f"{tag}.log"
        cmd = _build_command(args, mode, eps, steps, seed)

        if args.skip_existing and _has_finished_output(expected_dir):
            rc = 0
            elapsed = 0.0
            status = "skipped_existing"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("SKIPPED: existing attacked image and loss.npz found\n", encoding="utf-8")
        else:
            start = time.time()
            rc = _run_one(repo, cmd, log_path, args.dry_run)
            elapsed = time.time() - start
            status = "dry_run" if args.dry_run else ("ok" if rc == 0 else "failed")

        row = {
            "index": local_idx,
            "mode": mode,
            "epsilon": eps,
            "steps": steps,
            "random_start": _bool_string(args.random_start),
            "seed": seed,
            "textual_weight": args.textual_weight,
            "mmdit_weight": args.mmdit_weight,
            "use_step_loss": _effective_use_step_loss(args, mode),
            "lambda_step": args.lambda_step,
            "returncode": rc,
            "status": status,
            "elapsed_sec": round(elapsed, 3),
            "expected_output_dir": str(expected_dir),
            "log_path": str(log_path),
            "command": " ".join(cmd),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if rc != 0 and args.stop_on_fail and not args.continue_on_fail:
            break

    if rows:
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"manifest": str(manifest_path), "runs": len(rows)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
