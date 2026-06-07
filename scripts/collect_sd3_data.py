#!/usr/bin/env python3
"""Run SD3 experiment grids and collect logs/metadata.

The script runs commands sequentially and records every command, return code,
log path, and expected output root. It is designed to stop exactly where an
experiment pipeline fails.
"""
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


def _build_command(args, mode: str, epsilon: str, steps: str, seed: str) -> List[str]:
    return [
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
        f"attack.textual_objective={args.textual_objective}",
        f"attack.debug_grad={str(args.debug_grad)}",
        f"attack.run_sdedit={str(args.run_sdedit)}",
        f"attack.paired_sdedit={str(args.paired_sdedit)}",
        f"attack.sdedit_steps={args.sdedit_steps}",
        f"attack.output_path={args.output_path}",
    ]


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--entrypoint", default="code/diff_mist_SD3_v2.py")
    parser.add_argument("--modes", default="O_repo,C")
    parser.add_argument("--epsilons", default="8")
    parser.add_argument("--steps", default="2")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--max-exp-num", type=int, default=1)
    parser.add_argument("--opt-direction", default="maximize")
    parser.add_argument("--textual-objective", default="toward_target")
    parser.add_argument("--output-path", default="out_sd3/")
    parser.add_argument("--log-root", default="out_sd3/experiment_logs")
    parser.add_argument("--run-sdedit", action="store_true")
    parser.add_argument("--paired-sdedit", action="store_true")
    parser.add_argument("--sdedit-steps", type=int, default=28)
    parser.add_argument("--debug-grad", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-fail", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    entrypoint = repo / args.entrypoint
    if not entrypoint.exists():
        raise FileNotFoundError(f"entrypoint not found: {entrypoint}")

    log_root = Path(args.log_root)
    if not log_root.is_absolute():
        log_root = repo / log_root
    log_root.mkdir(parents=True, exist_ok=True)
    manifest_path = log_root / "manifest.csv"

    modes = _split_csv(args.modes)
    epsilons = _split_csv(args.epsilons)
    steps_values = _split_csv(args.steps)
    seeds = _split_csv(args.seeds)

    rows: List[Dict[str, object]] = []
    for mode, eps, steps, seed in itertools.product(modes, epsilons, steps_values, seeds):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        tag = f"{stamp}_{mode}_eps{eps}_steps{steps}_seed{seed}"
        log_path = log_root / f"{tag}.log"
        cmd = _build_command(args, mode, eps, steps, seed)
        start = time.time()
        rc = _run_one(repo, cmd, log_path, args.dry_run)
        elapsed = time.time() - start
        row = {
            "mode": mode,
            "epsilon": eps,
            "steps": steps,
            "seed": seed,
            "returncode": rc,
            "elapsed_sec": round(elapsed, 3),
            "log_path": str(log_path),
            "command": " ".join(cmd),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if rc != 0 and args.stop_on_fail:
            break

    if rows:
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"manifest": str(manifest_path), "runs": len(rows)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
