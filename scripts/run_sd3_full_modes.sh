#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${1:-cuda:0}"
INPUT_SIZE="${2:-512}"
MAX_EXP_NUM="${3:-5}"

python scripts/collect_sd3_data.py \
  --stop-on-fail \
  --skip-existing \
  --modes O_repo,O_fair,A,B,C,D \
  --epsilons 4,8,16 \
  --steps 20,50 \
  --seeds 0,1,2,3,4 \
  --device "$DEVICE" \
  --input-size "$INPUT_SIZE" \
  --max-exp-num "$MAX_EXP_NUM" \
  --opt-direction maximize \
  --textual-objective toward_target \
  --debug-grad \
  --run-sdedit \
  --paired-sdedit \
  --sdedit-steps 28 \
  --output-path out_sd3/ \
  --log-root out_sd3/full_logs

python code/metrics/compute_sd3_metrics.py \
  --root out_sd3 \
  --out out_sd3/full_metrics.csv

python code/metrics/aggregate_sd3_results.py \
  --metrics out_sd3/full_metrics.csv \
  --group-by mode,epsilon,steps,sigma \
  --out out_sd3/full_summary.csv

python code/metrics/compute_fid_kid.py \
  --root out_sd3 \
  --out out_sd3/full_fid_kid.json \
  --device "$DEVICE" || true

python code/metrics/plot_sd3_paper_figures.py \
  --metrics out_sd3/full_metrics.csv \
  --summary out_sd3/full_summary.csv \
  --out-dir out_sd3/paper_figures || true

python code/plot_loss.py \
  --root out_sd3 \
  --all \
  --components \
  --smooth 5 \
  --dpi 300 || true

echo "[run_sd3_full_modes] done"
