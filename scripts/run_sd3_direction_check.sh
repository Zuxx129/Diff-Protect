#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${1:-cuda:0}"

python scripts/collect_sd3_data.py \
  --stop-on-fail \
  --modes A,B,C,D \
  --epsilons 8 \
  --steps 2 \
  --seeds 0 \
  --device "$DEVICE" \
  --input-size 256 \
  --max-exp-num 1 \
  --opt-direction maximize \
  --textual-objective toward_target \
  --debug-grad \
  --output-path out_sd3/ \
  --log-root out_sd3/direction_logs

python code/metrics/compute_sd3_metrics.py \
  --root out_sd3 \
  --out out_sd3/direction_metrics.csv

python code/metrics/aggregate_sd3_results.py \
  --metrics out_sd3/direction_metrics.csv \
  --group-by mode,epsilon,steps \
  --out out_sd3/direction_summary.csv || true

echo "[run_sd3_direction_check] done"
