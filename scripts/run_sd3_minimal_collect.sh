#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${1:-cuda:0}"

python scripts/collect_sd3_data.py \
  --stop-on-fail \
  --modes O_repo,C \
  --epsilons 8,16 \
  --steps 2,5 \
  --seeds 0 \
  --device "$DEVICE" \
  --input-size 256 \
  --max-exp-num 1 \
  --opt-direction maximize \
  --textual-objective toward_target \
  --debug-grad \
  --output-path out_sd3/ \
  --log-root out_sd3/minimal_logs

python code/metrics/compute_sd3_metrics.py \
  --root out_sd3 \
  --out out_sd3/minimal_metrics.csv

python code/plot_loss.py \
  --root out_sd3 \
  --all \
  --components \
  --smooth 3 \
  --dpi 200 || true

echo "[run_sd3_minimal_collect] done"
