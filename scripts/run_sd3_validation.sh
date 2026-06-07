#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python scripts/validate_sd3_pipeline.py
python scripts/collect_sd3_data.py \
  --dry-run \
  --stop-on-fail \
  --modes O_repo,C \
  --epsilons 8 \
  --steps 1 \
  --seeds 0 \
  --input-size 256 \
  --max-exp-num 1 \
  --output-path out_sd3/ \
  --log-root out_sd3/validation_logs

python code/metrics/compute_sd3_metrics.py \
  --root out_sd3 \
  --out out_sd3/validation_metrics.csv

echo "[run_sd3_validation] done"
