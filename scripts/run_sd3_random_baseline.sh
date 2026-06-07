#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EPSILONS="${1:-4,8,16}"
SEEDS="${2:-0,1,2,3,4}"

IFS=',' read -ra EPS_ARR <<< "$EPSILONS"
IFS=',' read -ra SEED_ARR <<< "$SEEDS"

for EPS in "${EPS_ARR[@]}"; do
  for SEED in "${SEED_ARR[@]}"; do
    python code/metrics/random_linf_baseline.py \
      --input test_images/to_protect \
      --output out_sd3 \
      --epsilon "$EPS" \
      --mode uniform \
      --seed "$SEED" \
      --max-exp-num 100
  done
done

python code/metrics/compute_sd3_metrics.py \
  --root out_sd3 \
  --out out_sd3/random_linf_metrics.csv

python code/metrics/aggregate_sd3_results.py \
  --metrics out_sd3/random_linf_metrics.csv \
  --group-by mode,epsilon,seed \
  --out out_sd3/random_linf_summary.csv || true

echo "[run_sd3_random_baseline] done"
