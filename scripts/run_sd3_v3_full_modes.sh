#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python scripts/collect_sd3_v3_data.py \
  --modes textual_only,E,O,A,B,C,D,FMP_single,FMP_multi \
  --epsilons 4,8,16 \
  --steps 50,100 \
  --seeds 0,1,2 \
  --random-start true \
  --textual-weight 1 \
  --mmdit-weight 100000 \
  --input-size 512 \
  --max-exp-num 100 \
  --run-sdedit true \
  --paired-sdedit true \
  --output-path out_sd3_v3/ \
  --log-root out_sd3_v3/full_logs \
  --skip-existing

python code/metrics/compute_sd3_metrics.py \
  --root out_sd3_v3 \
  --out out_sd3_v3/full_metrics.csv

python code/analysis/analyze_sd3_v3_results.py \
  --metrics out_sd3_v3/full_metrics.csv \
  --out-dir out_sd3_v3/analysis

python -m draw.figures.make_all_figures \
  --root out_sd3_v3 \
  --metrics out_sd3_v3/full_metrics.csv \
  --analysis-dir out_sd3_v3/analysis \
  --out-dir out_sd3_v3/figures \
  --loss-image suzume
