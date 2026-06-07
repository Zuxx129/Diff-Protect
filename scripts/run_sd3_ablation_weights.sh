#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${1:-cuda:0}"
MODE="${2:-C}"
INPUT_SIZE="${3:-512}"
MAX_EXP_NUM="${4:-5}"

for TW in 0.25 1.0 4.0; do
  for MW in 0.25 1.0 4.0; do
    python scripts/collect_sd3_data.py \
      --stop-on-fail \
      --modes "$MODE" \
      --epsilons 8,16 \
      --steps 50 \
      --seeds 0,1,2 \
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
      --log-root "out_sd3/ablation_${MODE}_tw${TW}_mw${MW}_logs" \
      --extra "attack.textual_weight=${TW},attack.mmdit_weight=${MW}"
  done
done

python code/metrics/compute_sd3_metrics.py \
  --root out_sd3 \
  --out "out_sd3/ablation_${MODE}_metrics.csv"

python code/metrics/aggregate_sd3_results.py \
  --metrics "out_sd3/ablation_${MODE}_metrics.csv" \
  --group-by mode,epsilon,steps,sigma \
  --out "out_sd3/ablation_${MODE}_summary.csv"

echo "[run_sd3_ablation_weights] done"
