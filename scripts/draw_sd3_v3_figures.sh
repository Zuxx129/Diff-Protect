#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-out_sd3_v3}"
PROMPT="${PROMPT:-a photo}"
OUT_DIR="${OUT_DIR:-$ROOT/figures}"
FORMATS="${FORMATS:-png,svg}"
DPI="${DPI:-300}"
LOSS_IMAGE="${LOSS_IMAGE:-suzume}"
LEGACY_LOSS_ROOT="${LEGACY_LOSS_ROOT:-}"
METRICS="$ROOT/full_metrics.csv"
ANALYSIS_DIR="$ROOT/analysis"

if [[ "${FORCE_METRICS:-0}" == "1" || ! -f "$METRICS" ]]; then
  COMPUTE_ARGS=(code/metrics/compute_sd3_metrics.py --root "$ROOT" --prompt "$PROMPT" --out "$METRICS")
  if [[ "${CLIP:-0}" == "1" ]]; then
    COMPUTE_ARGS+=(--clip)
  fi
  if [[ "${LPIPS:-0}" == "1" ]]; then
    COMPUTE_ARGS+=(--lpips)
  fi
  python "${COMPUTE_ARGS[@]}"
fi

HAS_ROWS="0"
if [[ -f "$METRICS" ]]; then
  HAS_ROWS="$(python -c "import csv,sys; print(1 if sum(1 for _ in csv.DictReader(open(sys.argv[1], encoding='utf-8'))) else 0)" "$METRICS")"
fi

if [[ "$HAS_ROWS" == "1" ]]; then
  python code/analysis/analyze_sd3_v3_results.py --metrics "$METRICS" --out-dir "$ANALYSIS_DIR"
else
  echo "Skip analysis tables: metrics file is missing or has no data rows: $METRICS"
fi

DRAW_ARGS=(
  -m draw.figures.make_all_figures
  --root "$ROOT"
  --metrics "$METRICS"
  --analysis-dir "$ANALYSIS_DIR"
  --out-dir "$OUT_DIR"
  --loss-image "$LOSS_IMAGE"
  --formats "$FORMATS"
  --dpi "$DPI"
)
if [[ -n "$LEGACY_LOSS_ROOT" ]]; then
  DRAW_ARGS+=(--legacy-loss-root "$LEGACY_LOSS_ROOT")
fi
python "${DRAW_ARGS[@]}"

echo "SD3 v3.1 figures written to $OUT_DIR"
