#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_zfc_ablation.sh <mode|all> <final_training_config.yml> [seed] [total_iter]
#
# Examples:
#   bash scripts/run_zfc_ablation.sh zfc /path/to/final_GSBlur.yml 42 25000
#   bash scripts/run_zfc_ablation.sh all /path/to/final_GSBlur.yml 42 25000
#
# Optional environment overrides:
#   NUM_GPU=1 BATCH_SIZE_PER_GPU=1 GT_SIZE=384 VAL_FREQ=5000 SAVE_FREQ=5000

MODE="${1:?Please provide a mode: none|zfc|se|se_matched|cross_se_matched|film|film_bottleneck|all}"
BASE_CONFIG="${2:?Please provide the actual final training YAML as the second argument}"
SEED="${3:-42}"
TOTAL_ITER="${4:-25000}"
NUM_GPU="${NUM_GPU:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "[ERROR] Base config not found: $BASE_CONFIG" >&2
  exit 1
fi

run_one() {
  local mode="$1"
  local tmp_cfg
  tmp_cfg="$(mktemp "/tmp/omni_dc_${mode}_XXXXXX.yml")"
  local args=(
    --base-config "$BASE_CONFIG"
    --mode "$mode"
    --seed "$SEED"
    --num-gpu "$NUM_GPU"
    --output "$tmp_cfg"
  )

  if [[ -n "$TOTAL_ITER" ]]; then
    args+=(--total-iter "$TOTAL_ITER")
  fi
  if [[ -n "${BATCH_SIZE_PER_GPU:-}" ]]; then
    args+=(--batch-size-per-gpu "$BATCH_SIZE_PER_GPU")
  fi
  if [[ -n "${GT_SIZE:-}" ]]; then
    args+=(--gt-size "$GT_SIZE")
  fi
  if [[ -n "${VAL_FREQ:-}" ]]; then
    args+=(--val-freq "$VAL_FREQ")
  fi
  if [[ -n "${SAVE_FREQ:-}" ]]; then
    args+=(--save-freq "$SAVE_FREQ")
  fi

  echo
  echo "============================================================"
  echo "[RUN] mode=$mode seed=$SEED total_iter=$TOTAL_ITER num_gpu=$NUM_GPU"
  echo "============================================================"
  "$PYTHON_BIN" scripts/prepare_zfc_ablation_config.py "${args[@]}"
  "$PYTHON_BIN" basicsr/train.py -opt "$tmp_cfg"
  rm -f "$tmp_cfg"
}

if [[ "$MODE" == "all" ]]; then
  MODES=(none se se_matched cross_se_matched film zfc)
  for mode in "${MODES[@]}"; do
    run_one "$mode"
  done
else
  run_one "$MODE"
fi
