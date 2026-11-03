#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_zfc_only_ablation.sh <mode> <paper_final_config.yml> <run_name>
#   bash scripts/run_zfc_only_ablation.sh all <paper_final_config.yml> <run_prefix>
#
# The input config must be the exact YAML used for the paper's reported run.
# No data path, loss, optimizer, augmentation, batch size or iteration setting
# is changed by this launcher.

MODE="${1:?mode required: none|se_matched|film|zfc|all}"
BASE_CONFIG="${2:?path to the exact paper YAML required}"
RUN_NAME="${3:?unique run name/prefix required}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "[ERROR] Config not found: $BASE_CONFIG" >&2
  exit 1
fi

run_one() {
  local mode="$1"
  local name="${RUN_NAME}_${mode}"
  local config="options/ablation/${name}.yml"
  "$PYTHON_BIN" scripts/prepare_zfc_only_config.py \
    --base-config "$BASE_CONFIG" --mode "$mode" \
    --run-name "$name" --output "$config"
  echo "[TRAIN] $name"
  "$PYTHON_BIN" basicsr/train.py -opt "$config"
}

if [[ "$MODE" == "all" ]]; then
  for mode in none se_matched film zfc; do
    run_one "$mode"
  done
else
  run_one "$MODE"
fi
