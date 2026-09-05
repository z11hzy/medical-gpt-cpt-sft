#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/set_env.sh"

QUANT_ARGS=()
if [[ "${USE_QLORA:-0}" == "1" ]]; then QUANT_ARGS+=(--qlora); fi
"$PYTHON_BIN" "$PROJECT_ROOT/tools/run_sft_stage.py" base --mode formal \
  --model-path "$MODEL_7B_PATH" \
  --data-dir "$PROJECT_ROOT/data/processed/sft-clean-20k-v2" \
  --output-dir "$OUTPUT_ROOT/sft-base-20k-7b-v2" \
  --train-batch-size 1 --eval-batch-size 1 --gradient-accumulation-steps 4 \
  --gradient-checkpointing --eval-steps 1000 --save-steps 2500 "${QUANT_ARGS[@]}"
