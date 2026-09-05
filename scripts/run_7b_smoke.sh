#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/set_env.sh"

QUANT_ARGS=()
SUFFIX=""
if [[ "${USE_QLORA:-0}" == "1" ]]; then
  QUANT_ARGS+=(--qlora)
  SUFFIX="-qlora"
fi

"$PYTHON_BIN" "$PROJECT_ROOT/tools/run_cpt_stage.py" cpt \
  --model-path "$MODEL_7B_PATH" \
  --data-dir "$PROJECT_ROOT/data/processed/pretrain-clean-20k-v2" \
  --output-dir "$OUTPUT_ROOT/pt-smoke-7b-v2$SUFFIX" \
  --train-samples 128 --eval-samples 32 \
  --train-batch-size 1 --eval-batch-size 1 --gradient-accumulation-steps 8 \
  --gradient-checkpointing --eval-steps 16 --save-steps 16 "${QUANT_ARGS[@]}"

"$PYTHON_BIN" "$PROJECT_ROOT/tools/run_sft_stage.py" base --mode smoke \
  --model-path "$MODEL_7B_PATH" \
  --data-dir "$PROJECT_ROOT/data/processed/sft-clean-20k-v2" \
  --output-dir "$OUTPUT_ROOT/sft-base-smoke-7b-v2$SUFFIX" \
  --train-samples 128 --eval-samples 32 \
  --train-batch-size 1 --eval-batch-size 1 --gradient-accumulation-steps 4 \
  --gradient-checkpointing --eval-steps 32 --save-steps 32 "${QUANT_ARGS[@]}"

echo "7B CPT and Base-to-SFT smoke tests completed."
