#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/set_env.sh"

QUANT_ARGS=()
if [[ "${USE_QLORA:-0}" == "1" ]]; then QUANT_ARGS+=(--qlora); fi
"$PYTHON_BIN" "$PROJECT_ROOT/tools/run_cpt_stage.py" base \
  --model-path "$MODEL_7B_PATH" \
  --data-dir "$PROJECT_ROOT/data/processed/pretrain-clean-20k-v2" \
  --output-dir "$OUTPUT_ROOT/pt-base-full-eval-7b-v2" \
  --eval-batch-size 1

"$PYTHON_BIN" "$PROJECT_ROOT/tools/run_cpt_stage.py" cpt \
  --model-path "$MODEL_7B_PATH" \
  --data-dir "$PROJECT_ROOT/data/processed/pretrain-clean-20k-v2" \
  --output-dir "$OUTPUT_ROOT/pt-full20k-7b-v2" \
  --peer-identity "$OUTPUT_ROOT/pt-base-full-eval-7b-v2/dataset_identity.json" \
  --train-batch-size 1 --eval-batch-size 1 --gradient-accumulation-steps 8 \
  --gradient-checkpointing --eval-steps 500 --save-steps 1000 "${QUANT_ARGS[@]}"
