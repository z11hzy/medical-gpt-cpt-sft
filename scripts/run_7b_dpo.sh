#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/set_env.sh"
"$PYTHON_BIN" "$PROJECT_ROOT/tools/run_dpo_from_sft_adapter_v2.py" \
  --model-path "$MODEL_7B_PATH" --sft-adapter "$OUTPUT_ROOT/sft-base-20k-7b-v2" \
  --data-dir "$PROJECT_ROOT/data/processed/dpo-zh-10k-v1" \
  --output-dir "$OUTPUT_ROOT/dpo-base-sft-zh-v1" \
  --epochs 1 --batch-size 1 --gradient-accumulation-steps 8 --eval-steps 250 --save-steps 500
