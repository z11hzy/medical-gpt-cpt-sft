#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/set_env.sh"
"$PYTHON_BIN" "$PROJECT_ROOT/tools/run_reward_model.py" \
 --model "$AUTODL_STORAGE_ROOT/models/Qwen2.5-0.5B-SFT-merged" \
 --data-dir "$PROJECT_ROOT/data/processed/dpo-zh-10k-v1" \
 --output "$OUTPUT_ROOT/rm-sft-0.5b-v1" --epochs 1 --eval-steps 250 --save-steps 500
