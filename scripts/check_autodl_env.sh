#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/set_env.sh"

EXTRA_ARGS=()
if [[ "${USE_QLORA:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--qlora)
fi
"$PYTHON_BIN" "$PROJECT_ROOT/tools/check_autodl_env.py" \
  --model "$MODEL_7B_PATH" \
  --cpt-data "$PROJECT_ROOT/data/processed/pretrain-clean-20k-v2" \
  --sft-data "$PROJECT_ROOT/data/processed/sft-clean-20k-v2" \
  --medicalgpt "$MEDICALGPT_ROOT" \
  --require-cuda "${EXTRA_ARGS[@]}"
