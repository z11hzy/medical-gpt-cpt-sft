#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/set_env.sh"

MEDICALGPT_COMMIT=ccc05f4b46442ecdefcc95d53aeedc9834d09dd6
QWEN_REVISION=d149729398750b98c0af14eb82c78cfe92750796

python - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(f"Python >= 3.11 is required by the pinned environment, got {sys.version}")
PY

if [[ ! -x "$AUTODL_VENV/bin/python" ]]; then
  python -m venv --system-site-packages "$AUTODL_VENV"
fi
PYTHON_BIN="$AUTODL_VENV/bin/python"
export PYTHON_BIN
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r "$PROJECT_ROOT/requirements-autodl.txt"

if [[ ! -d "$MEDICALGPT_ROOT/.git" ]]; then
  mkdir -p "$(dirname "$MEDICALGPT_ROOT")"
  git clone https://github.com/shibing624/MedicalGPT.git "$MEDICALGPT_ROOT"
fi
git -C "$MEDICALGPT_ROOT" fetch origin "$MEDICALGPT_COMMIT"
git -C "$MEDICALGPT_ROOT" checkout --detach "$MEDICALGPT_COMMIT"

if [[ "${DOWNLOAD_MODEL:-1}" == "1" ]]; then
  "$AUTODL_VENV/bin/hf" download \
    Qwen/Qwen2.5-7B --revision "$QWEN_REVISION" --local-dir "$MODEL_7B_PATH"
fi

echo "AutoDL setup complete. Run: bash scripts/check_autodl_env.sh"
