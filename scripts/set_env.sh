#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d /root/autodl-tmp ]]; then
  DEFAULT_STORAGE_ROOT=/root/autodl-tmp
else
  DEFAULT_STORAGE_ROOT="$ROOT_DIR"
fi

export PROJECT_ROOT="$ROOT_DIR"
export AUTODL_STORAGE_ROOT="${AUTODL_STORAGE_ROOT:-$DEFAULT_STORAGE_ROOT}"
export HF_HOME="${HF_HOME:-$AUTODL_STORAGE_ROOT/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$AUTODL_STORAGE_ROOT/cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$AUTODL_STORAGE_ROOT/cache/pip}"
export TMPDIR="${TMPDIR:-$AUTODL_STORAGE_ROOT/tmp}"
export MEDICALGPT_ROOT="${MEDICALGPT_ROOT:-$ROOT_DIR/vendor/MedicalGPT}"
export MODEL_7B_PATH="${MODEL_7B_PATH:-$AUTODL_STORAGE_ROOT/models/Qwen2.5-7B}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$AUTODL_STORAGE_ROOT/medical-gpt-outputs}"
export AUTODL_VENV="${AUTODL_VENV:-$AUTODL_STORAGE_ROOT/venvs/medical-gpt}"
if [[ -x "$AUTODL_VENV/bin/python" ]]; then
  export PYTHON_BIN="${PYTHON_BIN:-$AUTODL_VENV/bin/python}"
else
  export PYTHON_BIN="${PYTHON_BIN:-python}"
fi
export PYTHONUTF8=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$MEDICALGPT_ROOT/training${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME" \
  "$PIP_CACHE_DIR" "$TMPDIR" "$OUTPUT_ROOT" "$(dirname "$MODEL_7B_PATH")"
