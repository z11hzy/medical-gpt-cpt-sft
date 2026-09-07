#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp
PROJECT="$ROOT/medical-gpt-cpt-sft"
PYTHON="$ROOT/venvs/medical-gpt/bin/python"
MODEL="$ROOT/models/Qwen2.5-7B-SFT-merged"
GRPO="$ROOT/medical-gpt-outputs/grpo-sft-7b-v1"
REWARD_BASE="$ROOT/models/Qwen2.5-0.5B-SFT-merged"
REWARD_ADAPTER="$ROOT/medical-gpt-outputs/rm-sft-0.5b-v1"
TEST="$PROJECT/data/processed/sft-clean-20k-v2/test/test.jsonl"
CEVAL="$PROJECT/data/eval/ceval"
REPORTS="$PROJECT/reports/results/7b"
LOG="$PROJECT/logs/formal-eval-grpo-7b.log"

mkdir -p "$REPORTS" "$PROJECT/logs"
exec > >(tee -a "$LOG") 2>&1
cd "$PROJECT"

echo "[$(date -Is)] Full held-out generation evaluation starts"
"$PYTHON" tools/eval_grpo_testset.py \
  --policy-model "$MODEL" \
  --grpo-adapter "$GRPO" \
  --reward-base "$REWARD_BASE" \
  --reward-adapter "$REWARD_ADAPTER" \
  --test-file "$TEST" \
  --output "$REPORTS/generation-sft-vs-grpo-test-500.jsonl" \
  --summary "$REPORTS/generation-sft-vs-grpo-test-500.summary.json" \
  --max-new-tokens 256 \
  --batch-size 4

echo "[$(date -Is)] C-Eval SFT baseline starts"
"$PYTHON" eval/ceval_medical.py \
  --model "$MODEL" \
  --dataset-root "$CEVAL" \
  --output "$REPORTS/ceval-sft-merged.json"

echo "[$(date -Is)] C-Eval GRPO starts"
"$PYTHON" eval/ceval_medical.py \
  --model "$MODEL" \
  --adapter "$GRPO" \
  --dataset-root "$CEVAL" \
  --output "$REPORTS/ceval-grpo-7b.json"

echo "[$(date -Is)] FORMAL_EVAL_COMPLETE"
