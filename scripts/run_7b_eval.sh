#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/set_env.sh"

RESULTS_DIR="$PROJECT_ROOT/reports/results/7b"
mkdir -p "$RESULTS_DIR"
CPT_ADAPTER="$OUTPUT_ROOT/pt-full20k-7b-v2"
BASE_SFT_ADAPTER="$OUTPUT_ROOT/sft-base-20k-7b-v2"
CPT_SFT_ADAPTER="$OUTPUT_ROOT/sft-cpt-20k-7b-v2"
CPT_TEST="$PROJECT_ROOT/data/raw/shibing624-medical/pretrain/test_encyclopedia.json"
CEVAL_ROOT="$PROJECT_ROOT/data/eval/ceval"

"$PYTHON_BIN" "$PROJECT_ROOT/eval/test_lm.py" --model "$MODEL_7B_PATH" --test-file "$CPT_TEST" --output "$RESULTS_DIR/test-lm-base.json"
"$PYTHON_BIN" "$PROJECT_ROOT/eval/test_lm.py" --model "$MODEL_7B_PATH" --adapter "$CPT_ADAPTER" --test-file "$CPT_TEST" --output "$RESULTS_DIR/test-lm-cpt.json"

for branch in base cpt; do
  if [[ "$branch" == "base" ]]; then adapter="$BASE_SFT_ADAPTER"; else adapter="$CPT_SFT_ADAPTER"; fi
  "$PYTHON_BIN" "$MEDICALGPT_ROOT/training/supervised_finetuning.py" \
    --model_name_or_path "$MODEL_7B_PATH" \
    --validation_file_dir "$PROJECT_ROOT/data/processed/sft-clean-20k-v2/test" \
    --output_dir "$OUTPUT_ROOT/eval-sft-$branch-test-7b-v2" \
    --do_eval --use_peft True --peft_path "$adapter" --train_on_inputs False --model_max_length 512 \
    --max_eval_samples 500 --per_device_eval_batch_size 1 --torch_dtype bfloat16 --bf16 \
    --dataloader_num_workers 0 --report_to none --disable_tqdm True
done

declare -a NAMES=(base cpt sft-base sft-cpt)
declare -a ADAPTERS=("" "$CPT_ADAPTER" "$BASE_SFT_ADAPTER" "$CPT_SFT_ADAPTER")
for index in "${!NAMES[@]}"; do
  args=(--model "$MODEL_7B_PATH" --dataset-root "$CEVAL_ROOT" --output "$RESULTS_DIR/ceval-${NAMES[$index]}.json")
  if [[ -n "${ADAPTERS[$index]}" ]]; then args+=(--adapter "${ADAPTERS[$index]}"); fi
  "$PYTHON_BIN" "$PROJECT_ROOT/eval/ceval_medical.py" "${args[@]}"
done

echo "7B evaluation completed: $RESULTS_DIR"
