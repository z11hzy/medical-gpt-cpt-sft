"""Evaluate Base/CPT checkpoints on the public C-Eval medical validation set."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SUBJECTS = ("basic_medicine", "clinical_medicine", "physician")
CHOICES = ("A", "B", "C", "D")


def format_prompt(row: dict[str, object]) -> str:
    return (
        "以下是中国考试的单项选择题，请选出正确答案。\n"
        f"问题：{row['question']}\n"
        f"A. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}\n"
        "答案："
    )


@torch.inference_mode()
def predict_choice(model, tokenizer, prompt: str, device: torch.device) -> tuple[str, dict[str, float]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    encoded_choices = {
        choice: tokenizer(choice, add_special_tokens=False)["input_ids"] for choice in CHOICES
    }
    sequences = [prompt_ids + encoded_choices[choice] for choice in CHOICES]
    max_length = max(map(len, sequences))
    input_ids = torch.full(
        (len(sequences), max_length), tokenizer.pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, sequence in enumerate(sequences):
        input_ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[index, : len(sequence)] = 1

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    scores: dict[str, float] = {}
    for index, choice in enumerate(CHOICES):
        answer_ids = encoded_choices[choice]
        score = 0.0
        for offset, token_id in enumerate(answer_ids):
            score += log_probs[index, len(prompt_ids) - 1 + offset, token_id].item()
        scores[choice] = score
    return max(scores, key=scores.get), scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    device = next(model.parameters()).device

    started = time.perf_counter()
    subject_results: dict[str, dict[str, object]] = {}
    predictions: list[dict[str, object]] = []
    total_correct = 0
    total_questions = 0
    for subject in SUBJECTS:
        data_path = args.dataset_root / subject / "val-00000-of-00001.parquet"
        frame = pd.read_parquet(data_path)
        correct = 0
        for row_index, row in frame.iterrows():
            row_dict = row.to_dict()
            prediction, scores = predict_choice(model, tokenizer, format_prompt(row_dict), device)
            gold = str(row_dict["answer"]).strip()
            correct += int(prediction == gold)
            predictions.append(
                {
                    "subject": subject,
                    "id": int(row_dict.get("id", row_index)),
                    "gold": gold,
                    "prediction": prediction,
                    "choice_log_likelihood": scores,
                }
            )
        count = len(frame)
        subject_results[subject] = {
            "correct": correct,
            "total": count,
            "accuracy": correct / count,
        }
        total_correct += correct
        total_questions += count

    result = {
        "benchmark": "ceval/ceval-exam public validation split",
        "subjects": subject_results,
        "micro_accuracy": total_correct / total_questions,
        "macro_accuracy": sum(x["accuracy"] for x in subject_results.values()) / len(subject_results),
        "correct": total_correct,
        "total": total_questions,
        "model": str(args.model.resolve()),
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
        "scoring": "fixed zero-shot prompt; conditional log-likelihood of A/B/C/D",
        "choice_token_ids": {
            choice: tokenizer(choice, add_special_tokens=False)["input_ids"] for choice in CHOICES
        },
        "runtime_seconds": time.perf_counter() - started,
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "predictions"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
