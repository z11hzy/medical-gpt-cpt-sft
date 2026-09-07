#!/usr/bin/env python
"""Full held-out generation evaluation for the SFT baseline vs GRPO adapter."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

SYSTEM = (
    "你是健康信息助手。提供通俗、审慎的健康教育，不替代医生诊断或处方；"
    "遇到紧急危险信号应建议立即寻求急救或线下医疗帮助。"
)


def read_prompts(path: str):
    rows = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        conversations = item.get("conversations", [])
        human = next((x.get("value", "") for x in conversations if x.get("from") == "human"), "")
        reference = next((x.get("value", "") for x in conversations if x.get("from") == "gpt"), "")
        if human:
            rows.append({"id": index, "prompt": human, "reference_response": reference})
    return rows


def make_text(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def generate_batch(model, tokenizer, texts, max_new_tokens):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    responses = []
    for index in range(len(texts)):
        new_ids = output[index, inputs["input_ids"].shape[1] :]
        responses.append(
            {
                "text": tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
                "tokens": int(len(new_ids)),
                "terminated": bool(len(new_ids) and new_ids[-1].item() == tokenizer.eos_token_id),
            }
        )
    return responses


def score_rows(rows, reward_base, reward_adapter):
    tokenizer = AutoTokenizer.from_pretrained(reward_base, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base = AutoModelForSequenceClassification.from_pretrained(
        reward_base, num_labels=1, dtype=torch.bfloat16, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(base, reward_adapter, is_trainable=False).eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    texts = []
    locations = []
    for index, row in enumerate(rows):
        prompt = make_text(tokenizer, row["prompt"])
        for key in ("sft", "grpo"):
            texts.append(prompt + row[f"{key}_response"])
            locations.append((index, key))
    values = []
    with torch.inference_mode():
        for start in range(0, len(texts), 16):
            batch = tokenizer(
                texts[start : start + 16], add_special_tokens=False, padding=True,
                truncation=True, max_length=512, return_tensors="pt"
            ).to(model.device)
            result = model(**batch).logits.squeeze(-1).float().cpu().flatten().tolist()
            values.extend(result if isinstance(result, list) else [result])
    for (index, key), value in zip(locations, values):
        rows[index][f"{key}_reward"] = float(value)


def avg(values):
    return statistics.fmean(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--grpo-adapter", required=True)
    parser.add_argument("--reward-base", required=True)
    parser.add_argument("--reward-adapter", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    rows = read_prompts(args.test_file)
    tokenizer = AutoTokenizer.from_pretrained(args.policy_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(args.policy_model, dtype=torch.bfloat16, low_cpu_mem_usage=True).cuda().eval()
    model = PeftModel.from_pretrained(base, args.grpo_adapter, is_trainable=False).eval()
    texts = [make_text(tokenizer, row["prompt"]) for row in rows]
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        with model.disable_adapter():
            sft = generate_batch(model, tokenizer, texts[start : start + args.batch_size], args.max_new_tokens)
        grpo = generate_batch(model, tokenizer, texts[start : start + args.batch_size], args.max_new_tokens)
        for row, sft_result, grpo_result in zip(batch_rows, sft, grpo):
            row.update(
                {
                    "sft_response": sft_result["text"], "sft_tokens": sft_result["tokens"],
                    "sft_terminated": sft_result["terminated"], "grpo_response": grpo_result["text"],
                    "grpo_tokens": grpo_result["tokens"], "grpo_terminated": grpo_result["terminated"],
                }
            )
        if (start // args.batch_size + 1) % 10 == 0:
            print(f"GENERATION_PROGRESS={min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)

    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    score_rows(rows, args.reward_base, args.reward_adapter)
    deltas = [row["grpo_reward"] - row["sft_reward"] for row in rows]
    summary = {
        "samples": len(rows), "decoding": "greedy", "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size, "sft_reward_mean": avg([x["sft_reward"] for x in rows]),
        "grpo_reward_mean": avg([x["grpo_reward"] for x in rows]), "reward_delta_mean": avg(deltas),
        "grpo_reward_wins": sum(x > 1e-6 for x in deltas), "ties": sum(abs(x) <= 1e-6 for x in deltas),
        "sft_reward_wins": sum(x < -1e-6 for x in deltas),
        "sft_mean_tokens": avg([x["sft_tokens"] for x in rows]), "grpo_mean_tokens": avg([x["grpo_tokens"] for x in rows]),
        "sft_termination_rate": avg([float(x["sft_terminated"]) for x in rows]),
        "grpo_termination_rate": avg([float(x["grpo_terminated"]) for x in rows]),
        "important_limitation": "Automatic reward comparison reuses the training reward model and is not an independent medical judge.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FULL_EVAL_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
