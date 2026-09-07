#!/usr/bin/env python
"""Compare SFT and DPO generation on the same held-out prompts.

The generation comparison is deterministic by default.  An optional reward
model can provide a relative score, but that score must not be treated as an
independent medical evaluation.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


SYSTEM = (
    "你是健康信息助手。提供通俗、审慎的健康教育，不替代医生诊断或处方；"
    "遇到紧急危险信号应建议立即寻求急救或线下医疗帮助。"
)


def read_prompts(path: Path, limit: int = -1) -> list[dict[str, object]]:
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        conversations = item.get("conversations", [])
        prompt = next(
            (
                turn.get("value", turn.get("content", ""))
                for turn in conversations
                if turn.get("from") in {"human", "user"} or turn.get("role") == "user"
            ),
            "",
        )
        if prompt:
            rows.append({"id": index, "prompt": prompt})
        if limit > 0 and len(rows) >= limit:
            break
    if not rows:
        raise ValueError(f"No user prompts found in {path}")
    return rows


def make_text(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def generate_batch(model, tokenizer, texts, adapter_name: str, max_new_tokens: int):
    model.set_adapter(adapter_name)
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(next(model.parameters()).device)
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
                "terminated": bool(
                    len(new_ids)
                    and tokenizer.eos_token_id is not None
                    and new_ids[-1].item() == tokenizer.eos_token_id
                ),
            }
        )
    return responses


def mean(values):
    return statistics.fmean(values) if values else None


@torch.inference_mode()
def score_with_reward_model(rows, reward_base: str, reward_adapter: str):
    tokenizer = AutoTokenizer.from_pretrained(reward_base, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base = AutoModelForSequenceClassification.from_pretrained(
        reward_base,
        num_labels=1,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    reward_model = PeftModel.from_pretrained(base, reward_adapter).eval()
    reward_model.config.pad_token_id = tokenizer.pad_token_id

    texts = []
    locations = []
    for index, row in enumerate(rows):
        prompt = make_text(tokenizer, row["prompt"])
        for name in ("sft", "dpo"):
            texts.append(prompt + row[f"{name}_response"])
            locations.append((index, name))

    values = []
    for start in range(0, len(texts), 16):
        batch = tokenizer(
            texts[start : start + 16],
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=768,
            return_tensors="pt",
        ).to(next(reward_model.parameters()).device)
        logits = reward_model(**batch).logits.squeeze(-1).float().cpu().flatten().tolist()
        values.extend(logits if isinstance(logits, list) else [logits])

    for (index, name), value in zip(locations, values):
        rows[index][f"{name}_reward"] = float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--sft-adapter", required=True)
    parser.add_argument("--dpo-adapter", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--print-samples", type=int, default=10)
    parser.add_argument("--reward-base")
    parser.add_argument("--reward-adapter")
    args = parser.parse_args()
    if bool(args.reward_base) != bool(args.reward_adapter):
        parser.error("--reward-base and --reward-adapter must be provided together")

    rows = read_prompts(Path(args.test_file), args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, args.sft_adapter, adapter_name="sft")
    model.load_adapter(args.dpo_adapter, adapter_name="dpo")
    model.eval()

    texts = [make_text(tokenizer, row["prompt"]) for row in rows]
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        batch_texts = texts[start : start + args.batch_size]
        sft = generate_batch(model, tokenizer, batch_texts, "sft", args.max_new_tokens)
        dpo = generate_batch(model, tokenizer, batch_texts, "dpo", args.max_new_tokens)
        for row, sft_result, dpo_result in zip(batch_rows, sft, dpo):
            row.update(
                {
                    "sft_response": sft_result["text"],
                    "sft_tokens": sft_result["tokens"],
                    "sft_terminated": sft_result["terminated"],
                    "dpo_response": dpo_result["text"],
                    "dpo_tokens": dpo_result["tokens"],
                    "dpo_terminated": dpo_result["terminated"],
                }
            )
        done = min(start + args.batch_size, len(rows))
        if done % max(args.batch_size * 10, 1) == 0 or done == len(rows):
            print(f"GENERATION_PROGRESS={done}/{len(rows)}", flush=True)

    if args.reward_base:
        del model, base
        gc.collect()
        torch.cuda.empty_cache()
        score_with_reward_model(rows, args.reward_base, args.reward_adapter)

    summary = {
        "samples": len(rows),
        "decoding": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "sft_mean_tokens": mean([row["sft_tokens"] for row in rows]),
        "dpo_mean_tokens": mean([row["dpo_tokens"] for row in rows]),
        "sft_termination_rate": mean([float(row["sft_terminated"]) for row in rows]),
        "dpo_termination_rate": mean([float(row["dpo_terminated"]) for row in rows]),
        "reward_model_used": bool(args.reward_base),
        "important_limitation": (
            "Reward comparison, when enabled, reuses the training reward model and is not an independent medical judge."
        ),
    }
    if args.reward_base:
        deltas = [row["dpo_reward"] - row["sft_reward"] for row in rows]
        summary.update(
            {
                "sft_reward_mean": mean([row["sft_reward"] for row in rows]),
                "dpo_reward_mean": mean([row["dpo_reward"] for row in rows]),
                "reward_delta_mean": mean(deltas),
                "dpo_reward_wins": sum(delta > 1e-6 for delta in deltas),
                "ties": sum(abs(delta) <= 1e-6 for delta in deltas),
                "sft_reward_wins": sum(delta < -1e-6 for delta in deltas),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for row in rows[: args.print_samples]:
        print(f"\n=== SAMPLE {row['id']} ===\n问题：{row['prompt']}\n\n[SFT]\n{row['sft_response']}\n\n[DPO]\n{row['dpo_response']}")
    print("SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
