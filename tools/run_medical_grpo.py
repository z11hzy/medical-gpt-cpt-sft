#!/usr/bin/env python
"""Run memory-conscious GRPO on a merged SFT policy with a learned reward model."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    set_seed,
)
from trl import GRPOConfig, GRPOTrainer


class PeakMemoryCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


class LearnedReward:
    """Score each prompt/completion with the preference reward model."""

    def __init__(self, base_path: str, adapter_path: str, max_length: int = 512):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        base = AutoModelForSequenceClassification.from_pretrained(
            base_path,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
        )
        self.model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    @torch.inference_mode()
    def __call__(self, prompts, completions, **kwargs):
        texts = [str(p) + str(c) for p, c in zip(prompts, completions)]
        batch = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        scores = self.model(**batch).logits.squeeze(-1).float().cpu().tolist()
        if isinstance(scores, float):
            scores = [scores]
        return scores


def length_guard_reward(completions, completion_ids, **kwargs):
    """Small anti-degeneracy term; the learned RM remains the main reward."""
    rewards = []
    for text, ids in zip(completions, completion_ids):
        clean = str(text).strip()
        n_tokens = len(ids)
        if not clean:
            rewards.append(-1.0)
        elif n_tokens < 8:
            rewards.append(-0.4)
        elif n_tokens < 20:
            rewards.append(-0.1)
        else:
            rewards.append(0.0)
    return rewards


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--reward-base", required=True)
    parser.add_argument("--reward-adapter", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=64)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--log-completions", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.time()
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("json", data_files=args.train_file, split="train")
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    if "prompt" not in dataset.column_names:
        raise ValueError(f"Dataset must contain 'prompt'; got {dataset.column_names}")
    if args.batch_size % args.num_generations != 0:
        raise ValueError("batch-size must be divisible by num-generations on one GPU")

    policy_tokenizer = AutoTokenizer.from_pretrained(args.policy_model, use_fast=True)
    if policy_tokenizer.pad_token_id is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token
    policy_tokenizer.padding_side = "left"

    reward_model = LearnedReward(args.reward_base, args.reward_adapter)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model_init_kwargs = {
        "quantization_config": quant_config,
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0},
        "use_cache": False,
    }
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    train_args = GRPOConfig(
        output_dir=str(output_dir),
        model_init_kwargs=model_init_kwargs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, math.ceil(args.max_steps * 0.03)),
        optim="paged_adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=0.9,
        top_p=0.95,
        repetition_penalty=1.05,
        beta=args.beta,
        loss_type="grpo",
        reward_weights=[1.0, 0.1],
        scale_rewards="group",
        logging_steps=1,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
        data_seed=args.seed,
        log_completions=args.log_completions,
        num_completions_to_print=4 if args.log_completions else None,
    )

    trainer = GRPOTrainer(
        model=args.policy_model,
        reward_funcs=[reward_model, length_guard_reward],
        args=train_args,
        train_dataset=dataset,
        processing_class=policy_tokenizer,
        peft_config=peft_config,
        callbacks=[PeakMemoryCallback()],
    )
    result = trainer.train()
    trainer.save_model(str(output_dir))
    policy_tokenizer.save_pretrained(str(output_dir))
    trainer.save_state()

    metrics = dict(result.metrics)
    metrics.update(
        {
            "runtime_wall_seconds": time.time() - started,
            "peak_gpu_memory_gib": (
                torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
            ),
            "policy_model": args.policy_model,
            "reward_base": args.reward_base,
            "reward_adapter": args.reward_adapter,
            "train_file": args.train_file,
            "train_samples": len(dataset),
            "num_generations": args.num_generations,
            "max_completion_length": args.max_completion_length,
            "beta": args.beta,
            "loss_type": "grpo",
        }
    )
    with (output_dir / "grpo_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print("GRPO_SUMMARY=" + json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
