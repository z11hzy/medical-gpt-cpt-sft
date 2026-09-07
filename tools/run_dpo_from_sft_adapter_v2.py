"""Memory-conscious DPO: start from an SFT LoRA and retain it as the reference."""
import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sft-adapter", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    set_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.sft_adapter, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16,
                                                  trust_remote_code=True, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, args.sft_adapter, adapter_name="policy", is_trainable=True)
    model.load_adapter(args.sft_adapter, adapter_name="reference", is_trainable=False)
    model.set_adapter("policy")
    for name, parameter in model.named_parameters():
        if ".reference." in name:
            parameter.requires_grad = False
    model.config.use_cache = False
    model.enable_input_require_grads()
    data = load_dataset("json", data_files={"train": str(Path(args.data_dir) / "train.jsonl"),
                                               "validation": str(Path(args.data_dir) / "validation.jsonl")})
    keep = ["prompt", "chosen", "rejected"]
    data = data.remove_columns([c for c in data["train"].column_names if c not in keep])
    config = DPOConfig(
        output_dir=str(output), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, beta=args.beta, max_length=args.max_length,
        bf16=True, tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=10,
        eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="steps",
        save_steps=args.save_steps, save_total_limit=2, report_to="tensorboard",
        remove_unused_columns=False, model_adapter_name="policy", ref_adapter_name="reference",
        seed=args.seed, data_seed=args.seed)
    trainer = DPOTrainer(model=model, ref_model=None, args=config, train_dataset=data["train"],
                         eval_dataset=data["validation"], processing_class=tokenizer)
    train_result = trainer.train()
    trainer.save_state()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    summary = {"method": "DPO from Base->SFT LoRA; frozen SFT reference adapter",
               "model_path": args.model_path, "sft_adapter": args.sft_adapter,
               "data_dir": args.data_dir, "train_rows": len(data["train"]),
               "validation_rows": len(data["validation"]), "max_length": args.max_length,
               "beta": args.beta, "train_metrics": train_result.metrics}
    (output / "dpo_run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
