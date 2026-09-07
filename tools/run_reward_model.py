"""Train a LoRA scalar reward model from standard preference JSONL."""
import argparse
import json
from pathlib import Path
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--data-dir", required=True)
p.add_argument("--output", required=True)
p.add_argument("--max-steps", type=int, default=-1)
p.add_argument("--epochs", type=float, default=1.0)
p.add_argument("--max-length", type=int, default=768)
a = p.parse_args()
tokenizer = AutoTokenizer.from_pretrained(a.model)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1, dtype=torch.bfloat16,
                                                             low_cpu_mem_usage=True)
model.config.pad_token_id = tokenizer.pad_token_id
data = load_dataset("json", data_files={"train": str(Path(a.data_dir)/"train.jsonl"),
                                         "validation": str(Path(a.data_dir)/"validation.jsonl")})
keep = {"prompt", "chosen", "rejected"}
data = data.remove_columns([c for c in data["train"].column_names if c not in keep])
cfg = RewardConfig(output_dir=a.output, max_length=a.max_length, max_steps=a.max_steps,
                   num_train_epochs=a.epochs, per_device_train_batch_size=2,
                   per_device_eval_batch_size=2, gradient_accumulation_steps=4,
                   learning_rate=1e-4, bf16=True, tf32=True, gradient_checkpointing=True,
                   eval_strategy="steps", eval_steps=100, save_strategy="steps", save_steps=250,
                   save_total_limit=2, logging_steps=10, report_to="tensorboard")
peft = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.05,
                  target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                  modules_to_save=["score"])
trainer = RewardTrainer(model=model, args=cfg, train_dataset=data["train"],
                        eval_dataset=data["validation"], processing_class=tokenizer, peft_config=peft)
result = trainer.train()
metrics = trainer.evaluate()
trainer.save_model(a.output); tokenizer.save_pretrained(a.output)
Path(a.output, "rm_summary.json").write_text(json.dumps({"train": result.metrics, "eval": metrics,
 "train_rows": len(data["train"]), "validation_rows": len(data["validation"])}, indent=2), encoding="utf-8")
print(metrics)
