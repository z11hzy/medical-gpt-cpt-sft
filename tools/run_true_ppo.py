"""Actual TRL PPO (not MedicalGPT's RLOO wrapper), for a 0.5B validation run."""
import argparse
from pathlib import Path
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl.experimental.ppo import PPOConfig, PPOTrainer

p=argparse.ArgumentParser(); p.add_argument("--sft-model",required=True); p.add_argument("--reward-adapter",required=True)
p.add_argument("--data-dir",required=True); p.add_argument("--output",required=True); p.add_argument("--episodes",type=int,default=64); p.add_argument("--response-length",type=int,default=64)
a=p.parse_args(); dtype=torch.bfloat16
tok=AutoTokenizer.from_pretrained(a.sft_model); tok.pad_token=tok.pad_token or tok.eos_token
policy=AutoModelForCausalLM.from_pretrained(a.sft_model,dtype=dtype,low_cpu_mem_usage=True)
reward_base=AutoModelForSequenceClassification.from_pretrained(a.sft_model,num_labels=1,dtype=dtype,low_cpu_mem_usage=True)
reward_base.config.pad_token_id=tok.pad_token_id
reward=PeftModel.from_pretrained(reward_base,a.reward_adapter).eval()
value=AutoModelForSequenceClassification.from_pretrained(a.sft_model,num_labels=1,dtype=dtype,low_cpu_mem_usage=True)
value.config.pad_token_id=tok.pad_token_id
data=load_dataset("json",data_files={"train":str(Path(a.data_dir)/"train.jsonl"),"validation":str(Path(a.data_dir)/"validation.jsonl")})
def tokenize(batch): return tok(batch["prompt"],padding=False)
data=data.map(tokenize,batched=True,remove_columns=data["train"].column_names)
cfg=PPOConfig(output_dir=a.output,total_episodes=a.episodes,learning_rate=3e-6,
 per_device_train_batch_size=4,gradient_accumulation_steps=1,num_ppo_epochs=1,num_mini_batches=1,
 local_rollout_forward_batch_size=4,response_length=a.response_length,stop_token="eos",
 missing_eos_penalty=1.0,bf16=True,tf32=True,gradient_checkpointing=True,logging_steps=1,
 report_to="tensorboard",save_strategy="no",eval_strategy="no",seed=20260907,data_seed=20260907)
peft=LoraConfig(task_type=TaskType.CAUSAL_LM,r=8,lora_alpha=16,lora_dropout=0.05,
 target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
trainer=PPOTrainer(args=cfg,processing_class=tok,model=policy,ref_model=None,reward_model=reward,
 value_model=value,train_dataset=data["train"],eval_dataset=data["validation"],peft_config=peft)
trainer.train(); trainer.save_model(a.output)
