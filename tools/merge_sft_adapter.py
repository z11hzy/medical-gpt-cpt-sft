"""Merge an SFT LoRA into its causal-LM base for downstream RL stages."""
import argparse
import json
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--base", required=True)
p.add_argument("--adapter", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, low_cpu_mem_usage=True)
merged = PeftModel.from_pretrained(base, a.adapter).merge_and_unload(safe_merge=True)
merged.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(a.adapter).save_pretrained(out)
(out / "merge_manifest.json").write_text(json.dumps({"base": a.base, "adapter": a.adapter}, indent=2), encoding="utf-8")
print(out)
