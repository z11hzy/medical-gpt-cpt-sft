"""在从未用于调参的官方 CPT test split 上计算因果语言模型 loss/PPL。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_and_pack(path: Path, tokenizer, block_size: int = 256) -> torch.Tensor:
    texts = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"测试集第 {line_no} 行没有有效 text")
        texts.append(text.strip())
    if len(texts) != 500:
        raise ValueError(f"预期 500 篇测试文本，实际 {len(texts)}")
    # 与 MedicalGPT 相同：每批 1000 篇拼接，文档之间补 EOS，丢弃批尾不足一块的 token。
    blocks = []
    for start in range(0, len(texts), 1000):
        encoded = tokenizer(texts[start:start + 1000])["input_ids"]
        merged = []
        for ids in encoded:
            merged.extend(ids)
            if not ids or ids[-1] != tokenizer.eos_token_id:
                merged.append(tokenizer.eos_token_id)
        kept = len(merged) // block_size * block_size
        blocks.extend(merged[i:i + block_size] for i in range(0, kept, block_size))
    result = torch.tensor(blocks, dtype=torch.long)
    if result.shape != (733, 256):
        raise ValueError(f"测试集分块不符合审计值：(733, 256)，实际 {tuple(result.shape)}")
    return result


@torch.inference_mode()
def evaluate(model, blocks: torch.Tensor, batch_size: int = 2) -> tuple[float, float, int]:
    loader = DataLoader(TensorDataset(blocks), batch_size=batch_size, shuffle=False)
    device = next(model.parameters()).device
    weighted_loss = 0.0
    predicted_tokens = 0
    model.eval()
    for (input_ids,) in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=input_ids)
        count = input_ids.shape[0] * (input_ids.shape[1] - 1)
        weighted_loss += output.loss.float().item() * count
        predicted_tokens += count
    loss = weighted_loss / predicted_tokens
    return loss, math.exp(loss), predicted_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"不覆盖已有测试结果：{args.output}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, local_files_only=True)
    blocks = load_and_pack(args.test_file, tokenizer)
    packed_sha = hashlib.sha256(blocks.numpy().tobytes()).hexdigest()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True, local_files_only=True
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    started = time.perf_counter()
    loss, perplexity, predicted_tokens = evaluate(model, blocks)
    result = {
        "split": "shibing624/medical pretrain/test_encyclopedia.json",
        "documents": 500,
        "blocks": len(blocks),
        "block_size": 256,
        "input_tokens": blocks.numel(),
        "predicted_tokens": predicted_tokens,
        "batch_size": 2,
        "eval_loss": loss,
        "perplexity": perplexity,
        "model": str(args.model.resolve()),
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
        "test_file_sha256": sha256(args.test_file),
        "packed_test_sha256": packed_sha,
        "runtime_seconds": time.perf_counter() - started,
        "method": "causal next-token cross-entropy; MedicalGPT-compatible EOS packing in 256-token blocks",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
