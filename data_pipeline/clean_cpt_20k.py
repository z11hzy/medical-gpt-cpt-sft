"""保守清洗已审计的 20k 子集；保留原文件、隔离记录及逐条来源。"""
from __future__ import annotations

import collections
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "e581b66eefb8fd8c98b6343eeba6b9395fd3f3ce709b4ffe54dbe72565c52304"
VALID_SHA = "7d1d963c0f3999175e8724652cffd5c58c2025f9ad50384e40de3764de272172"
OUT = ROOT / "data/processed/pretrain-clean-20k-v2"
# 仅针对锁定 SHA 的源文件行号；不是自动医学事实判定。
EXCLUSIONS = {
    38: ("quality_review", "正文含损坏词语和未经核实的绝对化治疗承诺，隔离待复核"),
    1073: ("quality_review", "直接推荐偏方并承诺短期效果，隔离待专业复核"),
    3649: ("off_scope", "仅植物分类，当前医学知识实验暂不使用"),
    6754: ("corrupted_text", "已确认 U+FFFD 和私用字符混杂乱码"),
    8025: ("corrupted_text", "已确认大量不自然词语替换及损坏句子"),
    9013: ("off_scope", "人物生平，当前医学知识实验暂不使用"),
    12313: ("validation_overlap", "与官方验证集第 399 行明显共享正文"),
    13707: ("validation_overlap", "与官方验证集第 233 行明显共享正文"),
    8477: ("reviewed_near_duplicate", "与第 487 行同主题、主体正文重复"),
    13336: ("reviewed_near_duplicate", "与第 2877 行主体正文重复，主要为标题改写"),
    6212: ("reviewed_near_duplicate", "与第 4781 行同主题、主体正文重复，仅少量增补"),
    19787: ("reviewed_near_duplicate", "与第 7384 行主体正文重复，主要为标题改写"),
    13725: ("reviewed_near_duplicate", "与第 10447 行重复且含肾癌误写为深爱"),
    17996: ("reviewed_near_duplicate", "与第 14856 行主体正文重复，仅标题与标点变化"),
    19916: ("reviewed_near_duplicate", "与第 15358 行主体正文重复，仅标题改写"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_records(records, exclusions):
    kept, rejected, provenance, seen = [], [], [], {}
    for line, row in enumerate(records, 1):
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"源文件第 {line} 行不是有效非空文本")
        text = text.strip()  # 不改写正文、剂量、单位、标点或否定词。
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        reason = exclusions.get(line)
        if reason is None and key in seen:
            reason = ("exact_duplicate", f"与源文件第 {seen[key]} 行完全重复")
        if reason:
            rejected.append({"source_line": line, "reason": reason[0], "detail": reason[1], "text": text})
        else:
            seen[key] = line
            kept.append({"text": text})
            provenance.append({"output_line": len(kept), "source_line": line, "text_sha256": key})
    return kept, rejected, provenance


def packing_stats(records, tokenizer, block_size=256):
    blocks = tokens = dropped = 0
    for start in range(0, len(records), 1000):
        ids = tokenizer([x["text"] for x in records[start:start + 1000]])["input_ids"]
        size = sum(len(x) + (not x or x[-1] != tokenizer.eos_token_id) for x in ids)
        tokens += size
        kept = size // block_size * block_size if size >= block_size else size
        if kept % block_size:
            raise ValueError("出现非完整块，需要显式处理而不是误报 token 数")
        blocks += kept // block_size
        dropped += size - kept
    return {"documents": len(records), "tokens_with_eos": tokens, "blocks": blocks,
            "input_tokens": blocks * block_size, "dropped_tail_tokens": dropped}


def main():
    from transformers import AutoTokenizer
    train = ROOT / "data/processed/pretrain/train/train.jsonl"
    valid = ROOT / "data/processed/pretrain/valid/valid.jsonl"
    if digest(train) != SOURCE_SHA or digest(valid) != VALID_SHA:
        raise ValueError("源数据 SHA 不匹配，禁止套用旧行号清洗决定")
    if OUT.exists():
        raise FileExistsError(f"不覆盖已有数据版本：{OUT}")
    rows = [json.loads(x) for x in train.read_text(encoding="utf-8").splitlines()]
    vals = [json.loads(x) for x in valid.read_text(encoding="utf-8").splitlines()]
    kept, rejected, provenance = clean_records(rows, EXCLUSIONS)
    assert len(kept) + len(rejected) == len(rows) == 20000
    assert not ({x["text"] for x in kept} & {x["text"] for x in vals})
    assert not any("\ufffd" in x["text"] for x in kept)
    tok = AutoTokenizer.from_pretrained(ROOT / "models/Qwen2.5-0.5B", use_fast=False, local_files_only=True)
    train_stats, val_stats = packing_stats(kept, tok), packing_stats(vals, tok)
    steps = math.ceil(train_stats["blocks"] / 8)
    jsonl(OUT / "train/train.jsonl", kept)
    # 字节级保留官方验证文件；所有实验评测同一版本。
    (OUT / "valid").mkdir(parents=True)
    (OUT / "valid/valid.jsonl").write_bytes(valid.read_bytes())
    jsonl(OUT / "quarantine.jsonl", rejected)
    jsonl(OUT / "provenance.jsonl", provenance)
    audit = json.loads((ROOT / "reports/results/cpt-audit.json").read_text(encoding="utf-8"))
    reviews = []
    for pair in audit["near_duplicates"]["train_pair_examples"]:
        excluded = [i for i in pair if i in EXCLUSIONS]
        reviews.append({"source_lines": pair, "excluded_lines": excluded,
                        "decision": "排除已复核的重复副本" if excluded else "保留：实体、剂型、人群或主题差异，不按相似度自动合并"})
    dump(OUT / "near_duplicate_review.json", reviews)
    manifest = {
        "version": OUT.name, "source_train_sha256": SOURCE_SHA, "source_valid_sha256": VALID_SHA,
        "source_documents": len(rows), "excluded_documents": len(rejected),
        "exclusions_by_reason": dict(collections.Counter(x["reason"] for x in rejected)),
        "train": train_stats, "valid": val_stats,
        "train_sha256": digest(OUT / "train/train.jsonl"), "valid_sha256": digest(OUT / "valid/valid.jsonl"),
        "block_size": 256, "effective_batch": 8, "epochs": 1, "expected_steps": steps,
        "warmup_steps": math.ceil(steps * 0.05),
        "limitations": ["保守工程清洗，不是逐条医学事实认证", "未按关键词批量删除；其余质量候选尚未全部专业复核",
                        "近重复检测有限召回，保留实体/剂型差异；不能保证消除全部泄漏", "不补抽数据凑满 20000 条"],
    }
    dump(OUT / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
