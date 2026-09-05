"""流式审计 195 万条中文医学 SFT，并生成可复现的 20k ShareGPT 子集。"""
from __future__ import annotations

import argparse
import collections
import hashlib
import heapq
import json
import math
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/shibing624-medical/finetune"
CPT = ROOT / "data/processed/pretrain-clean-20k-v2/train/train.jsonl"
CEVAL = ROOT / "data/eval/ceval"
OUT = ROOT / "data/processed/sft-clean-20k-v2"
RESULT = ROOT / "reports/results/sft-audit-v2.json"
SOURCE_SHA = "98f111016800f98ef4e5c5e29f9260c8ce8ef4331897523d6c3a61003957fed3"
VALID_SHA = "397ce9692f0f5782227781ffc2cd9cbb18c105e03817000aedc97195584e6992"
TEST_SHA = "c3876d82565cb9958be72883a096311a5081fb6704f8607ec043921baf57b1a9"
MANUAL_EXCLUSIONS = {
    256529: "与官方 SFT test 第 167 条近重复",
    641506: "与官方 SFT valid 第 226 条近重复",
    552412: "回答把儿童脱发错配为痤疮且语言明显损坏",
    659609: "与另一条儿童脱发问答近重复且回答错配",
    609482: "机构价格营销内容",
    115026: "引导拨打专家咨询热线",
    1317386: "引导拨打专家咨询热线",
    633626: "医院外链营销内容",
    1331179: "产品官网内容",
    1330049: "医院外链内容",
    1182746: "推荐自行剪除鸡眼及刺激性偏方",
    1577127: "未经依据承诺中医可根治鲜红斑痣",
    911922: "血毒排毒叙述并承诺根治荨麻疹",
    444567: "推荐大量抗癌秘方并声称无副作用",
    1357445: "推荐肝硬化腹水秘方并声称效果非常好",
    41500: "推荐偏方辅助治疗且含未经核实因果断言",
    315814: "包含高风险肿瘤偏方、罂粟壳和百分百有效率",
    1520132: "建议偏方作为子宫肌瘤辅助治疗",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text).lower() if c.isalnum())


def validate(row: object, where: str) -> tuple[str, str, str]:
    if not isinstance(row, dict):
        raise ValueError(f"{where}: record is not an object")
    values = []
    for key in ("instruction", "input", "output"):
        value = row.get(key)
        if not isinstance(value, str):
            raise ValueError(f"{where}: {key} is not a string")
        values.append(value.strip())
    if not values[0] or not values[2]:
        raise ValueError(f"{where}: empty instruction/output")
    return tuple(values)


def user_text(instruction: str, extra: str) -> str:
    return instruction + ("\n\n" + extra if extra else "")


def read_split(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                ins, extra, answer = validate(json.loads(line), f"{path}:{line_no}")
                rows.append({"instruction": ins, "input": extra, "output": answer})
    return rows


def to_sharegpt(row: dict[str, str]) -> dict[str, object]:
    return {"conversations": [
        {"from": "human", "value": user_text(row["instruction"], row["input"])},
        {"from": "gpt", "value": row["output"]},
    ]}


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def token_stats(rows: list[dict[str, str]], tokenizer, max_length: int = 512) -> dict[str, object]:
    lengths, target_lengths = [], []
    truncated = 0
    for row in rows:
        user = user_text(row["instruction"], row["input"])
        source = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
        )
        source_ids = tokenizer.encode(source, add_special_tokens=True)
        target_ids = tokenizer.encode(row["output"], add_special_tokens=False)
        total = len(source_ids) + len(target_ids)
        source_cap = int(max_length * len(source_ids) / total)
        target_cap = int(max_length * len(target_ids) / total)
        was_truncated = len(source_ids) > source_cap or len(target_ids) > target_cap - 1
        source_ids = source_ids[:source_cap]
        target_ids = target_ids[:max(0, target_cap - 1)]
        if source_ids and source_ids[0] == tokenizer.eos_token_id:
            source_ids = source_ids[1:]
        if target_ids and target_ids[-1] == tokenizer.eos_token_id:
            target_ids = target_ids[:-1]
        lengths.append(len(source_ids) + len(target_ids) + 1)
        target_lengths.append(len(target_ids) + 1)
        truncated += was_truncated
    percentiles = [0, 50, 90, 95, 99, 100]
    return {
        "samples": len(rows), "max_length": max_length,
        "input_token_sum_after_truncation": int(sum(lengths)),
        "assistant_label_token_sum": int(sum(target_lengths)),
        "truncated_samples": truncated,
        "truncated_percent": truncated / len(rows) * 100,
        "input_length_percentiles": dict(zip(
            ["min", "p50", "p90", "p95", "p99", "max"],
            map(float, np.percentile(lengths, percentiles)))),
        "assistant_length_percentiles": dict(zip(
            ["min", "p50", "p90", "p95", "p99", "max"],
            map(float, np.percentile(target_lengths, percentiles)))),
    }


def near_duplicate_audit(train: list[dict[str, str]], heldout: list[dict[str, str]]) -> dict[str, object]:
    train_text = [norm(user_text(x["instruction"], x["input"]) + x["output"]) for x in train]
    heldout_text = [norm(user_text(x["instruction"], x["input"]) + x["output"]) for x in heldout]
    vectorizer = HashingVectorizer(analyzer="char", ngram_range=(3, 4), n_features=2**20,
                                   alternate_sign=False, norm=None, dtype=np.float32)
    tfidf = TfidfTransformer()
    matrix = tfidf.fit_transform(vectorizer.transform(train_text))
    nn = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute", n_jobs=2).fit(matrix)
    pairs = set()
    for start in range(0, len(train), 250):
        distances, indices = nn.kneighbors(matrix[start:start + 250])
        for local, (ds, js) in enumerate(zip(distances, indices)):
            i = start + local
            for distance, j in zip(ds, js):
                if i != j and distance <= 0.03 and train_text[i] != train_text[int(j)]:
                    pairs.add((min(i + 1, int(j) + 1), max(i + 1, int(j) + 1), float(1-distance)))
    held_matrix = tfidf.transform(vectorizer.transform(heldout_text))
    distances, indices = nn.kneighbors(held_matrix, n_neighbors=1)
    cross = [{"heldout_index": i + 1, "train_line": int(indices[i, 0]) + 1,
              "similarity": float(1 - distances[i, 0])}
             for i in range(len(heldout)) if distances[i, 0] <= 0.03]
    return {"method": "char 3/4-gram hashed TF-IDF; cosine>=0.97; candidate screening only",
            "train_candidate_pairs": len(pairs), "train_pair_examples": sorted(pairs)[:50],
            "heldout_train_candidates": cross}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_path, valid_path, test_path = (RAW / "train_zh_0.json", RAW / "valid_zh_0.json", RAW / "test_zh_0.json")
    for path, expected in ((train_path, SOURCE_SHA), (valid_path, VALID_SHA), (test_path, TEST_SHA)):
        if sha256(path) != expected:
            raise ValueError(f"源文件 SHA 不匹配：{path}")
    if OUT.exists() or RESULT.exists():
        raise FileExistsError("不覆盖已有 SFT 数据版本或审计结果")
    valid, test = read_split(valid_path), read_split(test_path)
    heldout = valid + test
    heldout_prompts = {norm(user_text(x["instruction"], x["input"])) for x in heldout}
    cpt_norms = set()
    with CPT.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cpt_norms.add(norm(json.loads(line)["text"]))
    ceval_questions = set()
    import pandas as pd
    for subject in ("basic_medicine", "clinical_medicine", "physician"):
        frame = pd.read_parquet(CEVAL / subject / "val-00000-of-00001.parquet")
        ceval_questions.update(norm(str(x)) for x in frame["question"])

    heap, seen_pairs, seen_prompts = [], set(), set()
    counts = collections.Counter()
    exclusion_examples: dict[str, list[int]] = collections.defaultdict(list)
    quality_patterns = {
        "replacement_character": re.compile("\ufffd"),
        "absolute_claim": re.compile(r"根治|包治|百分之百|百分百|无副作用|保证治愈|一定能治"),
        "folk_remedy": re.compile(r"偏方|秘方|祖传"),
        "contact_or_promotion": re.compile(r"加微信|加QQ|联系电话|咨询热线|点击购买"),
        "phone_like": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
        "url": re.compile(r"https?://|www\."),
    }
    started = time.perf_counter()
    with train_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            counts["records"] += 1
            try:
                row_raw = json.loads(line)
                ins, extra, answer = validate(row_raw, f"train:{line_no}")
            except Exception:
                counts["invalid"] += 1
                if len(exclusion_examples["invalid"]) < 20:
                    exclusion_examples["invalid"].append(line_no)
                continue
            row = {"instruction": ins, "input": extra, "output": answer}
            user = user_text(ins, extra)
            combined = user + "\n" + answer
            prompt_norm = norm(user)
            pair_norm = norm(user + answer)
            pair_digest = hashlib.sha256(pair_norm.encode("utf-8")).digest()
            prompt_digest = hashlib.sha256(prompt_norm.encode("utf-8")).digest()
            if prompt_digest in seen_prompts:
                counts["duplicate_prompt_excess"] += 1
            else:
                seen_prompts.add(prompt_digest)
            duplicate = pair_digest in seen_pairs
            seen_pairs.add(pair_digest)
            flags = {
                "exact_normalized_duplicate": duplicate,
                "heldout_prompt_overlap": prompt_norm in heldout_prompts,
                "cpt_pair_overlap": pair_norm in cpt_norms,
                "ceval_prompt_overlap": prompt_norm in ceval_questions,
                "replacement_character": "\ufffd" in user or "\ufffd" in answer,
                "url_or_contact": bool(quality_patterns["url"].search(combined)
                                       or quality_patterns["contact_or_promotion"].search(combined)
                                       or quality_patterns["phone_like"].search(combined)),
                "manual_review_exclusion": line_no in MANUAL_EXCLUSIONS,
            }
            for key, hit in flags.items():
                if hit:
                    counts[key] += 1
                    if len(exclusion_examples[key]) < 20:
                        exclusion_examples[key].append(line_no)
            for key, pattern in quality_patterns.items():
                if pattern.search(combined):
                    counts["quality_" + key] += 1
            if len(answer) < 10:
                counts["quality_answer_under_10_chars"] += 1
            if len(answer) > 4000:
                counts["quality_answer_over_4000_chars"] += 1
            if any(flags.values()):
                counts["excluded_union"] += 1
                continue
            canonical = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            score = int.from_bytes(hashlib.sha256(str(args.seed).encode("ascii") + canonical.encode("utf-8")).digest(), "big")
            item = (-score, line_no, canonical)
            if len(heap) < args.sample_size:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
            if line_no % 100_000 == 0:
                print(f"scanned={line_no:,} eligible={counts['records']-counts['excluded_union']:,} selected={len(heap):,}", flush=True)
    selected_items = sorted(heap, reverse=True)
    selected = [json.loads(x[2]) for x in selected_items]
    source_lines = [x[1] for x in selected_items]
    if len(selected) != args.sample_size:
        raise ValueError(f"清洗后不足 {args.sample_size} 条")

    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/Qwen2.5-0.5B", use_fast=False, local_files_only=True)
    selected_stats = token_stats(selected, tokenizer)
    valid_stats, test_stats = token_stats(valid, tokenizer), token_stats(test, tokenizer)
    near = near_duplicate_audit(selected, heldout)
    write_jsonl(OUT / "train/train.jsonl", map(to_sharegpt, selected))
    write_jsonl(OUT / "valid/valid.jsonl", map(to_sharegpt, valid))
    write_jsonl(OUT / "test/test.jsonl", map(to_sharegpt, test))
    write_jsonl(OUT / "provenance.jsonl", (
        {"output_line": i + 1, "source_line": source_lines[i],
         "source_record_sha256": hashlib.sha256(json.dumps(selected[i], ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
        for i in range(len(selected))))
    steps = math.ceil(len(selected) / 4)
    result = {
        "source": {"revision": "6e219f1a14856833ee436063d3b73c5f1ab9cfb9", "train_sha256": SOURCE_SHA,
                   "valid_sha256": VALID_SHA, "test_sha256": TEST_SHA},
        "selection": {"method": "lowest SHA-256(seed + canonical JSON) after objective exclusions",
                      "seed": args.seed, "requested": args.sample_size, "selected": len(selected)},
        "counts": dict(counts), "exclusion_first_source_lines": dict(exclusion_examples),
        "automatic_exclusions": ["invalid", "exact_normalized_duplicate", "heldout_prompt_overlap",
                                 "cpt_pair_overlap", "ceval_prompt_overlap", "replacement_character",
                                 "url_or_contact", "manual_review_exclusion"],
        "manual_exclusions": MANUAL_EXCLUSIONS,
        "quality_flags_are_not_automatic_exclusions": list(quality_patterns) + ["answer_under_10_chars", "answer_over_4000_chars"],
        "token_stats": {"train": selected_stats, "valid": valid_stats, "test": test_stats},
        "training_plan_estimate": {"effective_batch": 4, "steps_per_epoch": steps,
                                   "warmup_steps_at_5_percent": math.ceil(steps * 0.05)},
        "near_duplicates": near,
        "outputs": {"train_sha256": sha256(OUT / "train/train.jsonl"),
                    "valid_sha256": sha256(OUT / "valid/valid.jsonl"),
                    "test_sha256": sha256(OUT / "test/test.jsonl")},
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": ["关键词命中不等于医学错误", "未逐条完成临床事实认证",
                        "近重复为有限召回候选，尚需人工复核", "保留原文内容，仅转换为 ShareGPT 对话格式"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "records": counts["records"], "selected": len(selected),
                      "excluded_union": counts["excluded_union"], "steps": steps,
                      "warmup_steps": math.ceil(steps * 0.05), "near_duplicate_candidates": near},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
