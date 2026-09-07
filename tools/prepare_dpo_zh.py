"""Build a reproducible, length-bounded Chinese DPO split for Qwen chat models.

The source data stays untouched.  This script emits prompt/chosen/rejected JSONL
accepted by TRL's DPOTrainer and a manifest with all filtering decisions.
"""
import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


def norm(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).casefold()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def messages_for(row: dict) -> list[dict]:
    messages = []
    system = row.get("system", "").strip()
    if system:
        messages.append({"role": "system", "content": system})
    for turn in row.get("history", []):
        messages.extend([
            {"role": "user", "content": turn[0]},
            {"role": "assistant", "content": turn[1]},
        ])
    messages.append({"role": "user", "content": row["question"].strip()})
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-prompt-tokens", type=int, default=384)
    parser.add_argument("--max-completion-tokens", type=int, default=384)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    if args.validation_size < 1:
        raise ValueError("validation-size must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    eos = tokenizer.eos_token or ""
    counts, seen = Counter(), set()
    kept = []
    with args.input.open("r", encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            counts["input_rows"] += 1
            try:
                row = json.loads(line)
                required = [row[key] for key in ("question", "response_chosen", "response_rejected")]
                if not all(isinstance(item, str) and item.strip() for item in required):
                    raise ValueError("missing required text")
                if not isinstance(row.get("system", ""), str) or not isinstance(row.get("history", []), list):
                    raise ValueError("invalid context")
                if any(not isinstance(turn, list) or len(turn) != 2 or not all(isinstance(x, str) for x in turn)
                       for turn in row.get("history", [])):
                    raise ValueError("invalid history")
            except (json.JSONDecodeError, KeyError, ValueError):
                counts["dropped_invalid"] += 1
                continue
            if "\ufffd" in line:
                counts["dropped_replacement_char"] += 1
                continue
            chosen, rejected = row["response_chosen"].strip(), row["response_rejected"].strip()
            if norm(chosen) == norm(rejected):
                counts["dropped_equal_preference"] += 1
                continue
            prompt_key = json.dumps([norm(row.get("system", "")), row.get("history", []), norm(row["question"])],
                                    ensure_ascii=False, sort_keys=True)
            if prompt_key in seen:
                counts["dropped_duplicate_prompt"] += 1
                continue
            seen.add(prompt_key)
            prompt = tokenizer.apply_chat_template(messages_for(row), tokenize=False, add_generation_prompt=True)
            # Keep completions as raw assistant text; TRL concatenates them to prompt.
            chosen_ids = tokenizer(chosen + eos, add_special_tokens=False)["input_ids"]
            rejected_ids = tokenizer(rejected + eos, add_special_tokens=False)["input_ids"]
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if (len(prompt_ids) > args.max_prompt_tokens or
                    len(chosen_ids) > args.max_completion_tokens or
                    len(rejected_ids) > args.max_completion_tokens):
                counts["dropped_over_length"] += 1
                continue
            kept.append({"prompt": prompt, "chosen": chosen + eos, "rejected": rejected + eos,
                         "source_line": line_no})
            counts["kept"] += 1
    if len(kept) <= args.validation_size:
        raise RuntimeError(f"Only {len(kept)} rows retained; cannot reserve {args.validation_size} validation rows")
    random.Random(args.seed).shuffle(kept)
    validation, train = kept[:args.validation_size], kept[args.validation_size:]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", validation)):
        target = args.output_dir / f"{name}.jsonl"
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "source": {"path": str(args.input), "sha256": sha256(args.input)},
        "tokenizer": args.tokenizer,
        "seed": args.seed,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_completion_tokens": args.max_completion_tokens,
        "counts": dict(counts),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "outputs": {name: {"sha256": sha256(args.output_dir / f"{name}.jsonl")}
                    for name in ("train", "validation")},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
