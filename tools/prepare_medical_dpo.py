#!/usr/bin/env python
"""Convert MedicalGPT's official reward JSON files to TRL DPO JSONL.

The upstream files use question/response_chosen/response_rejected.  This
script preserves the official split, removes malformed or duplicated pairs,
and writes a manifest so the experiment is auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(source: Path, output: Path, seen: set[str], counters: dict[str, int]) -> int:
    text = source.read_text(encoding="utf-8")
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {source}")
    written = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            prompt = normalize(row.get("question"))
            chosen = normalize(row.get("response_chosen"))
            rejected = normalize(row.get("response_rejected"))
            if not prompt or not chosen or not rejected:
                counters["dropped_invalid"] += 1
                continue
            if chosen == rejected:
                counters["dropped_equal"] += 1
                continue
            key = hashlib.sha256((prompt + "\n" + chosen + "\n" + rejected).encode()).hexdigest()
            if key in seen:
                counters["dropped_duplicate"] += 1
                continue
            seen.add(key)
            handle.write(json.dumps({"prompt": prompt, "chosen": chosen, "rejected": rejected}, ensure_ascii=False) + "\n")
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    counters = {"dropped_invalid": 0, "dropped_equal": 0, "dropped_duplicate": 0}
    split_counts = {}
    source_hashes = {}
    for source_name, output_name in (("train.json", "train.jsonl"), ("valid.json", "validation.jsonl"), ("test.json", "test.jsonl")):
        source = args.input_dir / source_name
        if not source.exists():
            raise FileNotFoundError(source)
        source_hashes[source_name] = sha256(source)
        split_counts[output_name] = convert(source, args.output_dir / output_name, seen, counters)
    manifest = {
        "source": "shibing624/medical reward split",
        "source_dir": str(args.input_dir),
        "source_sha256": source_hashes,
        "format": "prompt/chosen/rejected",
        "split_counts": split_counts,
        "total_kept": sum(split_counts.values()),
        "filter_counts": counters,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
