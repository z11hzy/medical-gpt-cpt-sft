"""Validate MedicalGPT-compatible CPT, SFT or reward JSONL schemas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def valid_record(record: dict, schema: str) -> bool:
    if schema == "pt":
        return isinstance(record.get("text"), str) and bool(record["text"].strip())
    if schema == "sft":
        turns = record.get("conversations")
        return isinstance(turns, list) and len(turns) >= 2 and all(
            isinstance(turn, dict) and isinstance(turn.get("from"), str) and isinstance(turn.get("value"), str)
            for turn in turns
        )
    fields = ("question", "response_chosen", "response_rejected")
    return all(isinstance(record.get(field), str) and record[field].strip() for field in fields)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--schema", choices=("pt", "sft", "reward"), required=True)
    args = parser.parse_args()
    valid, invalid = 0, []
    for line_no, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            invalid.append(line_no)
            continue
        if isinstance(record, dict) and valid_record(record, args.schema):
            valid += 1
        else:
            invalid.append(line_no)
    print(f"valid={valid} invalid={len(invalid)}")
    if invalid:
        print(f"invalid line numbers (first 20): {invalid[:20]}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
