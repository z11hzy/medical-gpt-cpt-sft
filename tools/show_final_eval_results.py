"""在终端打印正式 test loss/PPL 与 C-Eval Base/CPT 对比。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports/results"


def read(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def reduction(a, b):
    return (a - b) / a * 100


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n))


def render():
    tb, tc = read("test-lm-base.json"), read("test-lm-cpt-full20k.json")
    cb, cc = read("ceval-base-final.json"), read("ceval-cpt-full20k.json")
    if tb["packed_test_sha256"] != tc["packed_test_sha256"]:
        raise ValueError("Base/CPT 测试块指纹不同，禁止比较")
    if cb["total"] != cc["total"] or cb["scoring"] != cc["scoring"]:
        raise ValueError("C-Eval 题数或评分方法不同")
    old = {(x["subject"], x["id"]): x for x in cb["predictions"]}
    new = {(x["subject"], x["id"]): x for x in cc["predictions"]}
    wrong_to_right = sum(old[k]["prediction"] != old[k]["gold"] and new[k]["prediction"] == new[k]["gold"] for k in old)
    right_to_wrong = sum(old[k]["prediction"] == old[k]["gold"] and new[k]["prediction"] != new[k]["gold"] for k in old)
    changed = sum(old[k]["prediction"] != new[k]["prediction"] for k in old)
    lines = ["=" * 78, " Qwen2.5-0.5B | FINAL HELD-OUT TEST + C-EVAL | BASE vs CPT", "=" * 78,
             f" Test: {tb['documents']} docs | {tb['blocks']} blocks | {tb['input_tokens']:,} input tokens",
             "-" * 78, f" {'Test metric':<16}{'Baseline':>14}{'CPT final':>14}{'Delta':>14}{'Reduction':>14}", "-" * 78]
    for key, label in (("eval_loss", "test_loss"), ("perplexity", "test_PPL")):
        a, b = tb[key], tc[key]
        lines.append(f" {label:<16}{a:>14.6f}{b:>14.6f}{b-a:>+14.6f}{reduction(a,b):>13.2f}%")
    lines += ["-" * 78, " C-Eval public validation split | fixed zero-shot conditional likelihood",
              f" {'Subject':<22}{'Baseline':>18}{'CPT final':>18}{'Delta':>12}", "-" * 78]
    labels = {"basic_medicine": "Basic medicine", "clinical_medicine": "Clinical medicine", "physician": "Physician"}
    for subject, label in labels.items():
        a, b = cb["subjects"][subject], cc["subjects"][subject]
        lines.append(f" {label:<22}{f'{a['correct']}/{a['total']} ({a['accuracy']*100:.2f}%)':>18}{f'{b['correct']}/{b['total']} ({b['accuracy']*100:.2f}%)':>18}{(b['accuracy']-a['accuracy'])*100:>+11.2f}pp")
    lines += [f" {'TOTAL':<22}{f'{cb['correct']}/{cb['total']} ({cb['micro_accuracy']*100:.2f}%)':>18}{f'{cc['correct']}/{cc['total']} ({cc['micro_accuracy']*100:.2f}%)':>18}{(cc['micro_accuracy']-cb['micro_accuracy'])*100:>+11.2f}pp",
              "-" * 78, f" Prediction changes: {changed} | wrong->right: {wrong_to_right} | right->wrong: {right_to_wrong}",
              f" Exact McNemar p-value: {mcnemar_exact(wrong_to_right, right_to_wrong):.6f}",
              " Test loss/PPL measure text prediction; C-Eval measures 90 multiple-choice items.",
              " Neither evaluation establishes clinical safety.", "=" * 78]
    summary = {"test": {"base": tb, "cpt": tc}, "ceval": {"base": {k:v for k,v in cb.items() if k!="predictions"},
              "cpt": {k:v for k,v in cc.items() if k!="predictions"}, "prediction_changes": changed,
              "wrong_to_right": wrong_to_right, "right_to_wrong": right_to_wrong,
              "mcnemar_exact_p": mcnemar_exact(wrong_to_right, right_to_wrong)}}
    return "\n".join(lines), summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    text, summary = render()
    print(text)
    if args.save:
        (RESULTS / "final-evaluation-comparison.txt").write_text(text + "\n", encoding="utf-8")
        (RESULTS / "final-evaluation-comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
