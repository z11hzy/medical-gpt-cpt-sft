"""Print a compact, screenshot-friendly summary of the SFT experiment."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def pct_change(new: float, old: float) -> float:
    return (new / old - 1.0) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()

    valid_base = read("outputs/sft-base-20k-v2/eval_results.json")
    valid_cpt = read("outputs/sft-cpt-20k-v2/eval_results.json")
    test_base = read("outputs/eval-sft-base-test-v2/eval_results.json")
    test_cpt = read("outputs/eval-sft-cpt-test-v2/eval_results.json")
    ceval_base = read("reports/results/ceval-sft-base-final.json")
    ceval_cpt = read("reports/results/ceval-sft-cpt-final.json")
    train_base = read("outputs/sft-base-20k-v2/train_results.json")
    train_cpt = read("outputs/sft-cpt-20k-v2/train_results.json")
    ident_base = read("outputs/sft-base-20k-v2/dataset_identity.json")
    ident_cpt = read("outputs/sft-cpt-20k-v2/dataset_identity.json")

    paired = list(zip(ceval_base["predictions"], ceval_cpt["predictions"]))
    base_right = sum(a["prediction"] == a["gold"] and b["prediction"] != b["gold"] for a, b in paired)
    cpt_right = sum(a["prediction"] != a["gold"] and b["prediction"] == b["gold"] for a, b in paired)
    changed = sum(a["prediction"] != b["prediction"] for a, b in paired)
    discordant = base_right + cpt_right
    tail = sum(math.comb(discordant, k) for k in range(min(base_right, cpt_right) + 1))
    p_value = min(1.0, 2.0 * tail / (2 ** discordant)) if discordant else 1.0

    lines = [
        "=" * 78,
        "Qwen2.5-0.5B 医疗 SFT 正式对照实验（20k，1 epoch，LoRA）",
        "=" * 78,
        "分支             验证Loss   验证PPL   测试Loss   测试PPL   C-Eval",
        f"Base -> SFT      {valid_base['eval_loss']:8.4f}  {valid_base['perplexity']:8.4f}  "
        f"{test_base['eval_loss']:8.4f}  {test_base['perplexity']:8.4f}   "
        f"{ceval_base['correct']:2d}/{ceval_base['total']} ({ceval_base['micro_accuracy']*100:5.2f}%)",
        f"CPT -> SFT       {valid_cpt['eval_loss']:8.4f}  {valid_cpt['perplexity']:8.4f}  "
        f"{test_cpt['eval_loss']:8.4f}  {test_cpt['perplexity']:8.4f}   "
        f"{ceval_cpt['correct']:2d}/{ceval_cpt['total']} ({ceval_cpt['micro_accuracy']*100:5.2f}%)",
        "-" * 78,
        "CPT 前置相对 Base->SFT：",
        f"  验证 Loss {pct_change(valid_cpt['eval_loss'], valid_base['eval_loss']):+.3f}% | "
        f"验证 PPL {pct_change(valid_cpt['perplexity'], valid_base['perplexity']):+.3f}%",
        f"  测试 Loss {pct_change(test_cpt['eval_loss'], test_base['eval_loss']):+.3f}% | "
        f"测试 PPL {pct_change(test_cpt['perplexity'], test_base['perplexity']):+.3f}%",
        f"  C-Eval {ceval_cpt['correct']-ceval_base['correct']:+d} 题 | "
        f"预测变化 {changed} 题 | McNemar exact p={p_value:.4f}",
        "-" * 78,
        f"训练耗时：Base {train_base['train_runtime']/60:.2f} min | CPT {train_cpt['train_runtime']/60:.2f} min",
        f"数据一致性：{ident_cpt.get('matches_base_branch') is True} | "
        f"train={ident_base['train_samples']} | valid={ident_base['validation_samples']} | test=500",
        "结论：CPT 小幅改善回答 token 的验证/测试 Loss 与 PPL；C-Eval 未提升且差异不显著。",
        "限制：C-Eval 使用公开 validation 90 题；这些结果不能证明临床安全性。",
        "=" * 78,
    ]
    output = "\n".join(lines) + "\n"
    print(output, end="")
    if args.save:
        destination = args.save if args.save.is_absolute() else ROOT / args.save
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
