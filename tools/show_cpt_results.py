"""无需绘图库：在终端展示真实进度、同验证集指标及评测历史。"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/pt-base-full-eval-0.5b-v2"
CPT = ROOT / "outputs/pt-full20k-0.5b-v2"


def read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def compare(base, cpt, base_identity, cpt_identity):
    if base_identity["eval_sha256"] != cpt_identity["eval_sha256"]:
        raise ValueError("验证块指纹不同，禁止生成可比结论")
    if base["eval_samples"] != cpt["eval_samples"]:
        raise ValueError("验证块数量不同")
    rows = []
    for key in ("eval_loss", "perplexity"):
        a, b = base[key], cpt[key]
        if not math.isfinite(a) or not math.isfinite(b) or a <= 0:
            raise ValueError("指标不是有效有限数值")
        rows.append({"metric": key, "base": a, "cpt": b, "delta": b-a, "reduction_percent": (a-b)/a*100})
    return rows


def render():
    manifest = read(ROOT / "data/processed/pretrain-clean-20k-v2/manifest.json")
    base, cpt = read(BASE / "eval_results.json"), read(CPT / "eval_results.json")
    progress = read(CPT / "progress.json") or read(BASE / "progress.json") or {}
    failed = read(CPT / "failure.json") or read(BASE / "failure.json")
    lines = ["=" * 78, " Qwen2.5-0.5B | BASE vs CPT | 20K-source / cleaned / LoRA", "=" * 78]
    if manifest:
        lines += [f" Train: {manifest['train']['documents']:,} docs | {manifest['train']['blocks']:,} blocks | {manifest['train']['input_tokens']:,} input tokens",
                  f" Eval : {manifest['valid']['documents']:,} docs | {manifest['valid']['blocks']:,} blocks | {manifest['valid']['input_tokens']:,} input tokens",
                  f" Setup: 1 epoch | block=256 | batch=2 x accum=4 | warmup={manifest['warmup_steps']} steps"]
    step, maximum = progress.get("step", 0), progress.get("max_steps", 0)
    lines.append(f" Status: {'FAILED' if failed else progress.get('status', 'not started')} | phase={progress.get('phase', '--')} | step={step}/{maximum}")
    if maximum > 0:
        lines.append(f" Progress: {min(step/maximum, 1)*100:.1f}%")
    if "learning_rate" in progress:
        lines.append(f" Latest logged LR: {progress['learning_rate']:.8f} | train_loss={progress.get('loss', float('nan')):.6f}")
    lines += ["-" * 78, f" {'Metric':<14}{'Baseline':>14}{'CPT final':>14}{'Delta':>14}{'Reduction':>14}", "-" * 78]
    rows = None
    if base and cpt:
        rows = compare(base, cpt, read(BASE / "dataset_identity.json"), read(CPT / "dataset_identity.json"))
        for row in rows:
            label = "PPL" if row["metric"] == "perplexity" else row["metric"]
            lines.append(f" {label:<14}{row['base']:>14.6f}{row['cpt']:>14.6f}{row['delta']:>+14.6f}{row['reduction_percent']:>13.2f}%")
    else:
        for key in ("eval_loss", "perplexity"):
            label = "PPL" if key == "perplexity" else key
            value = f"{base[key]:.6f}" if base else "pending"
            lines.append(f" {label:<14}{value:>14}{'pending':>14}{'--':>14}{'--':>14}")
    history = read(CPT / "eval_history.json") or []
    if history:
        lines += ["-" * 78, " Evaluation history (same full validation set)", f" {'Step':>9}{'Epoch':>12}{'eval_loss':>16}{'PPL':>16}"]
        seen = set()
        for row in history:
            if row["step"] in seen:
                continue
            seen.add(row["step"])
            lines.append(f" {row['step']:>9}{(row['epoch'] or 0):>12.3f}{row['eval_loss']:>16.6f}{row['perplexity']:>16.6f}")
    train = read(CPT / "train_results.json")
    if train:
        lines.append(f" Train runtime (incl. periodic eval): {train['train_runtime']/60:.2f} min")
    if failed:
        lines.append(f" ERROR: {failed['error']}")
    lines += ["-" * 78, " Lower eval_loss / PPL is better. Delta = CPT - Base.",
              " Positive reduction = improvement; negative = regression.",
              " PPL = exp(eval_loss); these are NOT two independent quality tests.",
              " Validation metrics do NOT establish medical correctness or safety.", "=" * 78]
    return "\n".join(lines), rows, progress.get("status") == "complete" and cpt is not None, bool(failed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="每 5 秒刷新，Ctrl+C 退出")
    parser.add_argument("--save", action="store_true", help="完整结果存在时保存截图用文本和 JSON")
    args = parser.parse_args()
    while True:
        output, rows, complete, failed = render()
        if args.watch:
            print("\033[2J\033[H", end="")
        print(output, flush=True)
        if args.save:
            if not complete or rows is None or failed:
                raise RuntimeError("训练尚未成功完成，不能保存最终对比")
            folder = ROOT / "reports/results"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "cpt-full20k-comparison.txt").write_text(output + "\n", encoding="utf-8")
            (folder / "cpt-full20k-comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        if not args.watch or complete or failed:
            break
        time.sleep(5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已退出展示；不会停止训练。")
