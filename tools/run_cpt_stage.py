"""Run audited MedicalGPT CPT/baseline evaluation with portable parameters."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import HfArgumentParser, TrainerCallback


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/processed/pretrain-clean-20k-v2"
DEFAULT_MODEL = ROOT / "models/Qwen2.5-0.5B"
DEFAULT_OUTPUTS = {
    "base": ROOT / "outputs/pt-base-full-eval-0.5b-v2",
    "cpt": ROOT / "outputs/pt-full20k-0.5b-v2",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def dataset_fingerprint(dataset) -> str:
    digest = hashlib.sha256()
    for row in dataset:
        digest.update(json.dumps({key: row[key] for key in sorted(row)}, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


class Progress(TrainerCallback):
    def __init__(self, output: Path, phase: str) -> None:
        self.output, self.phase, self.started = output, phase, time.monotonic()
        self.latest: dict[str, object] = {}
        self.history: list[dict[str, object]] = []

    def update(self, state, logs=None, status="running") -> None:
        if logs:
            self.latest.update(logs)
        value = dict(
            self.latest,
            phase=self.phase,
            status=status,
            step=state.global_step,
            max_steps=state.max_steps,
            epoch=state.epoch,
            elapsed_seconds=round(time.monotonic() - self.started, 1),
            updated_at=datetime.now().isoformat(timespec="seconds"),
        )
        if "eval_loss" in value:
            value["perplexity"] = math.exp(float(value["eval_loss"]))
        write_json(self.output / "progress.json", value)

    def on_train_begin(self, args, state, control, **kwargs):
        self.update(state)

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.update(state, logs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        loss = float(metrics["eval_loss"])
        row = {"step": state.global_step, "epoch": state.epoch, "eval_loss": loss, "perplexity": math.exp(loss)}
        self.history.append(row)
        write_json(self.output / "eval_history.json", self.history)
        self.update(state, metrics)
        print(f"[CPT EVAL] step={state.global_step} eval_loss={loss:.6f} PPL={math.exp(loss):.6f}", flush=True)


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("base", "cpt"))
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--peer-identity", type=Path, help="Optional baseline dataset_identity.json to compare")
    parser.add_argument("--train-samples", type=int, help="Cap tokenized training blocks (smoke tests)")
    parser.add_argument("--eval-samples", type=int, help="Cap tokenized validation blocks (smoke tests)")
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qlora", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--check-only", action="store_true")
    options = parser.parse_args()
    for name in ("train_samples", "eval_samples", "train_batch_size", "eval_batch_size", "gradient_accumulation_steps"):
        value = getattr(options, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= options.warmup_ratio < 1:
        parser.error("--warmup-ratio must be in [0, 1)")
    return options


def main() -> None:
    options = parse_options()
    data = options.data_dir.resolve()
    model = options.model_path.resolve()
    output = (options.output_dir or DEFAULT_OUTPUTS[options.phase]).resolve()
    manifest_path = data / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing CPT manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for split in ("train", "valid"):
        path = data / split / f"{split}.jsonl"
        actual = sha256(path)
        expected = manifest[f"{split}_sha256"]
        if actual != expected:
            raise ValueError(f"{split} SHA-256 mismatch: {actual} != {expected}")

    total_train = int(manifest["train"]["blocks"])
    total_eval = int(manifest["valid"]["blocks"])
    train_samples = min(options.train_samples or total_train, total_train)
    eval_samples = min(options.eval_samples or total_eval, total_eval)
    effective_batch = options.train_batch_size * options.gradient_accumulation_steps
    expected_steps = math.ceil(train_samples / effective_batch) if options.phase == "cpt" else 0
    warmup_steps = math.ceil(expected_steps * options.warmup_ratio)

    argv = [
        "--model_name_or_path", str(model),
        "--train_file_dir", str(data / "train"),
        "--validation_file_dir", str(data / "valid"),
        "--output_dir", str(output),
        "--per_device_eval_batch_size", str(options.eval_batch_size),
        "--max_eval_samples", str(eval_samples),
        "--do_eval", "--use_peft", "True", "--seed", "42", "--block_size", "256", "--packing", "True",
        "--preprocessing_num_workers", "1", "--target_modules", "all", "--lora_rank", str(options.lora_rank),
        "--lora_alpha", str(options.lora_alpha), "--lora_dropout", str(options.lora_dropout),
        "--torch_dtype", "bfloat16", "--bf16", "--report_to", "none", "--disable_tqdm", "True",
    ]
    if options.qlora:
        argv += ["--load_in_4bit", "True", "--qlora", "True"]
    if options.phase == "cpt":
        argv += [
            "--do_train", "--max_train_samples", str(train_samples),
            "--per_device_train_batch_size", str(options.train_batch_size),
            "--gradient_accumulation_steps", str(options.gradient_accumulation_steps),
            "--num_train_epochs", "1", "--learning_rate", str(options.learning_rate),
            "--warmup_steps", str(warmup_steps), "--lr_scheduler_type", "linear", "--weight_decay", "0.01",
            "--logging_strategy", "steps", "--logging_steps", "100", "--logging_first_step", "True",
            "--eval_strategy", "steps", "--eval_steps", str(min(options.eval_steps, expected_steps)),
            "--save_strategy", "steps", "--save_steps", str(min(options.save_steps, expected_steps)),
            "--save_total_limit", "2", "--gradient_checkpointing", str(options.gradient_checkpointing),
        ]

    training_dir = ROOT / "vendor/MedicalGPT/training"
    if not (training_dir / "pretraining.py").exists():
        raise FileNotFoundError(f"Missing pinned MedicalGPT source: {training_dir}")
    sys.path.insert(0, str(training_dir))
    module = importlib.import_module("pretraining")
    model_args, data_args, training_args, script_args = HfArgumentParser(
        (module.ModelArguments, module.DataArguments, module.Seq2SeqTrainingArguments, module.ScriptArguments)
    ).parse_args_into_dataclasses(argv, look_for_args_file=False)
    if data_args.max_eval_samples != eval_samples:
        raise ValueError("MedicalGPT did not parse max_eval_samples as expected")
    if options.phase == "cpt":
        if data_args.max_train_samples != train_samples:
            raise ValueError("MedicalGPT did not parse max_train_samples as expected")
        if training_args.get_warmup_steps(expected_steps) != warmup_steps:
            raise ValueError("Warmup calculation differs from the audited plan")
    if bool(model_args.load_in_4bit) != options.qlora or bool(script_args.qlora) != options.qlora:
        raise ValueError("QLoRA flags were not parsed consistently")

    audit = {
        "phase": options.phase, "model": str(model), "data": str(data), "output": str(output),
        "train_blocks": train_samples if options.phase == "cpt" else 0, "validation_blocks": eval_samples,
        "per_device_train_batch_size": options.train_batch_size if options.phase == "cpt" else None,
        "gradient_accumulation_steps": options.gradient_accumulation_steps if options.phase == "cpt" else None,
        "effective_batch_single_gpu": effective_batch if options.phase == "cpt" else None,
        "expected_steps_single_gpu": expected_steps, "warmup_steps": warmup_steps,
        "gradient_checkpointing": options.gradient_checkpointing, "qlora": options.qlora,
        "data_manifest": manifest, "argv": argv, "effective_training_args": training_args.to_dict(),
    }
    if options.check_only:
        print(json.dumps(audit, ensure_ascii=False, indent=2, default=str))
        return
    if not model.exists():
        raise FileNotFoundError(f"Missing model: {model}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment directory: {output}")
    output.mkdir(parents=True)
    write_json(output / "run_config.json", audit)
    write_json(output / "progress.json", {"phase": options.phase, "status": "loading", "step": 0, "max_steps": expected_steps})

    progress = Progress(output, options.phase)
    original = module.SavePeftModelTrainer

    class CheckedTrainer(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if len(self.eval_dataset) != eval_samples:
                raise ValueError(f"Unexpected validation blocks: {len(self.eval_dataset)} != {eval_samples}")
            identity = {"eval_blocks": len(self.eval_dataset), "eval_sha256": dataset_fingerprint(self.eval_dataset)}
            if options.phase == "cpt":
                if len(self.train_dataset) != train_samples:
                    raise ValueError(f"Unexpected training blocks: {len(self.train_dataset)} != {train_samples}")
                identity.update(train_blocks=len(self.train_dataset), train_sha256=dataset_fingerprint(self.train_dataset))
                peer = options.peer_identity
                if peer is None and model == DEFAULT_MODEL.resolve() and data == DEFAULT_DATA.resolve():
                    peer = DEFAULT_OUTPUTS["base"] / "dataset_identity.json"
                if peer is not None and peer.exists():
                    base_identity = json.loads(peer.read_text(encoding="utf-8"))
                    if identity["eval_sha256"] != base_identity["eval_sha256"]:
                        raise ValueError("Base/CPT validation tokenization differs")
                    identity["matches_baseline"] = True
            else:
                b_params = [parameter for name, parameter in self.model.named_parameters() if "lora_B" in name]
                if not b_params or not all(torch.count_nonzero(parameter).item() == 0 for parameter in b_params):
                    raise ValueError("Baseline LoRA delta is not zero-initialized")
                identity["baseline_zero_lora_delta_verified"] = True
            write_json(output / "dataset_identity.json", identity)
            self.add_callback(progress)

    module.SavePeftModelTrainer = CheckedTrainer
    module.logger.remove()
    module.logger.add(sys.stderr, level="INFO")
    sys.argv = [str(training_dir / "pretraining.py"), *argv]
    try:
        module.main()
        result = json.loads((output / "eval_results.json").read_text(encoding="utf-8"))
        if not math.isfinite(float(result["eval_loss"])) or not math.isfinite(float(result["perplexity"])):
            raise ValueError(f"Invalid evaluation result: {result}")
        if result["eval_samples"] != eval_samples:
            raise ValueError(f"Unexpected eval sample count: {result['eval_samples']}")
        if options.phase == "cpt":
            state = json.loads((output / "trainer_state.json").read_text(encoding="utf-8"))
            if state["global_step"] != expected_steps or abs(float(state["epoch"]) - 1.0) > 1e-8:
                raise ValueError(f"Training did not complete exactly one epoch: {state['global_step']}, {state['epoch']}")
        value = json.loads((output / "progress.json").read_text(encoding="utf-8"))
        value.update(status="complete", eval_loss=result["eval_loss"], perplexity=result["perplexity"])
        write_json(output / "progress.json", value)
    except BaseException as exc:
        write_json(output / "failure.json", {"error": repr(exc), "time": datetime.now().isoformat()})
        raise


if __name__ == "__main__":
    main()
