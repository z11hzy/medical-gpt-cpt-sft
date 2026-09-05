"""Fail-fast environment and data verification for the AutoDL 7B run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import torch


EXPECTED_MEDICALGPT_COMMIT = "ccc05f4b46442ecdefcc95d53aeedc9834d09dd6"
REQUIRED_PACKAGES = ("accelerate", "datasets", "huggingface-hub", "peft", "transformers")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_data(directory: Path, kind: str) -> dict[str, object]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {kind} manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = ("train", "valid") if kind == "CPT" else ("train", "valid", "test")
    result: dict[str, object] = {"directory": str(directory.resolve()), "files": {}}
    for name in names:
        path = directory / name / f"{name}.jsonl"
        expected = manifest[f"{name}_sha256"] if kind == "CPT" else manifest["outputs"][f"{name}_sha256"]
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{kind} {name} SHA-256 mismatch: {actual} != {expected}")
        result["files"][name] = {"bytes": path.stat().st_size, "sha256": actual}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cpt-data", type=Path, required=True)
    parser.add_argument("--sft-data", type=Path, required=True)
    parser.add_argument("--medicalgpt", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--qlora", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    packages = {}
    for package in REQUIRED_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Missing package: {package}")
    if args.qlora:
        try:
            packages["bitsandbytes"] = importlib.metadata.version("bitsandbytes")
        except importlib.metadata.PackageNotFoundError:
            errors.append("QLoRA requested but bitsandbytes is not installed")

    if not (args.model / "config.json").exists():
        errors.append(f"Model config not found: {args.model / 'config.json'}")
    commit = None
    upstream_script = args.medicalgpt / "training/pretraining.py"
    if not upstream_script.exists():
        errors.append(f"MedicalGPT training script not found: {upstream_script}")
    else:
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(args.medicalgpt), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT
            ).strip()
            if commit != EXPECTED_MEDICALGPT_COMMIT:
                errors.append(f"MedicalGPT commit is {commit}, expected {EXPECTED_MEDICALGPT_COMMIT}")
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"Cannot verify MedicalGPT commit: {exc}")

    cuda = torch.cuda.is_available()
    if args.require_cuda and not cuda:
        errors.append("CUDA is not available")
    gpu = None
    if cuda:
        props = torch.cuda.get_device_properties(0)
        gpu = {"name": props.name, "memory_gib": round(props.total_memory / 2**30, 2),
               "bf16_supported": bool(torch.cuda.is_bf16_supported())}
        if not gpu["bf16_supported"]:
            errors.append("GPU/PyTorch does not report BF16 support")
        if gpu["memory_gib"] < 22:
            warnings.append("GPU memory is below 22 GiB; use QLoRA or a larger GPU")

    try:
        cpt = check_data(args.cpt_data, "CPT")
        sft = check_data(args.sft_data, "SFT")
    except (FileNotFoundError, KeyError, ValueError) as exc:
        errors.append(str(exc))
        cpt = sft = None

    report = {
        "status": "PASS" if not errors else "FAIL", "python": sys.version.split()[0],
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "gpu": gpu,
        "packages": packages, "medicalgpt_commit": commit, "model": str(args.model.resolve()),
        "cpt_data": cpt, "sft_data": sft, "qlora": args.qlora, "warnings": warnings, "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
