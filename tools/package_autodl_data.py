"""Build a deterministic, private transfer archive for the AutoDL experiments."""
from __future__ import annotations

import gzip
import hashlib
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "transfer/autodl-data-v2.tar.gz"
INCLUDE = (
    ROOT / "data/processed/pretrain-clean-20k-v2",
    ROOT / "data/processed/sft-clean-20k-v2",
    ROOT / "data/raw/shibing624-medical/pretrain/test_encyclopedia.json",
    ROOT / "data/eval/ceval/README.md",
    ROOT / "data/eval/ceval/basic_medicine/val-00000-of-00001.parquet",
    ROOT / "data/eval/ceval/clinical_medicine/val-00000-of-00001.parquet",
    ROOT / "data/eval/ceval/physician/val-00000-of-00001.parquet",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_to_include() -> list[Path]:
    files: list[Path] = []
    for item in INCLUDE:
        if not item.exists():
            raise FileNotFoundError(f"Required transfer input is missing: {item}")
        files.extend(sorted(path for path in item.rglob("*") if path.is_file()) if item.is_dir() else [item])
    return sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())


def main() -> None:
    files = files_to_include()
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with ARCHIVE.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in files:
                    relative = path.relative_to(ROOT).as_posix()
                    info = archive.gettarinfo(str(path), arcname=relative)
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    with path.open("rb") as source:
                        archive.addfile(info, source)

    archive_hash = sha256(ARCHIVE)
    member_manifest = [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                        "sha256": sha256(path)} for path in files]
    manifest = {"archive": ARCHIVE.name, "bytes": ARCHIVE.stat().st_size, "sha256": archive_hash,
                "members": member_manifest, "extract_at": "repository root"}
    manifest_path = ARCHIVE.with_suffix("").with_suffix(".manifest.json")
    checksum_path = ARCHIVE.with_suffix("").with_suffix(".sha256")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    checksum_path.write_bytes(f"{archive_hash}  transfer/{ARCHIVE.name}\n".encode("ascii"))
    archived_hashes = {}
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        actual_members = sorted(member.name for member in archive.getmembers() if member.isfile())
        for member in archive.getmembers():
            if member.isfile():
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"Cannot read archived member: {member.name}")
                archived_hashes[member.name] = hashlib.sha256(extracted.read()).hexdigest()
    expected_members = sorted(item["path"] for item in member_manifest)
    if actual_members != expected_members:
        raise RuntimeError("Archive member verification failed")
    for item in member_manifest:
        if archived_hashes[item["path"]] != item["sha256"]:
            raise RuntimeError(f"Archive content verification failed: {item['path']}")
    print(json.dumps({"archive": str(ARCHIVE), "bytes": ARCHIVE.stat().st_size,
                      "sha256": archive_hash, "files": len(files),
                      "extract": "tar -xzf transfer/autodl-data-v2.tar.gz"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
