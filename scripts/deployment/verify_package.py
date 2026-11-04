"""Verify every payload file against the release manifest; no ML dependencies."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(root, target_engine=False):
    root = Path(root).resolve()
    manifest = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
    failures = []
    seen = set()
    for item in manifest["files"]:
        relative = PurePosixPath(item["path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in item["path"] or ":" in item["path"]:
            raise ValueError("Unsafe manifest path")
        path = root.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(root) or item["path"] in seen:
            raise ValueError("Escaping/duplicate manifest path")
        seen.add(item["path"])
        if target_engine and item["path"] in (
            "artifacts/b1_stream_512_pose_v1/b1_rtx3050_fp16.engine",
            "artifacts/b1_stream_512_pose_v1/engine_cache.json",
        ):
            continue  # check_demo.py validates the replacement engine contract and target hardware.
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            failures.append(item["path"])
    if failures:
        raise ValueError("Missing/changed release files: " + ", ".join(failures))
    return len(seen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-engine", action="store_true")
    args = parser.parse_args()
    print(f"Release payload: PASS ({verify(ROOT, args.target_engine)} files; target engine={args.target_engine})")
