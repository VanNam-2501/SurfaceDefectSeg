"""Build the straw-only handover from explicit source/evidence allowlists."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from apps.stream_demo.serving_contract import file_sha256, load_contract
from apps.stream_demo.research_catalog import ResearchCatalog

ARTIFACT = "b1_stream_512_pose_v1"
SOURCE_EXTENSIONS = {".py", ".ps1", ".sh", ".js", ".cjs", ".mjs", ".html", ".css", ".cpp",
                     ".json", ".md", ".toml", ".txt", ".ipynb", ".svg", ".png", ".yaml", ".yml"}


def source_files():
    files = {}
    def add(path):
        relative = path.relative_to(ROOT).as_posix()
        if path.is_symlink() or any(part in {"__pycache__", "node_modules", ".pytest_cache"} for part in path.parts):
            return
        if path.suffix in SOURCE_EXTENSIONS or path.name in {"Dockerfile", "LICENSE", ".gitignore", ".dockerignore", ".gitattributes"}:
            files[relative] = path
    for directory in ("src/straw_global_local", "src/straw_defect_cls", "src/inspection_research",
                      "apps/stream_demo", "scripts/research", "scripts/deployment", "scripts/packaging", "docker/demo"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file():
                add(path)
    for pattern in ("configs/straw*.json", "requirements/straw*.txt", "requirements/b1*.txt",
                    "requirements/deepstream-demo.txt", "docs/STRAW*.md", "docs/*ONG_HUT*.md",
                    "docs/KAGGLE_ALL_IN_ONE.md", "docs/INSPECTION_MODEL_RESEARCH.md",
                    "docs/DEEPSTREAM_HANDOVER.md", "docs/assets/straw_overview/*",
                    "experiments/notebooks/straw*.ipynb"):
        for path in ROOT.glob(pattern):
            if path.is_file():
                add(path)
    for name in ("README.md", "START_HERE.md", "HANDOVER_STATUS.md", "pyproject.toml", ".gitignore",
                 "setup_demo.ps1", "verify_demo.ps1", "stop_demo.ps1", "rebuild_demo_engine.ps1", "package_project.ps1",
                 "run_demo.ps1", "run_b1_conveyor.ps1", "run_b1_stream.ps1", "run_b1_stream_upgrade.ps1", "run_b1_preview.ps1",
                 "run_b1_research.ps1", "run_b1_nvidia_gpu.ps1", "run_straw_thesis.ps1", "verify_straw.ps1",
                 "verify_inspection_math.ps1", "configs/project.yaml", "configs/thresholds.json",
                 "scripts/data/make_degraded_set.py", "scripts/data/make_b1_degraded_set.py",
                 "scripts/export_onnx.py", "STRAW_KAGGLE_ALL_EXPERIMENTS.ipynb",
                 "run_demo_docker.ps1", "run_demo_docker.sh", ".dockerignore", ".gitattributes",
                 "docs/SOURCE_SCOPE.md"):
        add(ROOT / name)
    add(ROOT / 'scripts/reporting/build_straw_dashboard.py')
    for path in (ROOT / "tests").glob("test_*"):
        if path.name not in {"test_web_api.py", "test_b1_deployment.py"}:
            add(path)
    return files


def handover_files():
    files = {}
    for entry in ResearchCatalog(ROOT).files().values():
        files[entry['path']] = ROOT / entry['path']
    for path in (ROOT / 'artifacts/research_dashboard').glob('index.*'):
        files[path.relative_to(ROOT).as_posix()] = path
    for path in (ROOT / 'demo_samples').glob('conveyor_replay.*'):
        files[path.relative_to(ROOT).as_posix()] = path
    wheels = list((ROOT / "runtime/wheels/demo").glob("*.whl"))
    if len(wheels) != 2:
        raise ValueError("Expected the pinned NumPy/Pillow Windows CPython 3.12 wheels in runtime/wheels/demo")
    for path in wheels:
        files[path.relative_to(ROOT).as_posix()] = path
    out = ROOT / "artifacts" / ARTIFACT
    for name in ("b1_rawrgb512.onnx", "b1_rtx3050_fp16.engine", "engine_cache.json", "manifest.json",
                 "nvinfer.txt", "quality.mp4", "load_30fps.mp4"):
        path = out / name
        files[path.relative_to(ROOT).as_posix()] = path
    for name in ('capture_probe', 'capture_probe.source.sha256'):
        path = out / name
        if not path.is_file():
            raise FileNotFoundError('Run a video once to build/verify the DeepStream capture probe before packaging')
        files[path.relative_to(ROOT).as_posix()] = path
    config = json.loads((ROOT / "configs/straw_realtime_stream512_pose_v1.json").read_text())
    checkpoint = ROOT / config["checkpoint"]
    if file_sha256(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("Refuse to package a checkpoint differing from the pinned demo model")
    files[config["checkpoint"]] = checkpoint
    # Frozen research records retain their original bytes and provenance. No training weights/caches.
    evidence = ROOT / "straw_resume/artifacts/straw_global_local/confirmatory_kaggle_v1"
    for path in evidence.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".csv", ".npz", ".md"}:
            files[path.relative_to(ROOT).as_posix()] = path
    for pattern in ("straw_resume/artifacts/straw_global_local/data_v1/*.json",
                    "straw_resume/artifacts/straw_global_local/data_v1/*.csv",
                    "artifacts/local_b1_check/*test.json", "artifacts/local_b1_check/REPORT.md",
                    "artifacts/b1_pose_illumination/*.json", "artifacts/b1_pose_illumination/*.csv",
                    "artifacts/b1_conveyor_audit/verified_v1/*.json"):
        for path in ROOT.glob(pattern):
            if path.is_file():
                files[path.relative_to(ROOT).as_posix()] = path
    # Runtime logs are mutable: deliver historical snapshots under evidence instead.
    for name in ("http_smoke.json", "legacy30_http.json", "file.jsonl.summary.json", "rtsp.jsonl.summary.json",
                 "sdk_version.txt", "gpu_identity.txt", "tensorrt_layers.json"):
        path = out / name
        if path.is_file():
            files[f"evidence/deepstream_original/{name}"] = path
    for path in (ROOT / "artifacts/verification/release").glob("*.log"):
        files[f"evidence/software_tests/{path.name}"] = path
    for version in ('upgrade_v2', 'upgrade_v3', 'upgrade_v3_2', 'upgrade_v3_3', 'upgrade_v3_4', 'upgrade_v3_5'):
        for path in (ROOT / 'artifacts/verification' / version).glob('*'):
            if path.is_file() and path.suffix in {'.json', '.log', '.png', '.jpg'} and not path.name.startswith('rtsp_replay'):
                files[f'evidence/{version}/{path.name}'] = path
    files["submission/01_report/Bao_cao_tong_quan_khoa_luan_phan_loai_khuyet_tat_ong_hut.docx"] = (
        ROOT / "submission/01_report/Bao_cao_tong_quan_khoa_luan_phan_loai_khuyet_tat_ong_hut.docx")
    files["submission/01_report/Bao_cao_tong_quan_khoa_luan_phan_loai_khuyet_tat_ong_hut.pdf"] = (
        ROOT / "submission/01_report/_report_work/render_v3/report.pdf")
    return files


def samples():
    rows = list(csv.DictReader((ROOT / "artifacts/straw_global_local/data_v1/val.csv").open(encoding="utf-8-sig")))
    files, cases = {}, []
    for class_name in ("good", "bent", "loose", "missing", "pierced"):
        row = next(row for row in rows if row["class_name"] == class_name)
        path = ROOT / "data/straw_defects_raw" / row["relative_path"]
        if file_sha256(path) != row["sha256"]:
            raise ValueError("Sample image differs from the frozen validation manifest")
        target = f"demo_samples/input/{class_name}_{row['sample_id']}{path.suffix.lower()}"
        files[target] = path
        cases.append({"sample_id": row["sample_id"], "class_name": class_name, "role": "val",
                      "path": target, "sha256": row["sha256"], "source_relative_path": row["relative_path"]})
    return files, {"scope": "Internal validation-photo demo samples; not a new evaluation set", "cases": cases}


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def create(output, kind, created):
    files = source_files()
    identity = load_contract(ROOT / "artifacts" / ARTIFACT) if kind == "handover" else None
    generated = {}
    if kind == "handover":
        files.update(handover_files())
        sample_files, sample_manifest = samples()
        files.update(sample_files)
        generated["demo_samples/manifest.json"] = json_bytes(sample_manifest)
    else:
        generated["START_HERE.md"] = ("# Gói mã nguồn ống hút\n\nGói source không chứa engine/checkpoint, "
            "ảnh mẫu, clip hoặc báo cáo nhị phân. Để chạy demo dùng gói straw_deepstream cùng ngày. "
            "Mã demo chính ở apps/stream_demo; hướng dẫn đầy đủ ở docs/DEEPSTREAM_HANDOVER.md.\n").encode("utf-8")
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    head = git_head.stdout.strip() if git_head.returncode == 0 else None
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    dirty = git_status.stdout.splitlines() if git_status.returncode == 0 else []
    generated["SOURCE_STATE.json"] = json_bytes({"base_commit": head, "working_tree_modified": bool(dirty),
        "working_tree_entry_count": len(dirty), "git_available": git_head.returncode == 0,
        "snapshot": "current working tree, including uncommitted sources",
        "created_at": created, "package_kind": kind, "artifact": ARTIFACT, "identity": identity})
    payload = {}
    for name, path in sorted(files.items()):
        if not path.is_file():
            raise FileNotFoundError(path)
        if any(part.lower() in {".git", ".aws", ".ssh", ".venv", ".venv-demo", "node_modules", "__pycache__"} for part in Path(name).parts):
            raise ValueError("Forbidden release path")
        if name in generated:
            continue
        if path.suffix == ".ipynb":
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for cell in notebook.get("cells", []):
                if cell.get("cell_type") == "code":
                    cell["outputs"], cell["execution_count"] = [], None
            generated[name] = json_bytes(notebook)
        else:
            payload[name] = path
    entries = []
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name in sorted(set(payload) | set(generated)):
                if name in generated:
                    data = generated[name]
                    archive.writestr(name, data)
                    size, digest = len(data), hashlib.sha256(data).hexdigest()
                else:
                    path = payload[name]
                    archive.write(path, name)
                    size, digest = path.stat().st_size, file_sha256(path)
                entries.append({"path": name, "bytes": size, "sha256": digest})
            manifest = {"schema": 1, "project": "straw-defect-inspection", "created_at": created,
                        "kind": kind, "files": entries}
            archive.writestr("release_manifest.json", json_bytes(manifest))
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP CRC verification failed")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    digest = file_sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="ascii")
    return {"file": output.name, "bytes": output.stat().st_size, "sha256": digest,
            "payload_files": len(entries), "kind": kind}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/releases/v3_5")
    parser.add_argument("--kind", choices=("all", "handover", "source"), default="all")
    args = parser.parse_args()
    now = datetime.now(timezone(timedelta(hours=7)))
    packages = []
    for kind in (("handover", "source") if args.kind == "all" else (args.kind,)):
        name = "straw_deepstream_v3_5" if kind == "handover" else "straw_source_v3_5"
        packages.append(create(args.output_dir / f"{name}_{now:%Y-%m-%d}.zip", kind, now.isoformat()))
    index = {"created_at": now.isoformat(), "packages": packages, "artifact": ARTIFACT}
    (args.output_dir / "release_index.json").write_bytes(json_bytes(index))
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
