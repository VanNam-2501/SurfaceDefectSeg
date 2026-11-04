"""Source-linked research catalog; frozen artifacts are never rewritten."""
from __future__ import annotations
import csv
import hashlib
import json
import mimetypes
from pathlib import Path
import statistics

GROUPS = {
    "global_local": ("B0–B6 · 21 lượt nghiên cứu", "Test nội bộ 299 ảnh; ba seed dùng chung tập Test"),
    "deployment": ("Tối ưu triển khai B1", "Các nhánh checkpoint/precision và Validation; đọc cấu hình của từng phép đo"),
    "pose": ("Pose / illumination", "Ảnh xoay và giảm sáng tổng hợp; chưa phải camera độc lập"),
    "stream": ("DeepStream / TensorRT", "Replay, parity và kiểm tra luồng; phân biệt từng engine"),
    "errors": ("Phân tích ảnh sai", "Phân tích chẩn đoán; không dùng như kiểm định mới"),
    "legacy": ("Pipeline ống hút trước đây", "Bảng Full5/cascade lịch sử; không gộp với B0–B6"),
    "reports": ("Báo cáo và hồ sơ", "Bản hiện có; chưa thay xác nhận rà dữ liệu hoặc duyệt của trường"),
    "verification": ("Kiểm chứng phần mềm", "Hồ sơ theo từng phiên bản: video/RTSP, preview và kiểm thử chức năng; không phải accuracy camera thật"),
}
ALLOWED = {".json", ".jsonl", ".csv", ".md", ".png", ".jpg", ".svg", ".pdf", ".docx", ".txt", ".npz"}


class ResearchCatalog:
    def __init__(self, root):
        self.root = Path(root)

    def files(self):
        files = {}
        def add(path, group):
            if path.is_file() and not path.is_symlink() and (path.suffix.lower() in ALLOWED or
                    (group == 'verification' and path.suffix.lower() == '.log')):
                relative = path.relative_to(self.root).as_posix()
                key = hashlib.sha256(relative.encode()).hexdigest()[:24]
                files[key] = {"id": key, "path": relative, "group": group, "name": path.name,
                              "bytes": path.stat().st_size, "url": f"/api/research/files/{key}"}
        for directory, group in (("straw_resume/artifacts/straw_global_local/confirmatory_kaggle_v1", "global_local"),
                                  ("straw_resume/artifacts/straw_global_local/data_v1", "global_local")):
            for path in (self.root / directory).rglob("*"):
                add(path, group)
        # Include report/metric files, but exclude probability caches and package repair tooling.
        for directory, group in (("artifacts/b1_deployment", "deployment"), ("artifacts/b1_nvidia_gpu", "deployment"),
                                  ("artifacts/b1_pose_illumination", "pose"),
                                  ("artifacts/local_b1_check", "deployment")):
            base = self.root / directory
            for path in base.glob("*"):
                if not path.name.endswith((".log", ".pid")):
                    add(path, group)
            for sub in base.iterdir() if base.exists() else []:
                if sub.is_dir() and not sub.name.startswith(("input_cache", "models", "tmp", "onnx", "tensorrt")):
                    for name in ("history.json", "complete.json", "test.json", "benchmark.json", "REPORT.md"):
                        add(sub / name, group)
        for directory in ("artifacts/b1_stream", "artifacts/b1_stream_512", "artifacts/b1_stream_512_pose_v1"):
            for name in ("REPORT.md", "report.json", "manifest.json", "engine_cache.json", "http_smoke.json",
                         "legacy30_http.json", "resident_probe.json", "file.jsonl.summary.json", "rtsp.jsonl.summary.json",
                         "sdk_version.txt", "gpu_identity.txt", "tensorrt_layers.json"):
                add(self.root / directory / name, "stream")
        for path in (self.root / "artifacts/b1_conveyor_audit/verified_v1").glob("*"):
            add(path, "stream")
        for path in (self.root / "straw_resume/error_analysis").rglob("*"):
            if "__pycache__" not in path.parts and path.suffix != ".npz":
                add(path, "errors")
        for directory in ("submission/tables", "submission/figures"):
            for path in (self.root / directory).glob("*"):
                add(path, "legacy")
        for pattern in ("docs/*ONG_HUT*.md", "docs/STRAW*.md", "docs/assets/straw_overview/*",
                        "submission/01_report/*ong_hut*.docx", "submission/01_report/*ong_hut*.pdf"):
            for path in self.root.glob(pattern):
                add(path, "reports")
        for directory in ('upgrade_v2', 'upgrade_v3', 'upgrade_v3_2', 'upgrade_v3_3', 'upgrade_v3_4', 'upgrade_v3_5'):
            for path in (self.root / 'artifacts/verification' / directory).glob('*'):
                if path.suffix in {'.json', '.log', '.jpg', '.png'} and not path.name.startswith('rtsp_replay'):
                    add(path, 'verification')
        return files

    def path(self, file_id):
        entry = self.files().get(file_id)
        if not entry:
            raise FileNotFoundError("Research artifact not found")
        path = (self.root / entry["path"]).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Research artifact escaped project")
        return path

    def snapshot(self, static=False):
        base = self.root / "straw_resume/artifacts/straw_global_local/confirmatory_kaggle_v1"
        def read(name):
            path = base / name
            return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        def csv_rows(name):
            path = base / name
            if not path.is_file():
                return []
            with path.open(encoding="utf-8-sig") as stream:
                return list(csv.DictReader(stream))
        runs = csv_rows("report/test_runs.csv")
        benchmarks = {r["variant"]: r for r in csv_rows("report/benchmarks.csv")}
        summary = []
        for variant in sorted({r["variant"] for r in runs}):
            rows = [r for r in runs if r["variant"] == variant]
            metrics = {}
            for key in ("macro_f1", "accuracy", "fnr", "fpr"):
                values = [float(r[key]) for r in rows]
                metrics[key] = {"mean": statistics.mean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0}
            summary.append({"variant": variant, "seeds": [int(r["seed"]) for r in rows], "metrics": metrics,
                            "benchmark": benchmarks.get(variant, {})})
        histories = {}
        for path in (base / "runs").glob("*/history.json"):
            histories[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
        freeze = read("freeze.json")
        review = freeze.get("review", {})
        reviewer, notes = str(review.get("reviewer", "")), str(review.get("notes", ""))
        human_review_verified = bool(reviewer and notes and "Tên người" not in reviewer and "Ghi ngắn" not in notes)
        files = list(self.files().values())
        if static:
            for entry in files:
                entry["url"] = "../../" + entry["path"]
        pose = self.root / "artifacts/b1_pose_illumination/test_comparison.json"
        return {"groups": [{"id": key, "title": value[0], "scope": value[1]} for key, value in GROUPS.items()],
                "files": files, "summary": summary, "runs": runs, "histories": histories,
                "readiness": read("report/readiness.json"), "experiment": read("experiment.json"),
                "selection": freeze.get('selection', {}),
                "human_review_verified": human_review_verified,
                "review_note": "Biên bản reviewer/notes còn là mẫu; trạng thái tự động không thay xác nhận thực." if not human_review_verified else "Có ghi nhận người rà; cần đối chiếu hồ sơ gốc.",
                "pose": json.loads(pose.read_text(encoding="utf-8")) if pose.is_file() else {},
                "scope": "Research checkpoints and deployment engines have separate results; no new factory accuracy claim"}
