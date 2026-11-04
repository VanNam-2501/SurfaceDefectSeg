"""Loopback viewer and local batch-image launcher for the TensorRT DeepStream demo."""
from __future__ import annotations

import argparse
import ctypes
import ipaddress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import platform
from pathlib import Path
import re
import signal
import threading
import time
import uuid
from urllib.parse import parse_qs, urlsplit
from PIL import UnidentifiedImageError
try:
    from .resident_inference import ResidentInference
    from .serving_contract import ARTIFACTS, DEFAULT_ARTIFACT, CLASSES
    from .workspace_api import WorkspaceAPI
    from .runtime_backend import native_runtime
except ImportError:
    from resident_inference import ResidentInference
    from serving_contract import ARTIFACTS, DEFAULT_ARTIFACT, CLASSES
    from workspace_api import WorkspaceAPI
    from runtime_backend import native_runtime

ROOT = Path(__file__).resolve().parents[2]
HTML = Path(__file__).with_name("preview.html")
MAX_IMAGE_BYTES = 30 * 1024 * 1024
MAX_BATCH_BYTES = 2 * 1024 * 1024 * 1024
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def host_hardware() -> dict:
    cpu_name = platform.processor() or platform.machine()
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except (OSError, ImportError):
            pass

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                        ("total_page", ctypes.c_ulonglong), ("available_page", ctypes.c_ulonglong),
                        ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                        ("available_extended", ctypes.c_ulonglong)]

        memory = MemoryStatus()
        memory.length = ctypes.sizeof(memory)
        try:
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
                raise OSError("Cannot read Windows memory status")
            total_gib = round(memory.total / (1024 ** 3), 1)
            available_gib = round(memory.available / (1024 ** 3), 1)
        except (AttributeError, OSError):
            total_gib = available_gib = None
    else:
        total_gib = available_gib = None
        try:
            values = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, value = line.split(":", 1)
                values[key] = int(value.strip().split()[0]) * 1024
            total_gib = round(values["MemTotal"] / (1024 ** 3), 1)
            available_gib = round(values["MemAvailable"] / (1024 ** 3), 1)
        except (OSError, KeyError, ValueError):
            pass
    return {"cpu_name": cpu_name, "cpu_threads": os.cpu_count(),
            "ram_total_gib": total_gib, "ram_available_gib": available_gib}


def make_handler(preview_dir: Path, wsl_distro: str, wsl_root: str, artifact: str = DEFAULT_ARTIFACT):
    out = ROOT / "artifacts" / artifact
    jobs_dir = out / "image_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    jobs: dict[str, dict] = {}
    jobs_lock = threading.Lock()
    resident = ResidentInference(ROOT, wsl_distro, wsl_root, artifact=artifact)
    workspace = WorkspaceAPI(ROOT, resident, wsl_distro, wsl_root, artifact)

    def latest_state() -> dict:
        candidates = list(preview_dir.glob("frame_*.json"))
        if not candidates:
            return {"ready": False, "prediction_available": False, "running": (preview_dir / "session_running").is_file()}
        path = max(candidates, key=lambda item: item.name)
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            state["running"] = (preview_dir / "session_running").is_file()
            return state
        except (OSError, json.JSONDecodeError):
            return {"ready": False, "prediction_available": False, "running": (preview_dir / "session_running").is_file()}

    def read_job(job_id: str) -> dict | None:
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                return None
            snapshot = dict(job)
            process = job.get("process")
            expected = len(job["files"])
        inference_path = Path(snapshot["path"]) / "inference.jsonl"
        predictions = []
        if inference_path.exists():
            try:
                for line in inference_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    try:
                        record = json.loads(line)
                        probabilities = record.get("probabilities", [])
                        if len(probabilities) != len(CLASSES):
                            continue
                        class_index = record.get("class_index")
                        if not isinstance(class_index, int) or class_index not in range(len(CLASSES)):
                            class_index = max(range(len(CLASSES)), key=lambda i: probabilities[i])
                        predictions.append({
                            "index": len(predictions) + 1,
                            "frame_num": record.get("frame_num"),
                            "class": CLASSES[class_index],
                            "defective": class_index != 0,
                            "confidence": probabilities[class_index],
                            "probabilities": dict(zip(CLASSES, probabilities)),
                        })
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
            except OSError:
                pass
        code = process.poll() if process is not None else None
        if process is not None and code is None:
            snapshot["status"] = "running" if (preview_dir / "session_running").is_file() else "starting"
            snapshot["message"] = f"TensorRT đang phân loại ảnh ({len(predictions)}/{expected})…"
        elif process is not None and code == 0:
            snapshot["status"] = "completed"
            snapshot["message"] = f"Đã phân loại xong {len(predictions)}/{expected} ảnh bằng TensorRT."
            if len(predictions) != expected:
                snapshot["message"] += " Số kết quả không khớp số ảnh; kiểm tra log lượt chạy."
        elif process is not None:
            snapshot["status"] = "error"
            log_path = Path(snapshot["path"]) / "run.log"
            try:
                detail = log_path.read_text(encoding="utf-8", errors="replace")[-700:]
            except OSError:
                detail = ""
            snapshot["message"] = f"DeepStream kết thúc với mã {code}. {detail}"
        snapshot["predictions"] = predictions
        snapshot["expected"] = expected
        snapshot.pop("received", None)
        snapshot.pop("process", None)
        snapshot.pop("path", None)
        snapshot.pop("files", None)
        return snapshot

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        workspace_api = workspace

        def log_message(self, fmt: str, *args: object) -> None:
            pass

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, value: dict) -> None:
            self._send(status, "application/json; charset=utf-8", json.dumps(value, ensure_ascii=False).encode("utf-8"))

        def _local_only(self) -> bool:
            try:
                origin = self.headers.get("Origin")
                address = ipaddress.ip_address(self.client_address[0])
                # The container publishes only host loopback; NAT arrives from
                # its private bridge gateway. Keep the same-origin check.
                local = address.is_loopback or (native_runtime() and address.is_private)
                return local and (
                    not origin or urlsplit(origin).netloc == self.headers.get("Host"))
            except ValueError:
                return False

        def do_GET(self) -> None:
            request = urlsplit(self.path)
            route = request.path
            try:
                if workspace.get(self, route, parse_qs(request.query)):
                    return
            except (ValueError, KeyError, OSError) as exc:
                self._json(404 if isinstance(exc, FileNotFoundError) else 400, {"error": str(exc)})
                return
            if route == "/conveyor":
                self._send(200, "text/html; charset=utf-8", HTML.read_bytes())
            elif route in ("/conveyor_gate.js", "/conveyor_app.js", "/conveyor_gate_worker.js"):
                self._send(200, "text/javascript; charset=utf-8", HTML.with_name(route[1:]).read_bytes())
            elif route == "/api/runtime":
                self._json(200, dict(resident.status(), host_hardware=host_hardware(),
                                     gpu_hardware=resident.hardware_status(), process_hardware=workspace.metrics.snapshot()))
            elif route.startswith("/api/decisions/"):
                request_id = route.removeprefix("/api/decisions/")
                if not re.fullmatch(r"[0-9a-f]{32}", request_id):
                    self._json(404, {"error": "Unknown request"})
                else:
                    result = resident.result(request_id)
                    row = workspace.service.store.by_request(request_id)
                    if row:
                        result["inspection_id"] = row["id"]
                        result["quality"] = row["quality"]
                    self._json(200, result)
            elif route == "/health":
                state = latest_state()
                image = preview_dir / str(state.get("frame_file", ""))
                available = bool(state.get("ready")) and image.is_file()
                runtime = resident.status()
                with workspace.video.lock:
                    available = available or workspace.video.jpeg is not None
                    video_running = workspace.video.state['status'] in ('starting', 'calibrating', 'running')
                self._json(200, {"service": "TensorRT DeepStream preview", "ready": runtime['state'] == 'ready',
                    "preview_available": available, "running": state["running"] or video_running,
                    "image_inference": bool(wsl_root) or native_runtime(), "api_version": 15,
                    "project_root": str(ROOT),
                    "artifact": artifact, "runtime_verified": runtime["state"] == "ready",
                    **resident.identity, "runtime": runtime})
            elif route == "/state.json":
                self._json(200, latest_state())
            elif route == "/frame.jpg":
                state = latest_state()
                requested = parse_qs(request.query).get("frame", [""])[0]
                frame_name = requested or state.get("frame_file", "")
                path = preview_dir / frame_name if isinstance(frame_name, str) and Path(frame_name).name == frame_name else preview_dir / "missing.jpg"
                try:
                    self._send(200, "image/jpeg", path.read_bytes())
                except FileNotFoundError:
                    self._send(404, "text/plain; charset=utf-8", b"No preview frame yet")
            elif route.startswith("/api/jobs/"):
                job_id = route.removeprefix("/api/jobs/").split("/", 1)[0]
                if not re.fullmatch(r"[0-9a-f]{32}", job_id):
                    self._json(404, {"error": "Không tìm thấy lượt chạy."})
                    return
                result = read_job(job_id)
                self._json(200 if result else 404, result or {"error": "Không tìm thấy lượt chạy."})
            elif route == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Connection", "close")
                self.end_headers()
                last_signature: tuple[str, int, int] | None = None
                try:
                    while True:
                        state = latest_state()
                        frame_name = state.get("frame_file", "")
                        path = preview_dir / frame_name if isinstance(frame_name, str) and Path(frame_name).name == frame_name else preview_dir / "missing.jpg"
                        try:
                            stat = path.stat()
                            signature = (path.name, stat.st_mtime_ns, stat.st_size)
                            if signature != last_signature:
                                jpeg = path.read_bytes()
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                                self.wfile.flush()
                                last_signature = signature
                        except FileNotFoundError:
                            pass
                        time.sleep(0.04)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    return
            else:
                self._send(404, "text/plain; charset=utf-8", b"Not found")

        def do_POST(self) -> None:
            if not self._local_only():
                self._json(403, {"error": "Chỉ chấp nhận yêu cầu từ máy local."})
                return
            route = urlsplit(self.path).path
            try:
                if workspace.post(self, route):
                    return
            except (ValueError, KeyError, TypeError, OSError, UnidentifiedImageError) as exc:
                self._json(400, {"error": str(exc)})
                return
            if route == "/api/trigger":
                self._trigger()
            elif route == "/api/jobs":
                self._create_job()
            elif route.startswith("/api/jobs/"):
                parts = route.strip("/").split("/")
                if len(parts) == 5 and parts[:2] == ["api", "jobs"] and parts[3] == "images" and parts[4].isdigit():
                    self._receive_image(parts[2], int(parts[4]))
                elif len(parts) == 3 and parts[:2] == ["api", "jobs"] and parts[2].endswith("-run"):
                    self._start_job(parts[2][:-4])
                else:
                    self._json(404, {"error": "Không tìm thấy API."})
            else:
                self._json(404, {"error": "Không tìm thấy API."})

        def _content_length(self, maximum: int) -> int:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return 0
            return length if 0 < length <= maximum else 0

        def _trigger(self):
            length = self._content_length(16 * 1024)
            if not length:
                self._json(400, {"error": "Invalid gate event"})
                return
            try:
                payload = json.loads(self.rfile.read(length))
                job_id, index, gate = payload["job_id"], int(payload["index"]), payload["gate"]
                if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id):
                    raise ValueError("Invalid job")
                if (not gate.get("crossed") or int(gate.get("confirmation_window", 0)) != 5
                        or int(gate.get("confirmed_frames", 0)) < 4
                        or float(gate.get("roi_overlap", 0)) < .8):
                    raise ValueError("Gate chưa xác nhận đủ 4/5 frame hoặc mask chưa đạt 80% ROI.")
                with jobs_lock:
                    job = jobs.get(job_id)
                    if job is None or index not in job["received"]:
                        raise ValueError("Ảnh chưa tải xong hoặc lượt chọn đã hết hạn; chọn ảnh lại.")
                    path = Path(job["path"]) / job["files"][index - 1]["stored"]
                row = workspace.service.submit(path, f"{job_id}:{index}", job_id, payload.get("track_id"),
                    {"job_id": job_id, "image_index": index, "track_id": payload.get("track_id"),
                     "rotation_degrees": job["rotation_degrees"], "gate": gate, "source_kind": "conveyor"})
                self._json(202, {"request_id": row["request_id"], "inspection_id": row["id"],
                    "status": row["state"], "decision": row["decision"], "quality": row["quality"], "error": row["error"]})
            except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
                self._json(409, {"error": str(exc)})

        def _create_job(self) -> None:
            length = self._content_length(8 * 1024 * 1024)
            if not length:
                self._json(413, {"error": "Danh sách ảnh không hợp lệ."})
                return
            try:
                payload = json.loads(self.rfile.read(length))
                files = payload["files"]
                rotation = payload.get("rotation_degrees", 0)
                if type(rotation) is not int or rotation not in (0, 90, 180, 270):
                    raise ValueError("Chọn hướng xoay 0°, 90°, 180° hoặc 270°.")
                if not isinstance(files, list) or not files:
                    raise ValueError("Chọn ít nhất một ảnh.")
                validated = []
                total = 0
                for item in files:
                    suffix = Path(str(item.get("name", ""))).suffix.lower()
                    size = int(item.get("size", 0))
                    if suffix not in IMAGE_SUFFIXES or size <= 0 or size > MAX_IMAGE_BYTES:
                        raise ValueError("Ảnh phải là JPG, PNG, WebP hoặc BMP và không quá 30 MB mỗi ảnh.")
                    total += size
                    if total > MAX_BATCH_BYTES:
                        raise ValueError("Tổng dung lượng ảnh không được quá 2 GB mỗi lượt.")
                    validated.append({"name": Path(str(item.get("name", "anh"))).name[:200], "size": size, "suffix": suffix})
                job_id = uuid.uuid4().hex
                job_path = jobs_dir / job_id
                job_path.mkdir(parents=True)
                workspace.service.store.session("conveyor", f"Lượt ảnh · {len(validated)} sản phẩm", session_id=job_id)
                for i, item in enumerate(validated, 1):
                    item["stored"] = f"image_{i:08d}{item['suffix']}"
                (job_path / "manifest.json").write_text(json.dumps({"files": validated, "rotation_degrees": rotation}, ensure_ascii=False, indent=2), encoding="utf-8")
                with jobs_lock:
                    jobs[job_id] = {"status": "uploading", "message": f"Đang nhận {len(validated)} ảnh…", "path": str(job_path), "files": validated, "rotation_degrees": rotation, "received": set(), "process": None}
                self._json(201, {"job_id": job_id, "count": len(validated)})
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})

        def _receive_image(self, job_id: str, index: int) -> None:
            with jobs_lock:
                job = jobs.get(job_id)
                if job is None or job.get("process") is not None:
                    self._json(404, {"error": "Lượt tải ảnh không còn mở."})
                    return
                files = job["files"]
                if not 1 <= index <= len(files) or index in job["received"]:
                    self._json(409, {"error": "Số thứ tự ảnh không hợp lệ hoặc đã tải."})
                    return
                spec = files[index - 1]
            length = self._content_length(MAX_IMAGE_BYTES)
            if length != spec["size"]:
                self._json(413, {"error": "Kích thước ảnh tải lên không khớp."})
                return
            path = Path(job["path"]) / spec["stored"]
            temporary = path.with_suffix(path.suffix + ".part")
            signature = b""
            try:
                remaining = length
                with temporary.open("wb") as target:
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise OSError("Kết nối bị ngắt khi tải ảnh.")
                        if len(signature) < 16:
                            signature += chunk[: 16 - len(signature)]
                        target.write(chunk)
                        remaining -= len(chunk)
                valid = (
                    (spec["suffix"] in {".jpg", ".jpeg"} and signature.startswith(b"\xff\xd8\xff"))
                    or (spec["suffix"] == ".png" and signature.startswith(b"\x89PNG\r\n\x1a\n"))
                    or (spec["suffix"] == ".webp" and signature.startswith(b"RIFF") and signature[8:12] == b"WEBP")
                    or (spec["suffix"] == ".bmp" and signature.startswith(b"BM"))
                )
                if not valid:
                    raise ValueError("Tệp không khớp định dạng ảnh đã chọn.")
                temporary.replace(path)
                with jobs_lock:
                    job["received"].add(index)
                    job["message"] = f"Đã tải {len(job['received'])}/{len(files)} ảnh lên máy local."
                self._json(201, {"received": index})
            except (OSError, ValueError) as exc:
                temporary.unlink(missing_ok=True)
                self._json(400, {"error": str(exc)})

        def _start_job(self, job_id: str) -> None:
            self._json(409, {"error": "Ảnh chỉ được phân loại qua worker thường trú khi track cắt ROI. Chờ GPU sẵn sàng rồi đưa ảnh vào băng chuyền."})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="Loopback only by default")
    parser.add_argument("--port", type=int, default=8033)
    parser.add_argument("--artifact", choices=ARTIFACTS, default=DEFAULT_ARTIFACT)
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--wsl-distro", default="TTTN-DeepStream")
    parser.add_argument("--wsl-root", default="")
    args = parser.parse_args()
    if args.preview_dir is None:
        args.preview_dir = ROOT / "artifacts" / args.artifact / "preview"
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.preview_dir, args.wsl_distro, args.wsl_root, args.artifact))
    server.daemon_threads = True
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    print(f"TensorRT preview ready at http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.RequestHandlerClass.workspace_api.close()
        server.server_close()


if __name__ == "__main__":
    main()
