"""Windows-side event queue for the already-loaded TensorRT worker."""
import hashlib
import csv
import json
from collections import OrderedDict
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
from PIL import Image, ImageOps

try:
    from .serving_contract import DEFAULT_ARTIFACT, ARTIFACTS, load_contract, check_identity
    from .runtime_backend import native_runtime, host_command
except ImportError:
    from serving_contract import DEFAULT_ARTIFACT, ARTIFACTS, load_contract, check_identity
    from runtime_backend import native_runtime, host_command


def raw_rgb_tensor(path: Path, size: int, rotation_degrees: int = 0) -> np.ndarray:
    if type(rotation_degrees) is not int or rotation_degrees not in (0, 90, 180, 270):
        raise ValueError("Image rotation must be 0, 90, 180 or 270 clockwise degrees")
    with Image.open(path) as source:
        # JPEG can decode directly at 1/2, 1/4, or 1/8 scale. The original
        # full-resolution decode made common 3000x3000 conveyor photos spend
        # ~200 ms in CPU preprocessing before the 19 ms TensorRT call.
        # Keep small inputs and non-JPEG formats on the exact original path.
        if source.format == "JPEG" and max(source.size) > size * 2:
            source.draft("RGB", (size, size))
        image = ImageOps.exif_transpose(source).convert("RGB")
        if rotation_degrees:
            image = image.transpose({90: Image.Transpose.ROTATE_270,
                180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_90}[rotation_degrees])
        image = image.resize((size, size), Image.Resampling.BILINEAR)
        return np.ascontiguousarray(np.asarray(image).transpose(2, 0, 1), dtype=np.float32)


def read_json_snapshot(path: Path) -> dict:
    # DrvFS rename/publication can briefly overlap a Windows sharing lock.
    # Retry the read, never interpret a lock as a prediction or a dead GPU.
    for attempt in range(5):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.01 * (attempt + 1))


class ResidentInference:
    def __init__(self, root: Path, distro: str, wsl_root: str, artifact: str = DEFAULT_ARTIFACT):
        if artifact not in ARTIFACTS:
            raise ValueError('Unsupported stream artifact')
        self.root, self.distro, self.wsl_root = root, distro, wsl_root
        self.out = root / 'artifacts' / artifact
        self.queue = self.out / "resident_queue"
        self.identity = load_contract(self.out)
        self.size = self.identity['model_size']
        self.queue.mkdir(parents=True, exist_ok=True)
        self.process = None
        self.lock = threading.Lock()
        self.last_start = 0
        self.submitted = {}
        self.observed_timings = OrderedDict()
        self.log = None
        self.launch_error = ""
        self.closed = threading.Event()
        self.hardware_lock = threading.Lock()
        self.hardware_checked_at = 0.0
        try:
            gpu_identity = json.loads((self.out / "engine_cache.json").read_text(encoding="utf-8"))["context"]["gpu_identity"]
            gpu_parts = [part.strip() for part in gpu_identity.split(",")]
            self.hardware_snapshot = {"gpu_name": gpu_parts[0] if gpu_parts else "GPU"}
            if len(gpu_parts) > 1:
                self.hardware_snapshot["gpu_driver"] = gpu_parts[1]
        except (OSError, ValueError, KeyError, TypeError):
            self.hardware_snapshot = {"gpu_name": "GPU"}
        if wsl_root or native_runtime():
            threading.Thread(target=self._monitor, daemon=True).start()

    def hardware_status(self):
        """Cache live GPU telemetry from the WSL host; keep certified identity on read errors."""
        with self.hardware_lock:
            if time.monotonic() - self.hardware_checked_at >= 2:
                self.hardware_checked_at = time.monotonic()
                try:
                    result = subprocess.run(
                        host_command(self.distro, "nvidia-smi",
                         "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
                         "--format=csv,noheader,nounits"),
                        check=True, capture_output=True, text=True, timeout=3,
                    )
                    values = next(csv.reader([result.stdout.strip()], skipinitialspace=True))
                    if len(values) >= 6:
                        def number(value, cast=float):
                            try:
                                return cast(value.strip())
                            except (ValueError, TypeError):
                                return None
                        self.hardware_snapshot.update({
                            "gpu_name": values[0].strip(), "vram_total_mib": number(values[1], int),
                            "vram_used_mib": number(values[2], int), "gpu_utilization_percent": number(values[3], int),
                            "gpu_temperature_c": number(values[4], int), "gpu_power_w": number(values[5]),
                            "telemetry_available": True, "telemetry_updated_at": time.time(),
                        })
                    else:
                        self.hardware_snapshot["telemetry_available"] = False
                except (OSError, subprocess.SubprocessError, StopIteration):
                    self.hardware_snapshot["telemetry_available"] = False
            return dict(self.hardware_snapshot)

    def status(self):
        try:
            value = read_json_snapshot(self.queue / "runtime.json")
            fresh = time.time() - value.get("heartbeat", 0) < 5
            if fresh:
                try:
                    check_identity(value, self.identity)
                except ValueError as exc:
                    return dict(state="error", error=str(exc), **self.identity)
                return value
        except (OSError, ValueError):
            pass
        return {"state": "starting" if self.wsl_root or native_runtime() else "unconfigured",
                "product_calls": 0, "warmup_calls": 0, "generation": "", "error": self.launch_error,
                **self.identity}

    def _monitor(self):
        while not self.closed.is_set():
            try:
                state = self.status()
                alive = self.process is not None and self.process.poll() is None
                if state["state"] not in ("ready", "loading") and not alive and time.time() - self.last_start > 10:
                    self.last_start = time.time()
                    if self.log is not None:
                        self.log.close()
                    self.log = (self.queue / "launcher.log").open("ab")
                    command = ([sys.executable, str(self.root / 'apps/stream_demo/resident_worker.py'),
                        '--engine', str(self.out / 'b1_rtx3050_fp16.engine'), '--queue', str(self.queue),
                        '--keepalive-seconds', '1'] if native_runtime() else
                        host_command(self.distro, 'bash', f"{self.wsl_root.rstrip('/')}/apps/stream_demo/run_resident.sh", self.out.name))
                    self.process = subprocess.Popen(command,
                        cwd=self.root, stdout=self.log, stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    self.launch_error = ""
            except OSError as exc:
                self.launch_error = str(exc)
            self.closed.wait(1)

    def close(self):
        self.closed.set()
        if native_runtime() and self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def submit(self, path: Path, event_key: str, metadata: dict):
        state = self.status()
        if state["state"] != "ready":
            raise RuntimeError("TensorRT đang khởi động hoặc kết nối GPU bị gián đoạn.")
        generation = state["generation"]
        request_id = uuid.uuid5(uuid.NAMESPACE_URL,
            f"{self.identity['contract_sha256']}:{generation}:{event_key}").hex
        with self.lock:
            result = self.queue / f"{request_id}.json"
            pending = self.queue / f"{request_id}.f32"
            if result.exists() or pending.exists() or request_id in self.submitted:
                return request_id
            if len(list(self.queue.glob("*.f32"))) >= 32:
                raise RuntimeError("Hàng đợi AI đã đầy; cần giảm tốc băng chuyền.")
            started = time.time()
            started_perf = time.perf_counter()
            rotation_degrees = metadata.get("rotation_degrees", 0)
            tensor = raw_rgb_tensor(path, self.size, rotation_degrees)
            payload = tensor.tobytes()
            meta = dict(metadata, request_id=request_id, submitted_at=started,
                        rotation_degrees=rotation_degrees,
                        contract_sha256=self.identity['contract_sha256'], generation=generation,
                        input_sha256=hashlib.sha256(payload).hexdigest(),
                        preprocess_ms=(time.perf_counter() - started_perf) * 1000)
            (self.queue / f"{request_id}.event.json").write_text(json.dumps(meta), encoding="utf-8")
            temporary = self.queue / f"{request_id}.part"
            temporary.write_bytes(payload)
            self.submitted[request_id] = {"started_perf": started_perf}
            temporary.replace(pending)
        return request_id

    def result(self, request_id):
        path = self.queue / f"{request_id}.json"
        try:
            result = read_json_snapshot(path)
            meta = read_json_snapshot(self.queue / f"{request_id}.event.json")
            if result.get("status") == "completed":
                check_identity(result, self.identity)
                check_identity(meta, self.identity)
                if (result.get("generation") != meta["generation"]
                        or result.get("input_sha256") != meta["input_sha256"]
                        or result.get("request_id") != request_id):
                    raise ValueError("Prediction does not belong to this image/worker generation")
            result["event"] = meta
            if result.get("status") in ("completed", "error"):
                # Windows and WSL wall clocks can drift. Measure both ends on
                # the Windows host and freeze the first-observation duration.
                with self.lock:
                    submitted = self.submitted.pop(request_id, None)
                    if submitted is not None:
                        self.observed_timings[request_id] = (time.perf_counter() - submitted["started_perf"]) * 1000
                        while len(self.observed_timings) > 1024:
                            self.observed_timings.popitem(last=False)
                    result["server_event_to_result_ms"] = self.observed_timings.get(request_id)
                result["server_timing_scope"] = "Windows submit to first observed result; includes queue and polling"
            return result
        except (FileNotFoundError, PermissionError):
            return {"status": "pending", "request_id": request_id}
        except (ValueError, KeyError) as exc:
            return {"status": "error", "request_id": request_id, "error": str(exc)}
