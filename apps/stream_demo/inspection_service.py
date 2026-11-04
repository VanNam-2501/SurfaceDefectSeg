"""One journal entry per product, including quality reviews and failed inference."""
from __future__ import annotations
from pathlib import Path
import threading
import time
from PIL import Image
try:
    from .image_quality import assess, QualityConfig
    from .inspection_store import InspectionStore
except ImportError:
    from image_quality import assess, QualityConfig
    from inspection_store import InspectionStore


class InspectionService:
    def __init__(self, root, resident, path=None):
        self.root, self.resident = Path(root), resident
        self.store = InspectionStore(path or self.root / "artifacts/inspection/history.sqlite3")
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._collect, daemon=True, name="inspection-journal")
        self.thread.start()

    def submit(self, path, event_key, session_id, track_id, metadata, quality=None, config=None):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Inspection image must be inside the project")
        if quality is None:
            with Image.open(path) as image:
                quality = assess(image, config or QualityConfig())
        with self.lock:
            row = self.store.add(event_key, session_id, track_id, path.relative_to(self.root).as_posix(),
                                 quality, self.resident.identity, metadata)
            if row["state"] != "preparing":
                return row
            if not quality["acceptable"]:
                self.store.finish(row["id"], "review", error="Image quality: " + ", ".join(quality["reasons"]))
            else:
                try:
                    request_id = self.resident.submit(path, event_key, dict(metadata, inspection_id=row["id"]))
                    self.store.pending(row["id"], request_id)
                except (RuntimeError, OSError, ValueError) as exc:
                    self.store.finish(row["id"], "error", error=str(exc))
            return self.store.get(row["id"])

    def collect_once(self):
        for row in self.store.unfinished():
            if row["state"] == "preparing":
                if time.time() - row["created"] > 10:
                    self.store.finish(row["id"], "error", error="Submission interrupted before GPU admission")
                continue
            try:
                result = self.resident.result(row["request_id"])
                if result["status"] == "completed":
                    if result.get("contract_sha256") != row["identity"]["contract_sha256"]:
                        raise ValueError("Result belongs to another model")
                    result["confidence_calibrated"] = False
                    self.store.finish(row["id"], "defect" if result["defective"] else "good", result)
                elif result["status"] == "error":
                    self.store.finish(row["id"], "error", result, result.get("error", "GPU inference error"))
                elif time.time() - row["created"] > 30:
                    self.store.finish(row["id"], "error", error="GPU result timeout; product not classified")
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                self.store.finish(row["id"], "error", error=str(exc))

    def _collect(self):
        while not self.closed.wait(.1):
            try:
                self.collect_once()
            except Exception:
                # SQLite busy / DrvFS publication can be transient. No prediction is invented.
                self.closed.wait(.25)

    def close(self):
        self.closed.set()
        self.thread.join(timeout=2)
