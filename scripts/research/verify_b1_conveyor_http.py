"""Exercise real image upload -> ROI trigger -> verified TensorRT result."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from apps.stream_demo.resident_inference import raw_rgb_tensor
from apps.stream_demo.serving_contract import CLASSES


def request(base, path, payload=None, raw=None):
    headers = {}
    data = raw
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    try:
        with urlopen(Request(base + path, data=data, headers=headers), timeout=15) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8033")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rotation-degrees", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--legacy30", action="store_true")
    parser.add_argument("--artifact", choices=("b1_stream_512_pose_v1", "b1_stream_512"), default="b1_stream_512_pose_v1")
    args = parser.parse_args()
    _, health = request(args.url, "/health")
    if not health["runtime_verified"] or health["artifact"] != args.artifact:
        raise RuntimeError("The HTTP service has not verified the intended worker")
    frame = pd.read_csv(ROOT / "artifacts/straw_global_local/data_v1/test.csv")
    cases = [frame[frame.class_name == label].iloc[0] for label in CLASSES]
    if args.legacy30:
        manifest = json.loads((ROOT / "artifacts/b1_conveyor_audit/verified_v1/rotation_manifest.json").read_text())
        cases = [SimpleNamespace(image_path=row["legacy_path"], class_name="bent",
            sample_id=Path(row["legacy_path"]).stem) for row in manifest]
    files = [{"name": Path(row.image_path).name, "size": Path(row.image_path).stat().st_size} for row in cases]
    invalid_rotation, _ = request(args.url, "/api/jobs", payload={"files": files, "rotation_degrees": 45})
    if invalid_rotation != 400:
        raise RuntimeError("API accepted an unsupported input rotation")
    code, job = request(args.url, "/api/jobs", payload={"files": files, "rotation_degrees": args.rotation_degrees})
    if code != 201:
        raise RuntimeError(job)
    rows = []
    for index, row in enumerate(cases, 1):
        code, ack = request(args.url, f"/api/jobs/{job['job_id']}/images/{index}", raw=Path(row.image_path).read_bytes())
        if code != 201:
            raise RuntimeError(ack)
        trigger = {"job_id": job["job_id"], "index": index, "track_id": 1000 + index,
                   "gate": {"crossed": False, "observations": 10, "roi_overlap": 1.0}}
        rejected, _ = request(args.url, "/api/trigger", payload=trigger)
        if rejected != 409:
            raise RuntimeError("The API accepted an event before the ROI gate crossed")
        trigger["gate"]["crossed"] = True
        code, ack = request(args.url, "/api/trigger", payload=trigger)
        if code != 202:
            raise RuntimeError(ack)
        request_id = ack["request_id"]
        _, duplicate = request(args.url, "/api/trigger", payload=trigger)
        if duplicate.get("request_id") != request_id:
            raise RuntimeError("Duplicate ROI event enqueued a second image inference")
        deadline = time.monotonic() + 20
        while True:
            _, result = request(args.url, "/api/decisions/" + request_id)
            if result["status"] != "pending":
                break
            if time.monotonic() > deadline:
                raise TimeoutError("HTTP inference did not complete")
            time.sleep(.03)
        expected_input = hashlib.sha256(raw_rgb_tensor(Path(row.image_path), 512, args.rotation_degrees).tobytes()).hexdigest()
        if (result["status"] != "completed" or result["input_sha256"] != expected_input
                or result["contract_sha256"] != health["contract_sha256"]
                or result["event"]["image_index"] != index
                or result["event"]["track_id"] != 1000 + index):
            raise RuntimeError(f"Wrong image/model/track association: {result}")
        if result["server_event_to_result_ms"] is None or result["server_event_to_result_ms"] < 0:
            raise RuntimeError("HTTP event timing is invalid")
        rows.append({"sample_id": row.sample_id, "expected": row.class_name,
            "prediction": result["class"], "request_id": request_id,
            "input_sha256": result["input_sha256"], "invalid_gate_status": rejected,
            "duplicate_deduplicated": True, "event_to_result_ms": result["server_event_to_result_ms"]})
    legacy_code, _ = request(args.url, f"/api/jobs/{job['job_id']}-run", payload={})
    if legacy_code != 409:
        raise RuntimeError("Legacy video batch fallback unexpectedly bypassed the resident worker")
    output = {"n": len(rows), "correct": sum(row["expected"] == row["prediction"] for row in rows),
              "identity": {key: health[key] for key in ("checkpoint_sha256", "engine_sha256", "contract_sha256")},
              "legacy_batch_fallback_rejected": True, "rotation_degrees": args.rotation_degrees, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
