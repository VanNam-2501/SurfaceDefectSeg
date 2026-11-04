"""Exercise shipped images through the live ROI API and serialized GPU engine."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from apps.stream_demo.resident_inference import raw_rgb_tensor
from apps.stream_demo.serving_contract import load_contract


def request(base, path, payload=None, raw=None):
    data, headers = raw, {}
    if payload is not None:
        data, headers = json.dumps(payload).encode(), {"Content-Type": "application/json"}
    try:
        with urlopen(Request(base + path, data=data, headers=headers), timeout=20) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, json.load(exc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8033")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/verification/demo_inference.json")
    args = parser.parse_args()
    identity = load_contract(ROOT / "artifacts/b1_stream_512_pose_v1")
    _, health = request(args.url, "/health")
    if (not health["runtime_verified"] or health["contract_sha256"] != identity["contract_sha256"]
            or Path(health["project_root"]).resolve() != ROOT):
        raise RuntimeError("HTTP service belongs to a different package/model")
    cases = json.loads((ROOT / "demo_samples/manifest.json").read_text(encoding="utf-8"))["cases"]
    files = [{"name": Path(row["path"]).name, "size": (ROOT / row["path"]).stat().st_size} for row in cases]
    code, job = request(args.url, "/api/jobs", {"files": files, "rotation_degrees": 0})
    if code != 201:
        raise RuntimeError(job)
    rows = []
    for index, case in enumerate(cases, 1):
        image = ROOT / case["path"]
        code, ack = request(args.url, f"/api/jobs/{job['job_id']}/images/{index}", raw=image.read_bytes())
        if code != 201:
            raise RuntimeError(ack)
        event = {"job_id": job["job_id"], "index": index, "track_id": 1000 + index,
                 "gate": {"crossed": False, "observations": 10, "roi_overlap": 1.0,
                          "confirmation_window": 5, "confirmed_frames": 5}}
        rejected, _ = request(args.url, "/api/trigger", event)
        if rejected != 409:
            raise RuntimeError("An image was admitted before crossing the ROI")
        event["gate"]["crossed"] = True
        code, ack = request(args.url, "/api/trigger", event)
        if code != 202:
            raise RuntimeError(ack)
        request_id = ack["request_id"]
        _, duplicate = request(args.url, "/api/trigger", event)
        if duplicate.get("request_id") != request_id:
            raise RuntimeError("Duplicate event scheduled another inference")
        deadline = time.monotonic() + 25
        while True:
            _, result = request(args.url, "/api/decisions/" + request_id)
            if result["status"] != "pending":
                break
            if time.monotonic() > deadline:
                raise TimeoutError("TensorRT inference timed out")
            time.sleep(.03)
        expected_hash = hashlib.sha256(raw_rgb_tensor(image, identity["model_size"]).tobytes()).hexdigest()
        if (result["status"] != "completed" or result["input_sha256"] != expected_hash
                or result["contract_sha256"] != identity["contract_sha256"]
                or result["event"]["image_index"] != index or result["event"]["track_id"] != 1000 + index):
            raise RuntimeError(f"Image/model/track mismatch: {request_id}")
        if result["server_event_to_result_ms"] is None or result["server_event_to_result_ms"] < 0:
            raise RuntimeError("Invalid latency measurement")
        rows.append({"sample_id": case["sample_id"], "expected": case["class_name"],
                     "predicted": result["class"], "input_sha256": expected_hash,
                     "duplicate_deduplicated": True, "invalid_gate_rejected": True,
                     "event_to_result_ms": result["server_event_to_result_ms"]})
    _, final_health = request(args.url, "/health")
    if final_health["runtime"]["generation"] != health["runtime"]["generation"]:
        raise RuntimeError("Worker reloaded during smoke")
    result = {"status": "PASS", "n": len(rows), "correct": sum(r["expected"] == r["predicted"] for r in rows),
              "scope": "API/engine smoke on shipped validation photos; not an accuracy benchmark or live camera test",
              "identity": identity, "worker_generation": health["runtime"]["generation"], "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
