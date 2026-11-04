"""Verified model identity shared by the HTTP service and TensorRT worker."""
from __future__ import annotations

import configparser
import hashlib
import json
from pathlib import Path

CLASSES = ("good", "bent", "loose", "missing", "pierced")
DEFAULT_ARTIFACT = "b1_stream_512_pose_v1"
ARTIFACTS = ("b1_stream", "b1_stream_512", "b1_stream_512_pose_v1")
PREPROCESS = {
    "exif_transpose": True, "color": "RGB", "resize": "square",
    "jpeg_decode": "PIL draft at target scale when source is over 2x model size",
    "resampling": "PIL bilinear", "layout": "NCHW", "dtype": "float32",
    "pixel_range": [0, 255], "normalization": "ImageNet embedded in ONNX",
    "additional_clockwise_rotation": "per request: 0, 90, 180 or 270 degrees after EXIF; default 0",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def load_contract(out: Path) -> dict:
    """Fail closed if the graph, engine, pinned checkpoint or policy disagree.

    The container mounts either artifact at the same /artifacts/b1_stream alias,
    so identity is based on file contents rather than the container path.
    """
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    cache = json.loads((out / "engine_cache.json").read_text(encoding="utf-8"))
    size = int(manifest["size"])
    if size not in (384, 512) or tuple(manifest["classes"]) != CLASSES:
        raise ValueError("Unsupported B1 input size or class order")
    config_name = manifest.get("config_file") or {
        "b1_stream": "straw_realtime.json",
        "b1_stream_512": "straw_realtime_stream512.json",
        "b1_stream_512_pose_v1": "straw_realtime_stream512_pose_v1.json",
    }.get(out.name)
    if not config_name:
        raise ValueError(f"No pinned stream config is declared for artifact {out.name}")
    config_path = out.parents[1] / "configs" / config_name
    pinned = json.loads(config_path.read_text(encoding="utf-8"))
    if (pinned["checkpoint_sha256"] != manifest["checkpoint_sha256"]
            or pinned["size"] != size or pinned.get("temperature", 1.0) != 1.0):
        raise ValueError("Artifact differs from pinned checkpoint/size/temperature")
    threshold = manifest.get("decision_threshold")
    if threshold is not None and not 0 < threshold < 1:
        raise ValueError("Invalid decision threshold")
    expected_threshold = None
    if pinned.get("decision_policy"):
        policy = json.loads((out.parents[1] / pinned["decision_policy"]).read_text(encoding="utf-8"))
        if (policy["checkpoint_sha256"] != pinned["checkpoint_sha256"]
                or policy["input_size"] != size or policy["temperature"] != 1.0):
            raise ValueError("Decision policy belongs to another model")
        expected_threshold = policy["threshold"]
    if threshold != expected_threshold:
        raise ValueError("Artifact decision policy differs from pinned policy")
    context = cache["context"]
    for field in ("checkpoint_sha256", "onnx_sha256"):
        if context[field] != manifest[field]:
            raise ValueError(f"Engine provenance mismatch: {field}")
    if file_sha256(out / f"b1_rawrgb{size}.onnx") != manifest["onnx_sha256"]:
        raise ValueError("ONNX bytes differ from manifest")
    engine_sha = file_sha256(out / "b1_rtx3050_fp16.engine")
    if engine_sha != cache["engine_sha256"]:
        raise ValueError("TensorRT engine bytes differ from certified engine")
    config = configparser.ConfigParser(interpolation=None)
    config.read(out / "nvinfer.txt", encoding="utf-8")
    canonical = json.dumps({s: dict(config.items(s)) for s in config.sections()}, sort_keys=True).encode()
    if hashlib.sha256(canonical).hexdigest() != context["inference_config_sha256"]:
        raise ValueError("Inference configuration differs from certified configuration")
    identity = {
        "schema": 1, "model_size": size, "classes": list(CLASSES),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "onnx_sha256": manifest["onnx_sha256"], "engine_sha256": engine_sha,
        "decision_threshold": threshold, "preprocessing": PREPROCESS,
    }
    identity["contract_sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return identity


def check_identity(record: dict, identity: dict) -> None:
    if record.get("contract_sha256") != identity["contract_sha256"]:
        raise ValueError("Worker/result model identity differs from the selected artifact; restart the worker")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    print(json.dumps(load_contract(parser.parse_args().artifact_dir), indent=2))
