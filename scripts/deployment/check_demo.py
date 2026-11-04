"""Check the shipped engine and optionally its WSL runtime, without PyTorch."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from apps.stream_demo.serving_contract import ARTIFACTS, DEFAULT_ARTIFACT, file_sha256, load_contract

IMAGE = "ghcr.io/nvidia/deepstream:9.2-samples-multiarch"


def check(artifact=DEFAULT_ARTIFACT, runtime=False, distro="TTTN-DeepStream"):
    out = ROOT / "artifacts" / artifact
    identity = load_contract(out)
    config_name = {"b1_stream": "straw_realtime.json", "b1_stream_512": "straw_realtime_stream512.json",
                   "b1_stream_512_pose_v1": "straw_realtime_stream512_pose_v1.json"}[artifact]
    config = json.loads((ROOT / "configs" / config_name).read_text(encoding="utf-8"))
    if file_sha256(ROOT / config["checkpoint"]) != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint bytes differ from the pinned model")
    if "onnx-file=" in (out / "nvinfer.txt").read_text().replace(" ", ""):
        raise ValueError("nvinfer must load the serialized engine")
    import numpy
    from PIL import Image
    result = {"status": "PASS", "artifact": artifact, "identity": identity,
              "host": {"python": sys.version.split()[0], "numpy": numpy.__version__,
                       "Pillow": importlib.metadata.version("Pillow")},
              "runtime_checked": runtime}
    if runtime:
        def wsl(*args):
            process = subprocess.run(["wsl.exe", "-d", distro, "-u", "root", "--", *args],
                                     capture_output=True, timeout=25)
            output = process.stdout.decode("utf-8", errors="replace").replace("\x00", "").strip()
            if process.returncode:
                error = process.stderr.decode("utf-8", errors="replace").replace("\x00", "")
                raise RuntimeError(f"WSL command failed: {output} {error}")
            return output
        cache = json.loads((out / "engine_cache.json").read_text())["context"]
        actual_image = wsl("docker", "image", "inspect", IMAGE, "--format", "{{.Id}}")
        actual_gpu = wsl("nvidia-smi", "--query-gpu=name,driver_version,uuid", "--format=csv,noheader")
        if actual_image != cache["sdk_image_id"] or actual_gpu != cache["gpu_identity"]:
            raise RuntimeError("GPU/driver/SDK differs from the certified engine. Run rebuild_demo_engine.ps1 on the target machine.")
        wsl("python3", "-c", "import numpy, gi")
        result["runtime"] = {"distro": distro, "image_id": actual_image, "gpu_identity": actual_gpu}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", choices=ARTIFACTS, default=DEFAULT_ARTIFACT)
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--distro", default="TTTN-DeepStream")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = check(args.artifact, args.runtime, args.distro)
    except Exception as exc:
        result = {"status": "FAIL", "error": str(exc)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
