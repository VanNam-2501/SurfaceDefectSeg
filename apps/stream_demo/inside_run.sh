#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH="/opt/nvidia/deepstream/deepstream/lib:${LD_LIBRARY_PATH:-}"
test -s /workspace/artifacts/b1_stream/b1_rtx3050_fp16.engine
if grep -Eq '^[[:space:]]*onnx-file[[:space:]]*=' /workspace/artifacts/b1_stream/nvinfer.txt; then
  echo 'Runtime must load the serialized TensorRT engine only, not a build graph' >&2
  exit 2
fi
export STRAW_DECISION_THRESHOLD=$(python3 -c 'import json; print(json.load(open("/workspace/artifacts/b1_stream/manifest.json")).get("decision_threshold") or "")')
if [[ $# -ge 4 && $# -le 6 ]]; then
  exec /workspace/artifacts/b1_stream/deepstream_probe "$@"
else
  echo 'Usage: inside_run.sh URI NvinferConfig OutputJsonl Seconds [PreviewDirectory]' >&2
  exit 2
fi
