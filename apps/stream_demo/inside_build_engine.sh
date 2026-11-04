#!/usr/bin/env bash
# Conversion stage only. Run once on the target NVIDIA GPU, before inference.
set -euo pipefail
out=/workspace/artifacts/b1_stream
export LD_LIBRARY_PATH="/opt/nvidia/deepstream/deepstream/lib:${LD_LIBRARY_PATH:-}"
size=$(python3 -c 'import json; print(json.load(open("/workspace/artifacts/b1_stream/manifest.json"))["size"])')
test -s "$out/b1_rawrgb${size}.onnx"
trtexec --onnx="$out/b1_rawrgb${size}.onnx" \
  --saveEngine="$out/b1_rtx3050_fp16.engine.tmp" \
  --fp16 --inputIOFormats=fp32:chw --outputIOFormats=fp32:chw \
  --memPoolSize=workspace:512 --builderOptimizationLevel=3 \
  --timingCacheFile="$out/b1_rtx3050.timing.cache" \
  --profilingVerbosity=detailed --exportLayerInfo="$out/tensorrt_layers.json" \
  --skipInference
test -s "$out/b1_rtx3050_fp16.engine.tmp"
mv "$out/b1_rtx3050_fp16.engine.tmp" "$out/b1_rtx3050_fp16.engine"
