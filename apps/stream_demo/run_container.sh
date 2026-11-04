#!/usr/bin/env bash
# Explicit GPU/video capabilities; headless, no host --privileged flag.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
mode="${1:-file}"
lab="${STRAW_STREAM_LAB:-tttn-b1-deepstream-lab}"
name="straw-ds-${mode}"
uri="file:///workspace/artifacts/b1_stream/quality.mp4"
output="${mode}.jsonl"
seconds=0
if [[ "$mode" == rtsp ]]; then
  # Camera credentials, if supplied later, belong in env, not source history.
  uri="${STRAW_RTSP_URI:-rtsp://127.0.0.1:8554/straw}"
  seconds="${STRAW_TEST_SECONDS:-8}"
elif [[ "$mode" != file ]]; then echo 'Use file or rtsp'; exit 2; fi
docker exec "$lab" bash /workspace/apps/stream_demo/inside_run.sh \
  "$uri" /workspace/artifacts/b1_stream/nvinfer.txt \
  "/workspace/artifacts/b1_stream/$output" "$seconds"
