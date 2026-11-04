#!/usr/bin/env bash
set -euo pipefail
IFS= read -r STRAW_CAPTURE_URI
export STRAW_CAPTURE_URI
session="${1:?session id required}"
[[ "$session" =~ ^[0-9a-f]{32}$ ]] || exit 2
out=/workspace/artifacts/b1_stream
source_file=/workspace/apps/stream_demo/capture_probe.cpp
export LD_LIBRARY_PATH="/opt/nvidia/deepstream/deepstream/lib:${LD_LIBRARY_PATH:-}"
source_hash=$(sha256sum "$source_file" | cut -d' ' -f1)
sdk_id=$(python3 -c 'import json; print(json.load(open("/workspace/artifacts/b1_stream/engine_cache.json"))["context"]["sdk_image_id"])')
build_key="$source_hash $sdk_id"
if [[ ! -s "$out/capture_probe" || ! -f "$out/capture_probe.source.sha256" || $(cat "$out/capture_probe.source.sha256") != "$build_key" ]]; then
  if ! pkg-config --exists gstreamer-app-1.0 || ! command -v g++ >/dev/null; then
    apt-get update -qq >&2
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends g++ pkg-config libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev >&2
  fi
  g++ -std=c++17 -O2 "$source_file" -o "$out/capture_probe.tmp" $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0) >&2
  mv "$out/capture_probe.tmp" "$out/capture_probe"
  printf '%s\n' "$build_key" > "$out/capture_probe.source.sha256"
fi
mkdir -p "$out/capture_pids"
echo $$ > "$out/capture_pids/$session.pid"
exec "$out/capture_probe"
