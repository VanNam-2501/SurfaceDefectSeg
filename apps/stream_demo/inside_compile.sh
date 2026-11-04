#!/usr/bin/env bash
set -euo pipefail
sdk=/opt/nvidia/deepstream/deepstream
export LD_LIBRARY_PATH="$sdk/lib:${LD_LIBRARY_PATH:-}"
if ! command -v g++ >/dev/null || ! pkg-config --exists gstreamer-1.0; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    g++ pkg-config libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
fi
headers="$sdk/sources/includes"
if [[ ! -f "$headers/gstnvdsinfer.h" ]]; then
  echo 'DeepStream development headers were not found:'
  find /opt/nvidia/deepstream -name gstnvdsinfer.h -o -name gstnvdsmeta.h
  exit 2
fi
g++ -std=c++17 -O2 /workspace/apps/stream_demo/deepstream_probe.cpp \
  -o /workspace/artifacts/b1_stream/deepstream_probe \
  -I"$headers" -I/usr/local/cuda/include -L"$sdk/lib" -Wl,-rpath,"$sdk/lib" \
  $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0) -lnvdsgst_meta -lnvds_meta -pthread
deepstream-app --version-all > /workspace/artifacts/b1_stream/sdk_version.txt 2>&1 || true
echo 'DeepStream probe compiled'
