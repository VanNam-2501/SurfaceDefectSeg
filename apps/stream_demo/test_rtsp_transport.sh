#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
out="$PWD/artifacts/b1_stream"
python3 apps/stream_demo/serve_rtsp.py "$out/load_30fps.mp4" --port 8555 > "$out/transport_server.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT INT TERM
for _ in {1..50}; do
  grep -q 'RTSP READY' "$out/transport_server.log" && break
  kill -0 "$server_pid" || { cat "$out/transport_server.log"; exit 3; }
  sleep .1
done
timeout 20 ffmpeg -nostdin -hide_banner -loglevel info -rtsp_transport tcp \
  -i rtsp://127.0.0.1:8555/straw -t 3 -an -f null - > "$out/transport.log" 2>&1
tail -n 3 "$out/transport.log"
