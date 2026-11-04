#!/usr/bin/env bash
# Local functional-test source; never claims to be a physical camera.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
out="$root/artifacts/verification/upgrade_v2"
mkdir -p "$out"
pidfile="$out/rtsp_replay.pid"
if [[ -f "$pidfile" ]]; then
  read -r pid < "$pidfile"
  if kill -0 "$pid" 2>/dev/null; then echo 'Replay RTSP server already active'; exit 0; fi
fi
nohup python3 "$root/apps/stream_demo/serve_rtsp.py" "$root/demo_samples/conveyor_replay.mp4" --port 8555 > "$out/rtsp_replay.log" 2>&1 </dev/null &
echo $! > "$pidfile"
for _ in {1..50}; do
  grep -q 'RTSP READY' "$out/rtsp_replay.log" && { echo 'Replay source: rtsp://127.0.0.1:8555/straw'; exit 0; }
  sleep .1
done
cat "$out/rtsp_replay.log" >&2
exit 2
