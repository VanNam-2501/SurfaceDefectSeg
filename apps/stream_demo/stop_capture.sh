#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
artifact="${1:?artifact required}"
session="${2:?session required}"
case "$artifact" in b1_stream|b1_stream_512|b1_stream_512_pose_v1) ;; *) exit 2;; esac
[[ "$session" =~ ^[0-9a-f]{32}$ ]] || exit 2
suffix=$(printf '%s' "$root" | sha256sum | cut -c1-10)
lab="tttn-b1-deepstream-resident-$artifact-$suffix"
mounted=$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/workspace/artifacts/b1_stream"}}{{.Source}}{{end}}{{end}}' "$lab")
[[ "$mounted" == "$root/artifacts/$artifact" ]] || exit 2
docker exec "$lab" bash -c 'pidfile="/workspace/artifacts/b1_stream/capture_pids/$1.pid"; if [[ -f "$pidfile" ]]; then read -r pid < "$pidfile"; if [[ $(cat "/proc/$pid/comm" 2>/dev/null) == capture_probe ]]; then kill -TERM "$pid"; fi; rm -f "$pidfile"; fi' _ "$session"
