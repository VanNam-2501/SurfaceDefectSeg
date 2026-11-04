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
[[ "$mounted" == "$root/artifacts/$artifact" ]] || { echo 'Capture container mount mismatch' >&2; exit 2; }
exec docker exec -i "$lab" bash /workspace/apps/stream_demo/inside_capture.sh "$session"
