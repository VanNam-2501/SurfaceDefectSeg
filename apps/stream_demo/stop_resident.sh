#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
artifact="${1:-b1_stream_512_pose_v1}"
case "$artifact" in b1_stream|b1_stream_512|b1_stream_512_pose_v1) ;; *) exit 2;; esac
suffix=$(printf '%s' "$root" | sha256sum | cut -c1-10)
lab="tttn-b1-deepstream-resident-$artifact-$suffix"
if docker container inspect "$lab" >/dev/null 2>&1; then
  mounted=$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/workspace/artifacts/b1_stream"}}{{.Source}}{{end}}{{end}}' "$lab")
  [[ "$mounted" == "$root/artifacts/$artifact" ]] || { echo 'Container mount mismatch' >&2; exit 2; }
  docker stop -t 5 "$lab" >/dev/null
fi
