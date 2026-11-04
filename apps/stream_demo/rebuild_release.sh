#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
suffix=$(printf '%s' "$root" | sha256sum | cut -c1-10)
export STRAW_STREAM_ARTIFACT=b1_stream_512_pose_v1
export STRAW_STREAM_LAB="tttn-b1-deepstream-release-build-$suffix"
exec bash "$root/apps/stream_demo/run_experiment.sh"
