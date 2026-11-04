#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
pidfile="$root/artifacts/verification/upgrade_v2/rtsp_replay.pid"
if [[ -f "$pidfile" ]]; then
  read -r pid < "$pidfile"
  if [[ -f "/proc/$pid/cmdline" ]] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Fq "$root/apps/stream_demo/serve_rtsp.py"; then
    kill "$pid"
  fi
  rm -f "$pidfile"
fi
