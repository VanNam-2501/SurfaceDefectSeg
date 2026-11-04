#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
for file in "$root"/apps/stream_demo/*.sh "$root"/scripts/deployment/*.sh; do
  bash -n "$file"
done
echo 'Bash syntax: PASS'
