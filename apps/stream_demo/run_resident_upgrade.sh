#!/usr/bin/env bash
# Compatibility entry point for the verified B1 512 launcher.
set -euo pipefail
exec bash "$(dirname "$0")/run_resident.sh" b1_stream_512
