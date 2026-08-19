#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/content/TTTN}"
WHEEL="${MAMBA_WHEEL:-/content/drive/MyDrive/TTTN/wheels/mamba_ssm-2.3.2.post1+cu128torch2.11sm75-cp312-cp312-linux_x86_64.whl}"
VMAMBA_DIR="$PROJECT_ROOT/src/threecad_segmentation/third_party/VMamba"

python - <<'PY'
import sys, torch
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("CC:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
assert sys.version_info[:2] == (3, 12), "The saved wheel is cp312; use Python 3.12 or rebuild it."
assert torch.__version__.startswith("2.11."), "The saved wheel was built for Torch 2.11.x."
assert torch.version.cuda and torch.version.cuda.startswith("12.8"), "The saved wheel was built with CUDA 12.8."
assert torch.cuda.is_available(), "CUDA GPU is required."
assert torch.cuda.get_device_capability(0) == (7, 5), "The saved wheel is T4/sm75-specific."
PY

if [ ! -f "$WHEEL" ]; then
  echo "Missing prebuilt Mamba wheel: $WHEEL"
  exit 2
fi

python -m pip install -q "setuptools<82"
python -m pip install -q --no-deps "$WHEEL"
python -m pip install -q timm einops fvcore yacs termcolor packaging ninja chardet

mkdir -p "$PROJECT_ROOT/src/threecad_segmentation/third_party"
if [ ! -f "$VMAMBA_DIR/vmamba.py" ]; then
  git clone --depth 1 https://github.com/MzeroMiko/VMamba.git "$VMAMBA_DIR"
fi

export VMAMBA_REPO="$VMAMBA_DIR"
python - <<PY
import os, sys
import selective_scan_cuda
print("selective_scan_cuda: OK ->", selective_scan_cuda.__file__)
sys.path.insert(0, "$VMAMBA_DIR")
import vmamba
print("VMamba:", vmamba.__file__)
print("WITH_SELECTIVESCAN_MAMBA =", vmamba.WITH_SELECTIVESCAN_MAMBA)
assert vmamba.WITH_SELECTIVESCAN_MAMBA is True
print("VMAMBA CUDA SETUP: PASS")
PY
