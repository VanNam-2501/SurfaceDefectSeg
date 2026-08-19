#!/usr/bin/env bash
# Install/check the VMamba runtime in a Kaggle GPU notebook.
#
# The wheel supplied with this project was built for one exact runtime:
# Python 3.12 + PyTorch 2.11 + CUDA 12.8 + Tesla T4/sm75.  Fail fast when
# Kaggle assigns another GPU or starts a different Python/PyTorch image.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/kaggle/working/threecad_ani_project}"
WHEEL="${MAMBA_WHEEL:-}"
VMAMBA_DIR="${VMAMBA_REPO:-$PROJECT_ROOT/src/threecad_segmentation/third_party/VMamba}"

python - <<'PY'
import sys
import torch
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("CC:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
assert torch.cuda.is_available(), "Enable a Kaggle GPU accelerator before setting up VMamba."
assert sys.version_info[:2] == (3, 12), "The supplied wheel is cp312; Kaggle must use Python 3.12."
assert torch.__version__.startswith("2.11."), "Install PyTorch 2.11.x before the supplied wheel."
assert torch.version.cuda and torch.version.cuda.startswith("12.8"), "Install the PyTorch CUDA 12.8 build."
assert torch.cuda.get_device_capability(0) == (7, 5), "Select a Tesla T4 GPU (sm75), not P100 or another GPU."
PY

if [ -z "$WHEEL" ] || [ ! -f "$WHEEL" ]; then
  echo "Missing MAMBA_WHEEL. Add a compatible mamba_ssm wheel as a Kaggle input," >&2
  echo "then set MAMBA_WHEEL=/kaggle/input/<wheel-dataset>/<wheel-file>.whl" >&2
  exit 2
fi

# Kaggle removes the "+" character from uploaded wheel filenames.  That turns
# the local-version segment into an invalid PEP 440 version and pip refuses to
# parse the file.  Copy the bytes to a canonical filename in the writable
# working directory before installation.  A real copy is intentional: using a
# symlink can still expose pip to the invalid source basename.
WHEEL_CACHE="${WHEEL_CACHE:-/kaggle/working/mamba_wheels}"
CANONICAL_WHEEL="$WHEEL_CACHE/mamba_ssm-2.3.2.post1+cu128torch2.11sm75-cp312-cp312-linux_x86_64.whl"
mkdir -p "$WHEEL_CACHE"
if [ "$WHEEL" != "$CANONICAL_WHEEL" ]; then
  cp -f "$WHEEL" "$CANONICAL_WHEEL"
fi
WHEEL="$CANONICAL_WHEEL"
echo "Installing Mamba wheel: $WHEEL"

python -m pip install -q "setuptools<82"
python -m pip install -q --no-deps "$WHEEL"
python -m pip install -q timm einops fvcore yacs termcolor packaging ninja chardet

mkdir -p "$(dirname "$VMAMBA_DIR")"
if [ ! -f "$VMAMBA_DIR/vmamba.py" ]; then
  echo "Cloning official VMamba source to $VMAMBA_DIR"
  git clone --depth 1 https://github.com/MzeroMiko/VMamba.git "$VMAMBA_DIR"
fi

export VMAMBA_REPO="$VMAMBA_DIR"
cd "$PROJECT_ROOT"
python tests/ml/test_vmamba_runtime.py
