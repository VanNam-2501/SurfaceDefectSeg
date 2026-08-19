from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import selective_scan_cuda

PROJECT = Path(__file__).resolve().parents[2] / "src" / "threecad_segmentation"
VMAMBA = PROJECT / "third_party" / "VMamba"
if str(VMAMBA) not in sys.path:
    sys.path.insert(0, str(VMAMBA))
import vmamba  # noqa: E402

from vmamba_t import build_vmamba_t_binary  # noqa: E402

print("selective_scan_cuda:", selective_scan_cuda.__file__)
print("VMamba:", vmamba.__file__)
print("WITH_SELECTIVESCAN_MAMBA =", vmamba.WITH_SELECTIVESCAN_MAMBA)
assert vmamba.WITH_SELECTIVESCAN_MAMBA is True, "CUDA selective scan is NOT active"

device = torch.device("cuda")
model = build_vmamba_t_binary(pretrained=False).to(device).eval()
x = torch.randn(1, 3, 512, 512, device=device)

# This is intentionally inference-only.  It validates that the CUDA selective
# scan extension and the VMamba graph run on the assigned Kaggle GPU; it does
# not train, update a weight or download a pretrained backbone.
with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
    y = model(x)
torch.cuda.synchronize()

torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
    y = model(x)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

print("Input:", tuple(x.shape))
print("Output:", tuple(y.shape))
print("Forward seconds:", elapsed)
print("Peak VRAM GB:", torch.cuda.max_memory_allocated() / 1024**3)
assert y.shape == (1, 1, 512, 512)
assert torch.isfinite(y).all()
print("VMAMBA MODEL TEST: PASS")
