"""Optional GPU integration test for the VMamba selective-scan runtime."""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import torch


@unittest.skipUnless(torch.cuda.is_available(), "VMamba integration test requires CUDA.")
class VmambaRuntimeTests(unittest.TestCase):
    def test_vmamba_cuda_runtime(self) -> None:
        try:
            import selective_scan_cuda
        except ModuleNotFoundError:
            self.skipTest("selective_scan_cuda is not installed.")

        project = Path(__file__).resolve().parents[2] / "src" / "threecad_segmentation"
        vmamba_path = project / "third_party" / "VMamba"
        if str(vmamba_path) not in sys.path:
            sys.path.insert(0, str(vmamba_path))
        import vmamba  # noqa: PLC0415
        from vmamba_t import build_vmamba_t_binary  # noqa: PLC0415

        self.assertTrue(vmamba.WITH_SELECTIVESCAN_MAMBA, "CUDA selective scan is NOT active")
        print("selective_scan_cuda:", selective_scan_cuda.__file__)
        print("VMamba:", vmamba.__file__)

        device = torch.device("cuda")
        model = build_vmamba_t_binary(pretrained=False).to(device).eval()
        x = torch.randn(1, 3, 512, 512, device=device)

        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            model(x)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(x)
        torch.cuda.synchronize()

        elapsed = time.perf_counter() - started
        print("Input:", tuple(x.shape))
        print("Output:", tuple(output.shape))
        print("Forward seconds:", elapsed)
        print("Peak VRAM GB:", torch.cuda.max_memory_allocated() / 1024**3)
        self.assertEqual(tuple(output.shape), (1, 1, 512, 512))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
