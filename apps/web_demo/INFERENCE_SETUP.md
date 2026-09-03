# Local inference setup

The demo exposes nine modes: three original raw model baselines, three
Adaptive single-model policies, and three Spatial two-model ensembles.
Raw baselines use the frozen Validation thresholds 0.49 (U-Net), 0.66
(SegFormer), and 0.51 (VMamba). From the workspace root, run:

```powershell
.\run_demo.ps1
```

The launcher uses the local final checkpoints and discovers the Adaptive
single-model policy plus the six Spatial policies under artifacts/reports/final/decision_and_test_audit/spatial. No external download path is required. ADAPTIVE_POLICY may override the
Adaptive policy, DECISION_POLICY may override one matching Spatial policy, and
DECISION_POLICY_ROOT may override the Spatial policy directory.

The default trained checkpoints are in:

```text
artifacts/checkpoints/final/unet_best.pt
artifacts/checkpoints/final/segformer_best.pt
artifacts/checkpoints/final/vmamba_best.pt
```

The paths may also be supplied through `UNET_CHECKPOINT`,
`SEGFORMER_CHECKPOINT`, and `VMAMBA_CHECKPOINT`. Start `backend/app.py` with
Uvicorn, then open the web interface. The VMamba checkpoint additionally
requires the VMamba runtime described in the main experiment README.

## Checkpoints from a retraining run

`scripts/training/retrain_models.ps1` stores its best checkpoints without overwriting older
experiments. Point the demo directly at the desired run; copying files is not
required. For example, from PowerShell:

```powershell
$env:SEGMENTATION_PROJECT_ROOT = "E:\Project\TTTN\src\threecad_segmentation"
$env:UNET_CHECKPOINT = "E:\Project\TTTN\artifacts\training\unet_r18\cleaned_v1_seed42\checkpoints\best.pt"
$env:SEGFORMER_CHECKPOINT = "E:\Project\TTTN\artifacts\training\segformer_b0\cleaned_v1_seed42\checkpoints\best.pt"
$env:VMAMBA_CHECKPOINT = "E:\Project\TTTN\artifacts\training\vmamba_t_s2l5\cleaned_v1_seed42\checkpoints\best.pt"

uvicorn backend.app:app --reload --port 8000
```

Set only the variables for checkpoints that exist. The Health/status panel
will show a missing or runtime-incompatible model as unavailable; it does not
fabricate a prediction.
