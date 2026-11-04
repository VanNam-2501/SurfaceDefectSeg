param([switch]$Recovery)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$b1Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $b1Python scripts/research/optimize_b1_deployment.py --phase all
if ($LASTEXITCODE -ne 0) { throw 'Deployment study failed; completed artifacts are preserved.' }
& $b1Python scripts/research/calibrate_b1_deployment.py
if ($LASTEXITCODE -ne 0) { throw 'Calibration study failed.' }
if ($Recovery) {
    & $b1Python scripts/research/recover_b1_deployment.py --method finetune --size 384 --epochs 5
    if ($LASTEXITCODE -ne 0) { throw 'Fine tuning stopped; resume from last.pt.' }
    & $b1Python scripts/research/recover_b1_deployment.py --method qat --size 512 --epochs 5
    if ($LASTEXITCODE -ne 0) { throw 'QAT stopped; resume from last.pt.' }
    foreach ($b1Seed in @(42, 43, 44)) {
        & $b1Python scripts/research/recover_b1_deployment.py --method finetune --size 384 --epochs 3 --seed $b1Seed --tag safety
        if ($LASTEXITCODE -ne 0) { throw 'Safety recovery stopped; resume from last.pt.' }
    }
    & $b1Python scripts/research/finalize_b1_deployment.py
    if ($LASTEXITCODE -ne 0) { throw 'Final benchmarking failed.' }
    & $b1Python scripts/research/select_b1_deployment.py
    if ($LASTEXITCODE -ne 0) { throw 'Deployment selection failed.' }
}
