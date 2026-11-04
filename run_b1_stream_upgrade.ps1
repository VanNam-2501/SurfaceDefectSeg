param([switch]$Prepare)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:STRAW_STREAM_ARTIFACT = 'b1_stream_512'
$env:STRAW_STREAM_LAB = 'tttn-b1-deepstream-lab-512'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$slashPath = $PSScriptRoot -replace '\\','/'
$linuxRoot = (& wsl -d TTTN-DeepStream -u root -- wslpath -u $slashPath).Trim()
if ($LASTEXITCODE -ne 0 -or -not $linuxRoot) { throw 'TTTN-DeepStream WSL project mount is unavailable.' }
if ($Prepare) {
    & $python scripts/research/prepare_b1_stream.py
    if ($LASTEXITCODE -ne 0) { throw 'B1 512 ONNX/parity preparation failed.' }
    wsl -d TTTN-DeepStream -u root -- env STRAW_STREAM_ARTIFACT=b1_stream_512 bash "$linuxRoot/apps/stream_demo/encode_replay.sh"
    if ($LASTEXITCODE -ne 0) { throw 'Validation video preparation failed.' }
    & $python scripts/research/reference_b1_stream.py
    if ($LASTEXITCODE -ne 0) { throw 'B1 512 H264 reference failed.' }
}
wsl -d TTTN-DeepStream -u root -- env STRAW_STREAM_ARTIFACT=b1_stream_512 STRAW_STREAM_LAB=tttn-b1-deepstream-lab-512 bash "$linuxRoot/apps/stream_demo/run_experiment.sh"
if ($LASTEXITCODE -ne 0) { throw 'DeepStream 512 experiment failed.' }
& $python scripts/research/report_b1_stream.py
if ($LASTEXITCODE -ne 0) { throw 'DeepStream 512 report failed.' }
