param([switch]$Prepare)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:STRAW_STREAM_ARTIFACT = 'b1_stream'
$env:STRAW_STREAM_LAB = 'tttn-b1-deepstream-lab'
$b1Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$slashPath = $PSScriptRoot -replace '\\','/'
$b1LinuxRoot = (& wsl -d TTTN-DeepStream -u root -- wslpath -u $slashPath).Trim()
if ($LASTEXITCODE -ne 0 -or -not $b1LinuxRoot) { throw 'Dedicated WSL TTTN-DeepStream is unavailable.' }
if ($Prepare) {
    & $b1Python scripts/research/prepare_b1_stream.py
    if ($LASTEXITCODE -ne 0) { throw 'Stream ONNX/input preparation failed.' }
    wsl -d TTTN-DeepStream -u root -- env STRAW_STREAM_ARTIFACT=b1_stream bash "$b1LinuxRoot/apps/stream_demo/encode_replay.sh"
    if ($LASTEXITCODE -ne 0) { throw 'H.264 replay preparation failed.' }
    & $b1Python scripts/research/reference_b1_stream.py
    if ($LASTEXITCODE -ne 0) { throw 'H.264 reference failed.' }
}
wsl -d TTTN-DeepStream -u root -- env STRAW_STREAM_ARTIFACT=b1_stream STRAW_STREAM_LAB=tttn-b1-deepstream-lab bash "$b1LinuxRoot/apps/stream_demo/run_experiment.sh"
if ($LASTEXITCODE -ne 0) { throw 'DeepStream experiment failed; inspect artifacts/b1_stream/*.log.' }
& $b1Python scripts/research/report_b1_stream.py
if ($LASTEXITCODE -ne 0) { throw 'Stream evidence is incomplete or mismatched.' }
