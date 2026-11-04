param([ValidateRange(10,300)][int]$Seconds=45,
      [ValidateSet('b1_stream_512_pose_v1','b1_stream_512','b1_stream')][string]$Artifact='b1_stream_512_pose_v1')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& (Join-Path $PSScriptRoot 'run_b1_conveyor.ps1') -Artifact $Artifact -NoBrowser
$slashPath = $PSScriptRoot -replace '\\','/'
$linuxRoot = (& wsl -d TTTN-DeepStream -u root -- wslpath -u $slashPath).Trim()
if ($LASTEXITCODE -ne 0 -or -not $linuxRoot) { throw 'Dedicated WSL TTTN-DeepStream is unavailable.' }
Start-Process 'http://127.0.0.1:8033/'
wsl -d TTTN-DeepStream -u root -- bash "$linuxRoot/apps/stream_demo/run_preview.sh" "$Seconds" "$Artifact"
if ($LASTEXITCODE -ne 0) { throw "TensorRT preview stopped with an error; inspect artifacts/$Artifact/preview_inference.log and preview_rtsp_server.log." }
