[CmdletBinding()]
param([string]$Distro = 'TTTN-DeepStream')
. (Join-Path $PSScriptRoot 'scripts\deployment\demo_common.ps1')
$linuxRoot = Get-DemoWslRoot $Distro
& wsl.exe -d $Distro -u root -- bash "$linuxRoot/apps/stream_demo/rebuild_release.sh"
if ($LASTEXITCODE -ne 0) { throw 'Engine build/replay verification failed. Inspect artifacts/b1_stream_512_pose_v1/*.log.' }
Write-Host 'Target engine certified after a complete 300-frame file replay. Run .\verify_demo.ps1 -Runtime.' -ForegroundColor Green
