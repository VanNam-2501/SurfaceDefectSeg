[CmdletBinding()]
param(
    [ValidateSet('b1_stream_512_pose_v1','b1_stream_512','b1_stream')][string]$Artifact = 'b1_stream_512_pose_v1',
    [string]$Distro = 'TTTN-DeepStream',
    [ValidateRange(1024,65535)][int]$Port = 8033
)
. (Join-Path $PSScriptRoot 'scripts\deployment\demo_common.ps1')
$scriptPath = Join-Path $PSScriptRoot 'apps\stream_demo\preview_web.py'
$listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique)
foreach ($listenerId in $listeners) {
    $listener = Get-CimInstance Win32_Process -Filter "ProcessId = $listenerId" -ErrorAction SilentlyContinue
    if (-not $listener -or -not $listener.CommandLine.Contains($scriptPath)) {
        throw "Refusing to stop another service on port $Port."
    }
    try { Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$Port/api/video/stop" -ContentType 'application/json' -Body '{}' -TimeoutSec 15 | Out-Null } catch {}
    Stop-Process -Id $listener.ProcessId -Force
}
$linuxRoot = Get-DemoWslRoot $Distro
& wsl.exe -d $Distro -u root -- bash "$linuxRoot/apps/stream_demo/stop_resident.sh" $Artifact
if ($LASTEXITCODE -ne 0) { throw 'Could not stop the dedicated TensorRT container.' }
Write-Host 'Demo stopped; inputs, logs and engine retained.' -ForegroundColor Green
