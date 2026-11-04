[CmdletBinding()]
param(
    [ValidateSet('b1_stream_512_pose_v1','b1_stream_512','b1_stream')][string]$Artifact = 'b1_stream_512_pose_v1',
    [string]$Distro = 'TTTN-DeepStream',
    [ValidateRange(1024,65535)][int]$Port = 8033,
    [switch]$NoBrowser,
    [switch]$Restart
)
. (Join-Path $PSScriptRoot 'scripts\deployment\demo_common.ps1')
$python = Get-DemoPython
$linuxRoot = Get-DemoWslRoot $Distro
$preflightJson = & $python (Join-Path $PSScriptRoot 'scripts\deployment\check_demo.py') --artifact $Artifact --runtime --distro $Distro
Write-Output $preflightJson
if ($LASTEXITCODE -ne 0) { throw 'Demo preflight failed. See docs/DEEPSTREAM_HANDOVER.md.' }
$expectedContract = (($preflightJson -join "`n") | ConvertFrom-Json).identity.contract_sha256
$out = Join-Path $PSScriptRoot "artifacts\$Artifact"
$preview = Join-Path $out 'preview'
$scriptPath = Join-Path $PSScriptRoot 'apps\stream_demo\preview_web.py'
$healthUrl = "http://127.0.0.1:$Port/health"
$health = $null
try { $health = Invoke-RestMethod $healthUrl -TimeoutSec 2 } catch {}
function Test-ViewerIdentity($value) {
    return ($null -ne $value -and $value.PSObject.Properties.Name -contains 'project_root' -and
        $value.service -eq 'TensorRT DeepStream preview' -and
        $value.project_root -eq $PSScriptRoot -and $value.artifact -eq $Artifact -and
        $value.image_inference -and $value.api_version -ge 15 -and
        $value.contract_sha256 -eq $expectedContract)
}
$listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique)
if ($Restart -or -not (Test-ViewerIdentity $health)) {
    foreach ($listenerId in $listeners) {
        $listener = Get-CimInstance Win32_Process -Filter "ProcessId = $listenerId" -ErrorAction SilentlyContinue
        if ($listener -and $listener.CommandLine.Contains($scriptPath)) {
            try { Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$Port/api/video/stop" -ContentType 'application/json' -Body '{}' -TimeoutSec 15 | Out-Null } catch {}
            Stop-Process -Id $listener.ProcessId -Force
        } else {
            throw "Port $Port is used by another service/checkout. Stop it or use -Port 8035."
        }
    }
    New-Item -ItemType Directory -Force -Path $preview | Out-Null
    Start-Process -FilePath $python -ArgumentList @(('"' + $scriptPath + '"'),'--host','127.0.0.1',
        '--port',"$Port",'--artifact',$Artifact,'--wsl-distro',$Distro,'--wsl-root',('"' + $linuxRoot + '"')) `
        -WorkingDirectory $PSScriptRoot -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $out 'preview_web.stdout.log') `
        -RedirectStandardError (Join-Path $out 'preview_web.stderr.log') | Out-Null
}
$deadline = (Get-Date).AddSeconds(50)
do {
    try { $health = Invoke-RestMethod $healthUrl -TimeoutSec 2 } catch { $health = $null }
    if ((Test-ViewerIdentity $health) -and $health.runtime_verified -and
        $health.runtime.contract_sha256 -eq $health.contract_sha256) { break }
    Start-Sleep -Milliseconds 500
} while ((Get-Date) -lt $deadline)
if (-not (Test-ViewerIdentity $health) -or -not $health.runtime_verified -or
    $health.runtime.contract_sha256 -ne $health.contract_sha256) {
    throw "TensorRT worker is not ready. Inspect $out\resident_queue\launcher.log and preview_web.stderr.log."
}
if (-not $NoBrowser) { Start-Process "http://127.0.0.1:$Port/" }
Write-Host "TensorRT ready: http://127.0.0.1:$Port/ | $Artifact | $($health.model_size) px" -ForegroundColor Green
