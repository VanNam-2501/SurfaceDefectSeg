[CmdletBinding()]
param([switch]$Runtime, [switch]$Inference, [switch]$TargetRebuiltEngine,
    [string]$Distro = 'TTTN-DeepStream', [int]$Port = 8033)
. (Join-Path $PSScriptRoot 'scripts\deployment\demo_common.ps1')
$python = Get-DemoPython
Push-Location $PSScriptRoot
try {
    if (Test-Path -LiteralPath 'release_manifest.json') {
        $manifestArgs = @('scripts/deployment/verify_package.py')
        if ($TargetRebuiltEngine) { $manifestArgs += '--target-engine' }
        & $python @manifestArgs
        if ($LASTEXITCODE -ne 0) { throw 'Release checksum verification failed.' }
    }
    $checkArgs = @('scripts/deployment/check_demo.py','--distro',$Distro,'--output','artifacts/verification/demo_preflight.json')
    if ($Runtime) { $checkArgs += '--runtime' }
    & $python @checkArgs
    if ($LASTEXITCODE -ne 0) { throw 'Demo preflight failed.' }
    if (Get-Command node -ErrorAction SilentlyContinue) {
        & node tests/test_conveyor_gate.cjs
        if ($LASTEXITCODE -ne 0) { throw 'Pixel gate regression failed.' }
    }
    if ($Inference) {
        & $python scripts/deployment/smoke_demo.py --url "http://127.0.0.1:$Port" --output artifacts/verification/demo_inference.json
        if ($LASTEXITCODE -ne 0) { throw 'Live TensorRT inference smoke failed.' }
    }
    Write-Host 'DeepStream demo verification: PASS' -ForegroundColor Green
}
finally { Pop-Location }
