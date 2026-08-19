$ErrorActionPreference = "Stop"
$launcher = Join-Path $PSScriptRoot "scripts\demo\start_decision_demo.ps1"
& $launcher @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

