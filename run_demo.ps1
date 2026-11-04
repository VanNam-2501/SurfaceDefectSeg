$ErrorActionPreference = "Stop"
$launcher = Join-Path $PSScriptRoot "run_b1_conveyor.ps1"
& $launcher @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

