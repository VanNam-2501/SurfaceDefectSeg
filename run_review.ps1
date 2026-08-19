$ErrorActionPreference = "Stop"
$launcher = Join-Path $PSScriptRoot "apps\dataset_review\start_review.ps1"
& $launcher @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

