$ErrorActionPreference = "Stop"
$launcher = Join-Path $PSScriptRoot "scripts\verification\verify_submission.ps1"
& $launcher @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

