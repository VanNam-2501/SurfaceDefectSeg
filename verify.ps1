$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $Python)) { $Python = 'python' }
Push-Location $PSScriptRoot
try {
    & $Python -m pytest @args
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally { Pop-Location }

