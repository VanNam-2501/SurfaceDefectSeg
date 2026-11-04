$ErrorActionPreference = 'Stop'
$Python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $Python)) { $Python = 'python' }
Push-Location $PSScriptRoot
try {
    & $Python -m unittest discover -s tests -p test_inspection_math.py -v
    if ($LASTEXITCODE -ne 0) { throw 'Inspection mathematical core tests failed' }
}
finally { Pop-Location }
