$ErrorActionPreference = 'Stop'
$Python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $Python)) { $Python = 'python' }
Push-Location $PSScriptRoot
try {
    & $Python -m unittest discover -s tests -p test_straw_thesis.py -v
    if ($LASTEXITCODE -ne 0) { throw 'Thesis protocol tests failed' }
    & $Python -m unittest discover -s tests -p test_decide.py -v
    if ($LASTEXITCODE -ne 0) { throw 'Existing decision tests failed' }
    Write-Host 'Straw protocol and decision tests: PASS' -ForegroundColor Green
}
finally { Pop-Location }
