[CmdletBinding()]
param([string]$Python)
$ErrorActionPreference = 'Stop'
$previousTemp = $env:TEMP
$previousTmp = $env:TMP
$previousUvCache = $env:UV_CACHE_DIR
Push-Location $PSScriptRoot
try {
    $setupTemp = Join-Path $PSScriptRoot 'artifacts\setup\tmp'
    New-Item -ItemType Directory -Force $setupTemp | Out-Null
    $env:TEMP = $setupTemp
    $env:TMP = $setupTemp
    if (-not $Python) {
        if (Get-Command uv -ErrorAction SilentlyContinue) {
            $env:UV_CACHE_DIR = Join-Path $PSScriptRoot 'artifacts\setup\uv_cache'
            $found = & uv python find 3.12 --no-python-downloads 2>$null
            if ($LASTEXITCODE -eq 0 -and $found) { $Python = ($found | Select-Object -Last 1).Trim() }
        }
        if (-not $Python -and (Get-Command py -ErrorAction SilentlyContinue)) {
            $found = & py -3.12 -c 'import sys; print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0 -and $found) { $Python = ($found | Select-Object -Last 1).Trim() }
        }
        if (-not $Python -and (Get-Command python -ErrorAction SilentlyContinue)) { $Python = 'python' }
    }
    if (-not $Python) { throw 'Install Python 3.12, then run .\setup_demo.ps1 -Python C:\path\to\python.exe' }
    & $Python -c 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 is required"'
    if ($LASTEXITCODE -ne 0) { throw 'A working Python 3.12 interpreter is required.' }
    $venvPython = Join-Path $PSScriptRoot '.venv-demo\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $venvPython)) {
        & $Python -m venv (Join-Path $PSScriptRoot '.venv-demo')
        if ($LASTEXITCODE -ne 0) { throw 'Cannot create .venv-demo.' }
    }
    & $venvPython -m pip --version *> $null
    if ($LASTEXITCODE -ne 0) {
        & $venvPython -m ensurepip --upgrade
        if ($LASTEXITCODE -ne 0) { throw 'Cannot bootstrap pip in .venv-demo.' }
    }
    $installArgs = @('-m','pip','install','--disable-pip-version-check','-r','requirements/deepstream-demo.txt')
    if (Test-Path -LiteralPath 'runtime/wheels/demo') {
        $installArgs += @('--no-index','--find-links','runtime/wheels/demo')
    }
    & $venvPython @installArgs
    if ($LASTEXITCODE -ne 0) { throw 'Demo dependency installation failed.' }
    & $venvPython -c 'import numpy, PIL; print("Demo host ready:", numpy.__version__, PIL.__version__)'
    if ($LASTEXITCODE -ne 0) { throw 'Demo dependency import failed.' }
    Write-Host 'Next: .\verify_demo.ps1 -Runtime, then .\run_b1_conveyor.ps1' -ForegroundColor Green
}
finally {
    Pop-Location
    $env:TEMP = $previousTemp
    $env:TMP = $previousTmp
    $env:UV_CACHE_DIR = $previousUvCache
}
