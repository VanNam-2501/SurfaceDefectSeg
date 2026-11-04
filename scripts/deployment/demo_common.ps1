Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$DemoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

function Get-DemoPython {
    foreach ($relative in @('.venv-demo\Scripts\python.exe', '.venv\Scripts\python.exe')) {
        $candidate = Join-Path $DemoRoot $relative
        if (Test-Path -LiteralPath $candidate) {
            & $candidate -c 'import numpy, PIL' 2>$null
            if ($LASTEXITCODE -eq 0) { return $candidate }
        }
    }
    throw 'Python demo is unavailable. Run .\setup_demo.ps1 first (or specify -Python with your Python 3.12 executable).'
}

function Get-DemoWslRoot([string]$Distro) {
    $converted = & wsl.exe -d $Distro -u root -- wslpath -u ($DemoRoot -replace '\\', '/')
    if ($LASTEXITCODE -ne 0 -or -not $converted) {
        throw "Cannot access WSL distro $Distro. See docs/DEEPSTREAM_HANDOVER.md."
    }
    return ($converted | Select-Object -Last 1).Trim()
}
