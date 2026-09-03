[CmdletBinding()]
param(
    [switch]$IncludeWeb
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$MlRoot = Join-Path $Root "src\threecad_segmentation"
$Dataset = Join-Path $Root "data\3cad_ani"
$Required = @(
    $Python,
    (Join-Path $Root "scripts\verification\check_protocol.py"),
    (Join-Path $Root "tests\ml\test_decision_policy.py"),
    (Join-Path $Root "tests\ml\test_web_demo_policy.py"),
    (Join-Path $Root "tests\ml\test_training_state.py"),
    (Join-Path $Root "apps\dataset_review\tests\test_review_tool.py"),
    (Join-Path $Root "artifacts\reports\final\visualizations\index.html"),
    (Join-Path $Root "artifacts\checkpoints\final\unet_best.pt"),
    (Join-Path $Root "artifacts\checkpoints\final\segformer_best.pt"),
    (Join-Path $Root "artifacts\checkpoints\final\vmamba_best.pt")
)

foreach ($Path in $Required) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required artifact is missing: $Path"
    }
}

function Assert-LastExitCode([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

$PreviousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = if ($PreviousPythonPath) { "$MlRoot;$PreviousPythonPath" } else { $MlRoot }
Push-Location $Root
try {
    & $Python ".\scripts\verification\check_protocol.py" --dataset-root $Dataset
    Assert-LastExitCode "Dataset protocol check"
    & $Python ".\tests\ml\test_decision_policy.py"
    Assert-LastExitCode "Decision policy tests"
    & $Python ".\tests\ml\test_web_demo_policy.py"
    Assert-LastExitCode "Web policy tests"
    & $Python ".\tests\ml\test_training_state.py"
    Assert-LastExitCode "Training state tests"
    & $Python -m unittest discover -s ".\apps\dataset_review\tests" -v
    Assert-LastExitCode "Dataset Review Studio tests"
}
finally {
    Pop-Location
    $env:PYTHONPATH = $PreviousPythonPath
}

if ($IncludeWeb) {
    Push-Location (Join-Path $Root "apps\web_demo")
    try {
        & npm.cmd test
        Assert-LastExitCode "Web demo tests"
    }
    finally {
        Pop-Location
    }
}

Write-Host "Submission verification: PASS" -ForegroundColor Green
