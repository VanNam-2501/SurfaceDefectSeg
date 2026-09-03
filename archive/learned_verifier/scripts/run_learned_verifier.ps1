[CmdletBinding()]
param(
    [string]$DatasetRoot = "",
    [string]$UnetPredictionRoot = "",
    [string]$SegformerPredictionRoot = "",
    [string]$OutputDir = "",
    [string]$BaseValDecisions = "",
    [string]$BaseTestDecisions = "",
    [double]$MaxFnr = 0.02,
    [double]$MaxDefectFpr = 0.10,
    [double]$FnrSafetyMargin = 0.005,
    [int]$Folds = 5,
    [int]$FeatureSize = 256,
    [switch]$RebuildFeatures
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$ProjectRoot = Join-Path $RepoRoot "src\threecad_segmentation"
if (-not $DatasetRoot) {
    $DatasetRoot = Join-Path $RepoRoot "data\3cad_ani"
}
if (-not $UnetPredictionRoot) {
    $UnetPredictionRoot = Join-Path $RepoRoot "artifacts\experiments\decision\predictions\unet"
}
if (-not $SegformerPredictionRoot) {
    $SegformerPredictionRoot = Join-Path $RepoRoot "artifacts\experiments\decision\predictions\segformer"
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $RepoRoot "artifacts\experiments\learned_verifier"
}
if (-not $BaseValDecisions) {
    $BaseValDecisions = Join-Path $RepoRoot "artifacts\experiments\decision\val\per_image_decisions.csv"
}
if (-not $BaseTestDecisions) {
    $BaseTestDecisions = Join-Path $RepoRoot "artifacts\experiments\decision\test\per_image_decisions.csv"
}

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Script = Join-Path $ProjectRoot "learned_decision_verifier.py"
foreach ($RequiredPath in @(
    $Python,
    $Script,
    $DatasetRoot,
    (Join-Path $UnetPredictionRoot "val\probability"),
    (Join-Path $UnetPredictionRoot "test\probability"),
    (Join-Path $SegformerPredictionRoot "val\probability"),
    (Join-Path $SegformerPredictionRoot "test\probability"),
    $BaseValDecisions,
    $BaseTestDecisions
)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required path not found: $RequiredPath"
    }
}

$Arguments = @(
    $Script,
    "--dataset-root", $DatasetRoot,
    "--prediction", "unet=$UnetPredictionRoot",
    "--prediction", "segformer=$SegformerPredictionRoot",
    "--output-dir", $OutputDir,
    "--base-val-decisions", $BaseValDecisions,
    "--base-test-decisions", $BaseTestDecisions,
    "--max-fnr", ([string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0}", $MaxFnr)),
    "--max-defect-fpr", ([string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0}", $MaxDefectFpr)),
    "--fnr-safety-margin", ([string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0}", $FnrSafetyMargin)),
    "--folds", "$Folds",
    "--feature-size", "$FeatureSize"
)
if ($RebuildFeatures) {
    $Arguments += "--rebuild-features"
}

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Learned verifier failed with exit code $LASTEXITCODE"
}

Write-Host "`nCompleted." -ForegroundColor Green
Write-Host "Comparison: $(Join-Path $OutputDir 'model_comparison.csv')"
Write-Host "Per image: $(Join-Path $OutputDir 'per_image_predictions.csv')"
Write-Host "Policy: $(Join-Path $OutputDir 'learned_policy.json')"
