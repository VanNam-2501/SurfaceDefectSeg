[CmdletBinding()]
param(
    [string]$DatasetRoot = "",
    [string]$UnetPredictionRoot = "",
    [string]$SegformerPredictionRoot = "",
    [string]$OutputDir = "",
    [double]$MaxFnr = 0.02,
    [switch]$RebuildCache
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$ProjectRoot = Join-Path $RepoRoot "src\threecad_segmentation"
if (-not $DatasetRoot) { $DatasetRoot = Join-Path $RepoRoot "data\3cad_ani" }
if (-not $UnetPredictionRoot) { $UnetPredictionRoot = Join-Path $RepoRoot "artifacts\experiments\decision\predictions\unet" }
if (-not $SegformerPredictionRoot) { $SegformerPredictionRoot = Join-Path $RepoRoot "artifacts\experiments\decision\predictions\segformer" }
if (-not $OutputDir) { $OutputDir = Join-Path $RepoRoot "artifacts\experiments\adaptive_component" }
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Script = Join-Path $ProjectRoot "adaptive_component_policy.py"
foreach ($RequiredPath in @(
    $Python,
    $Script,
    $DatasetRoot,
    (Join-Path $UnetPredictionRoot "val\probability"),
    (Join-Path $UnetPredictionRoot "test\probability"),
    (Join-Path $SegformerPredictionRoot "val\probability"),
    (Join-Path $SegformerPredictionRoot "test\probability")
)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) { throw "Required path not found: $RequiredPath" }
}

$Arguments = @(
    $Script,
    "--dataset-root", $DatasetRoot,
    "--prediction", "unet=$UnetPredictionRoot",
    "--prediction", "segformer=$SegformerPredictionRoot",
    "--output-dir", $OutputDir,
    "--max-fnr", ([string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0}", $MaxFnr))
)
if ($RebuildCache) { $Arguments += "--rebuild-cache" }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "Adaptive component policy failed with exit code $LASTEXITCODE" }
Write-Host "`nCompleted." -ForegroundColor Green
Write-Host "Comparison: $(Join-Path $OutputDir 'adaptive_model_comparison.csv')"
