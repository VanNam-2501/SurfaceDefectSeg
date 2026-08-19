[CmdletBinding()]
param(
    [ValidateSet("Export", "Calibrate", "Test", "All")]
    [string]$Action = "All",
    [string]$DatasetRoot = "",
    [string]$UnetCheckpoint = "",
    [string]$SegformerCheckpoint = "",
    [string]$VmambaCheckpoint = "",
    [string]$UnetPredictionRoot = "",
    [string]$SegformerPredictionRoot = "",
    [string]$VmambaPredictionRoot = "",
    [string]$Workspace = "",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [double]$FnrLimit = 0.02,
    [double]$MinDefectRecall = 0.80,
    [double]$ReviewCost = 0.25,
    [switch]$SaveTestMasks
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$ProjectRoot = Join-Path $RepoRoot "src\threecad_segmentation"
if (-not $DatasetRoot) {
    $DatasetRoot = Join-Path $RepoRoot "data\3cad_ani"
}
if (-not $Workspace) {
    $Workspace = Join-Path $RepoRoot "artifacts\experiments\decision"
}
$WeightRoot = Join-Path $RepoRoot "artifacts\checkpoints\final"
if (-not $UnetCheckpoint) { $UnetCheckpoint = Join-Path $WeightRoot "unet_best.pt" }
if (-not $SegformerCheckpoint) { $SegformerCheckpoint = Join-Path $WeightRoot "segformer_best.pt" }
if (-not $VmambaCheckpoint) { $VmambaCheckpoint = Join-Path $WeightRoot "vmamba_best.pt" }
if (-not $UnetPredictionRoot) {
    $UnetPredictionRoot = Join-Path $Workspace "predictions\unet"
}
if (-not $SegformerPredictionRoot) {
    $SegformerPredictionRoot = Join-Path $Workspace "predictions\segformer"
}
if (-not $VmambaPredictionRoot) {
    $VmambaPredictionRoot = Join-Path $Workspace "predictions\vmamba"
}

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Exporter = Join-Path $RepoRoot "apps\dataset_review\export_all_predictions.py"
$Calibrator = Join-Path $ProjectRoot "calibrate_decision_policy.py"
$Evaluator = Join-Path $RepoRoot "scripts\experiments\evaluate_decision_policy.py"
$PolicyOutput = Join-Path $Workspace "policy"
$PolicyPath = Join-Path $PolicyOutput "decision_policy.json"

foreach ($RequiredPath in @($Python, $Exporter, $Calibrator, $Evaluator, $DatasetRoot)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required path not found: $RequiredPath"
    }
}
New-Item -ItemType Directory -Path $Workspace -Force | Out-Null

$Models = [ordered]@{}
if ($UnetCheckpoint -and (Test-Path -LiteralPath $UnetCheckpoint)) {
    $Models["unet"] = @{ Checkpoint = $UnetCheckpoint; Predictions = $UnetPredictionRoot }
}
if ($SegformerCheckpoint -and (Test-Path -LiteralPath $SegformerCheckpoint)) {
    $Models["segformer"] = @{ Checkpoint = $SegformerCheckpoint; Predictions = $SegformerPredictionRoot }
}
if ($VmambaCheckpoint) {
    if (-not (Test-Path -LiteralPath $VmambaCheckpoint)) {
        throw "VMamba checkpoint not found: $VmambaCheckpoint"
    }
    $Models["vmamba"] = @{ Checkpoint = $VmambaCheckpoint; Predictions = $VmambaPredictionRoot }
}
if ($Models.Count -eq 0) {
    throw "No checkpoint is available."
}

function Invoke-Python {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

function Test-PredictionCacheComplete {
    param([string]$PredictionRoot)
    foreach ($Split in @("val", "test")) {
        $Csv = Join-Path $DatasetRoot "dataset_audit\splits\$Split.csv"
        $ProbabilityDirectory = Join-Path $PredictionRoot "$Split\probability"
        if (-not (Test-Path -LiteralPath $Csv) -or -not (Test-Path -LiteralPath $ProbabilityDirectory)) {
            return $false
        }
        $Expected = (Import-Csv -LiteralPath $Csv).Count
        $Actual = (Get-ChildItem -LiteralPath $ProbabilityDirectory -Filter "*.png" -File).Count
        if ($Actual -lt $Expected) {
            return $false
        }
    }
    return $true
}

function Export-Probabilities {
    foreach ($Entry in $Models.GetEnumerator()) {
        $Name = [string]$Entry.Key
        $Checkpoint = [string]$Entry.Value.Checkpoint
        $PredictionRoot = [string]$Entry.Value.Predictions
        Write-Host "`n=== EXPORT $($Name.ToUpperInvariant()) · VALIDATION + TEST ===" -ForegroundColor Cyan
        if (Test-PredictionCacheComplete -PredictionRoot $PredictionRoot) {
            Write-Host "Cache is complete; skipping inference: $PredictionRoot" -ForegroundColor DarkGreen
            continue
        }
        Invoke-Python @(
            $Exporter,
            "--model", $Name,
            "--checkpoint", $Checkpoint,
            "--dataset-root", $DatasetRoot,
            "--splits", "val", "test",
            "--output-root", $PredictionRoot,
            "--threshold", "0.5",
            "--device", $Device,
            "--model-code-dir", $ProjectRoot
        )
    }
}

function Get-PredictionArguments {
    $Arguments = [System.Collections.Generic.List[string]]::new()
    foreach ($Entry in $Models.GetEnumerator()) {
        $Root = [string]$Entry.Value.Predictions
        if (-not (Test-Path -LiteralPath (Join-Path $Root "val\probability"))) {
            throw "Validation probabilities missing for $($Entry.Key): $Root. Run -Action Export first."
        }
        $Arguments.Add("--prediction")
        $Arguments.Add("$($Entry.Key)=$Root")
    }
    return $Arguments.ToArray()
}

function Calibrate-Policy {
    Write-Host "`n=== CALIBRATE POLICY · VALIDATION ONLY ===" -ForegroundColor Yellow
    $Arguments = [System.Collections.Generic.List[string]]::new()
    foreach ($Item in @(
        $Calibrator,
        "--dataset-root", $DatasetRoot,
        "--output-dir", $PolicyOutput,
        "--fnr-limit", ([string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0}", $FnrLimit)),
        "--min-defect-recall", ([string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0}", $MinDefectRecall)),
        "--review-cost", ([string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0}", $ReviewCost))
    )) {
        $Arguments.Add([string]$Item)
    }
    foreach ($Item in (Get-PredictionArguments)) {
        $Arguments.Add($Item)
    }
    Invoke-Python $Arguments.ToArray()
}

function Test-FrozenPolicy {
    if (-not (Test-Path -LiteralPath $PolicyPath)) {
        throw "Frozen policy not found: $PolicyPath. Run -Action Calibrate first."
    }
    Write-Host "`n=== TEST · FROZEN VALIDATION POLICY ===" -ForegroundColor Green
    $Arguments = [System.Collections.Generic.List[string]]::new()
    foreach ($Item in @(
        $Evaluator,
        "--dataset-root", $DatasetRoot,
        "--split", "test",
        "--policy", $PolicyPath,
        "--output-dir", (Join-Path $Workspace "test")
    )) {
        $Arguments.Add([string]$Item)
    }
    foreach ($Item in (Get-PredictionArguments)) {
        $Arguments.Add($Item)
    }
    if ($SaveTestMasks) {
        $Arguments.Add("--save-masks")
    }
    Invoke-Python $Arguments.ToArray()
}

switch ($Action) {
    "Export" { Export-Probabilities }
    "Calibrate" { Calibrate-Policy }
    "Test" { Test-FrozenPolicy }
    "All" {
        Export-Probabilities
        Calibrate-Policy
        Test-FrozenPolicy
    }
}

Write-Host "`nCompleted." -ForegroundColor Green
Write-Host "Policy: $PolicyPath"
Write-Host "Test metrics: $(Join-Path $Workspace 'test\decision_metrics.json')"
