[CmdletBinding()]
param(
    [ValidateSet("Check", "All")]
    [string]$Action = "All",
    [string]$DatasetRoot = "",
    [string]$UnetPredictionRoot = "",
    [string]$SegformerPredictionRoot = "",
    [string]$VmambaPredictionRoot = "",
    [string]$ReportDir = "",
    [double]$MaxFnr = 0.02,
    [double]$MaxDefectFpr = 0.10,
    [double]$FnrSafetyMargin = 0.005,
    [int]$Folds = 5,
    [int]$FeatureSize = 256,
    [switch]$RebuildCaches
)

<##
.SYNOPSIS
Creates a report-ready, fair three-model evaluation package.

.DESCRIPTION
Every threshold/rule is fitted on Validation only.  The Test split is never
used to choose a policy.  The final package contains:
  * single-model adaptive component rules (U-Net, SegFormer, VMamba);
  * spatial PASS/REVIEW/DEFECT policies for each model, all pairs, and all 3;
  * a learned verifier for each model and learned three-model fusion;
  * fully automatic hybrid policies for every pair of models;
  * consolidated CSV tables for the thesis/report.

.EXAMPLE
.\run_three_model_experiments.ps1

.EXAMPLE
.\scripts\experiments\run_three_model_experiments.ps1 -VmambaPredictionRoot 'E:\Project\TTTN\artifacts\experiments\decision\predictions\vmamba'
##>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$ProjectRoot = Join-Path $RepoRoot "src\threecad_segmentation"
if (-not $DatasetRoot) { $DatasetRoot = Join-Path $RepoRoot "data\3cad_ani" }
if (-not $UnetPredictionRoot) { $UnetPredictionRoot = Join-Path $RepoRoot "artifacts\experiments\decision\predictions\unet" }
if (-not $SegformerPredictionRoot) { $SegformerPredictionRoot = Join-Path $RepoRoot "artifacts\experiments\decision\predictions\segformer" }
if (-not $VmambaPredictionRoot) { $VmambaPredictionRoot = Join-Path $RepoRoot "artifacts\experiments\decision\predictions\vmamba" }
if (-not $ReportDir) { $ReportDir = Join-Path $RepoRoot "artifacts\experiments\decision\three_model_experiment_report" }

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Calibrator = Join-Path $ProjectRoot "calibrate_decision_policy.py"
$Evaluator = Join-Path $RepoRoot "scripts\experiments\evaluate_decision_policy.py"
$Adaptive = Join-Path $ProjectRoot "adaptive_component_policy.py"
$Learned = Join-Path $ProjectRoot "learned_decision_verifier.py"
$Compiler = Join-Path $RepoRoot "scripts\reporting\compile_three_model_report.py"
$Invariant = [Globalization.CultureInfo]::InvariantCulture

$ModelRoots = [ordered]@{
    unet = $UnetPredictionRoot
    segformer = $SegformerPredictionRoot
    vmamba = $VmambaPredictionRoot
}

foreach ($RequiredPath in @($Python, $DatasetRoot, $Calibrator, $Evaluator, $Adaptive, $Learned, $Compiler)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required path not found: $RequiredPath"
    }
}

function Test-PredictionCacheComplete {
    param([string]$Model, [string]$Root)
    foreach ($Split in @("val", "test")) {
        $Csv = Join-Path $DatasetRoot "dataset_audit\splits\$Split.csv"
        $ProbabilityDirectory = Join-Path $Root "$Split\probability"
        if (-not (Test-Path -LiteralPath $ProbabilityDirectory)) {
            Write-Host "[$Model] Missing: $ProbabilityDirectory" -ForegroundColor Red
            return $false
        }
        $Expected = (Import-Csv -LiteralPath $Csv).Count
        $Actual = (Get-ChildItem -LiteralPath $ProbabilityDirectory -Filter "*.png" -File).Count
        if ($Actual -lt $Expected) {
            Write-Host "[$Model/$Split] $Actual/$Expected probability maps" -ForegroundColor Red
            return $false
        }
        Write-Host "[$Model/$Split] $Actual/$Expected probability maps" -ForegroundColor DarkGreen
    }
    return $true
}

function Invoke-Python {
    param([System.Collections.Generic.List[string]]$Arguments)
    & $Python @($Arguments.ToArray())
    if ($LASTEXITCODE -ne 0) { throw "Python failed with exit code $LASTEXITCODE" }
}

function Add-Predictions {
    param(
        [System.Collections.Generic.List[string]]$Arguments,
        [string[]]$Models
    )
    foreach ($Model in $Models) {
        $Arguments.Add("--prediction")
        $Arguments.Add("$Model=$($ModelRoots[$Model])")
    }
}

$CacheOk = $true
foreach ($Entry in $ModelRoots.GetEnumerator()) {
    if (-not (Test-PredictionCacheComplete -Model $Entry.Key -Root $Entry.Value)) { $CacheOk = $false }
}
if (-not $CacheOk) {
    throw @"
VMamba probabilities are not ready yet.  Copy the Kaggle export so this path exists:
  $VmambaPredictionRoot\val\probability
  $VmambaPredictionRoot\test\probability

U-Net and SegFormer may also be re-exported with apps\dataset_review\export_all_predictions.py if needed.
"@
}
if ($Action -eq "Check") {
    Write-Host "All 3 prediction caches are complete. You can run -Action All." -ForegroundColor Green
    exit 0
}

New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null
$fnrText = $MaxFnr.ToString("0.####", $Invariant)
$defectFprText = $MaxDefectFpr.ToString("0.####", $Invariant)
$marginText = $FnrSafetyMargin.ToString("0.####", $Invariant)

function Invoke-SpatialExperiment {
    param([string]$Id, [string[]]$Models)
    $Root = Join-Path $ReportDir "spatial\$Id"
    $PolicyDirectory = Join-Path $Root "policy"
    $PolicyPath = Join-Path $PolicyDirectory "decision_policy.json"
    New-Item -ItemType Directory -Force -Path $Root | Out-Null
    Write-Host "`n=== SPATIAL POLICY: $Id · VALIDATION ONLY ===" -ForegroundColor Cyan
    $CalibrateArguments = [System.Collections.Generic.List[string]]::new()
    foreach ($Item in @(
        $Calibrator, "--dataset-root", $DatasetRoot, "--output-dir", $PolicyDirectory,
        "--fnr-limit", $fnrText, "--min-defect-recall", "0.80", "--review-cost", "0.25"
    )) { $CalibrateArguments.Add([string]$Item) }
    Add-Predictions -Arguments $CalibrateArguments -Models $Models
    Invoke-Python -Arguments $CalibrateArguments

    foreach ($Split in @("val", "test")) {
        Write-Host "=== SPATIAL $Id · $($Split.ToUpperInvariant()) · FROZEN POLICY ===" -ForegroundColor Green
        $EvaluateArguments = [System.Collections.Generic.List[string]]::new()
        foreach ($Item in @(
            $Evaluator, "--dataset-root", $DatasetRoot, "--split", $Split,
            "--policy", $PolicyPath, "--output-dir", (Join-Path $Root $Split)
        )) { $EvaluateArguments.Add([string]$Item) }
        Add-Predictions -Arguments $EvaluateArguments -Models $Models
        Invoke-Python -Arguments $EvaluateArguments
    }
}

$Experiments = [ordered]@{
    unet = @("unet")
    segformer = @("segformer")
    vmamba = @("vmamba")
    unet_segformer = @("unet", "segformer")
    unet_vmamba = @("unet", "vmamba")
    segformer_vmamba = @("segformer", "vmamba")
    unet_segformer_vmamba = @("unet", "segformer", "vmamba")
}
foreach ($Entry in $Experiments.GetEnumerator()) {
    Invoke-SpatialExperiment -Id $Entry.Key -Models $Entry.Value
}

Write-Host "`n=== ADAPTIVE COMPONENT RULES · EACH MODEL ===" -ForegroundColor Yellow
$AdaptiveOutput = Join-Path $ReportDir "adaptive_single"
$AdaptiveArguments = [System.Collections.Generic.List[string]]::new()
foreach ($Item in @($Adaptive, "--dataset-root", $DatasetRoot, "--output-dir", $AdaptiveOutput, "--max-fnr", $fnrText)) {
    $AdaptiveArguments.Add([string]$Item)
}
Add-Predictions -Arguments $AdaptiveArguments -Models @("unet", "segformer", "vmamba")
if ($RebuildCaches) { $AdaptiveArguments.Add("--rebuild-cache") }
Invoke-Python -Arguments $AdaptiveArguments

Write-Host "`n=== LEARNED VERIFIER · ALL THREE MODELS ===" -ForegroundColor Yellow
$LearnedAllOutput = Join-Path $ReportDir "learned_all3"
$LearnedAllArguments = [System.Collections.Generic.List[string]]::new()
foreach ($Item in @(
    $Learned, "--dataset-root", $DatasetRoot, "--output-dir", $LearnedAllOutput,
    "--max-fnr", $fnrText, "--max-defect-fpr", $defectFprText,
    "--folds", "$Folds", "--feature-size", "$FeatureSize"
)) { $LearnedAllArguments.Add([string]$Item) }
Add-Predictions -Arguments $LearnedAllArguments -Models @("unet", "segformer", "vmamba")
if ($RebuildCaches) { $LearnedAllArguments.Add("--rebuild-features") }
Invoke-Python -Arguments $LearnedAllArguments

$Pairs = [ordered]@{
    unet_segformer = @("unet", "segformer")
    unet_vmamba = @("unet", "vmamba")
    segformer_vmamba = @("segformer", "vmamba")
}
foreach ($Entry in $Pairs.GetEnumerator()) {
    $Id = [string]$Entry.Key
    $Models = [string[]]$Entry.Value
    Write-Host "`n=== FULLY AUTOMATIC HYBRID: $Id ===" -ForegroundColor Magenta
    $HybridOutput = Join-Path $ReportDir "hybrid_pairs\$Id"
    $SpatialRoot = Join-Path $ReportDir "spatial\$Id"
    $HybridArguments = [System.Collections.Generic.List[string]]::new()
    foreach ($Item in @(
        $Learned, "--dataset-root", $DatasetRoot, "--output-dir", $HybridOutput,
        "--base-val-decisions", (Join-Path $SpatialRoot "val\per_image_decisions.csv"),
        "--base-test-decisions", (Join-Path $SpatialRoot "test\per_image_decisions.csv"),
        "--max-fnr", $fnrText, "--max-defect-fpr", $defectFprText,
        "--fnr-safety-margin", $marginText, "--folds", "$Folds", "--feature-size", "$FeatureSize"
    )) { $HybridArguments.Add([string]$Item) }
    Add-Predictions -Arguments $HybridArguments -Models $Models
    if ($RebuildCaches) { $HybridArguments.Add("--rebuild-features") }
    Invoke-Python -Arguments $HybridArguments
}

Write-Host "`n=== COMPILE REPORT TABLES ===" -ForegroundColor Green
$CompileArguments = [System.Collections.Generic.List[string]]::new()
foreach ($Item in @($Compiler, "--report-dir", $ReportDir, "--dataset-root", $DatasetRoot)) {
    $CompileArguments.Add([string]$Item)
}
Invoke-Python -Arguments $CompileArguments

Write-Host "`nDONE. Open these report-ready files:" -ForegroundColor Green
Write-Host "  $(Join-Path $ReportDir 'tables\01_master_test_comparison.csv')"
Write-Host "  $(Join-Path $ReportDir 'tables\02_fully_automatic_comparison.csv')"
Write-Host "  $(Join-Path $ReportDir 'tables\03_defect_group_recall.csv')"
