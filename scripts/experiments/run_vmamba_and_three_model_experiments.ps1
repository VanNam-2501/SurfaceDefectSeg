[CmdletBinding()]
param(
    [string]$VmambaCheckpoint = "",
    [string]$VmambaPredictionRoot = "",
    [string]$ReportDir = "",
    [int]$TileBatchSize = 2
)

<#
Runs the complete local, resumable 3-model evaluation:
1. Export VMamba probability maps for Validation and Test.
2. Run every standalone / spatial / learned / pair-hybrid experiment.

The exporter skips existing PNGs, therefore the script is safe to resume after
an interruption. TileBatchSize=2 was measured as safe on the local RTX 3050.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$WeightRoot = Join-Path $Root "artifacts\checkpoints\final"
if (-not $VmambaCheckpoint) { $VmambaCheckpoint = Join-Path $WeightRoot "vmamba_best.pt" }
if (-not $VmambaPredictionRoot) { $VmambaPredictionRoot = Join-Path $Root "artifacts\experiments\decision\predictions\vmamba" }
if (-not $ReportDir) { $ReportDir = Join-Path $Root "artifacts\experiments\decision\three_model_experiment_report" }
$Dataset = Join-Path $Root "data\3cad_ani"
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Exporter = Join-Path $Root "scripts\experiments\export_probability_cache.py"
$Pipeline = Join-Path $Root "scripts\experiments\run_three_model_experiments.ps1"
$Log = Join-Path $ReportDir "run_local_pipeline.log"

foreach ($Required in @($Python, $Exporter, $Pipeline, $Dataset, $VmambaCheckpoint)) {
    if (-not (Test-Path -LiteralPath $Required)) { throw "Required path not found: $Required" }
}
New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null

function Write-Stage([string]$Message) {
    "`n[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message" |
        Tee-Object -FilePath $Log -Append
}

Write-Stage "START · VMamba probability export (Validation + Test)"
$ExportArguments = @(
    $Exporter,
    "--model", "vmamba",
    "--checkpoint", $VmambaCheckpoint,
    "--dataset-root", $Dataset,
    "--output-root", $VmambaPredictionRoot,
    "--splits", "val", "test",
    "--tile-size", "512", "--stride", "256", "--tile-batch-size", "$TileBatchSize"
)
& $Python @ExportArguments 2>&1 |
    Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "VMamba export failed with exit code $LASTEXITCODE" }

Write-Stage "VMamba map export complete · begin all three-model experiments"
& $Pipeline -Action All -VmambaPredictionRoot $VmambaPredictionRoot -ReportDir $ReportDir 2>&1 |
    Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "Three-model experiment pipeline failed with exit code $LASTEXITCODE" }

Write-Stage "COMPLETE · report tables are ready"
