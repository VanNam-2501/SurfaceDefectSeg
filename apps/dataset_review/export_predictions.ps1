param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("unet", "segformer", "vmamba")]
    [string]$Model,
    [string[]]$Splits = @("train", "val", "test"),
    [string]$DatasetRoot = "",
    [string]$ResultsRoot = "",
    [string]$Checkpoint = "",
    [string]$OutputRoot = "",
    [int]$TileBatchSize = 1,
    [int]$MaxImages = 0,
    [switch]$Overwrite,
    [switch]$NoAmp
)

$toolDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $toolDir "..\..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python runtime not found: $python"
}
if (-not $DatasetRoot) {
    $DatasetRoot = Join-Path $repoRoot "data\3cad_ani"
}

$runFolder = switch ($Model) {
    "unet" { "unet_r18" }
    "segformer" { "segformer_b0" }
    "vmamba" { "vmamba_t" }
}
if (-not $Checkpoint) {
    if ($ResultsRoot) {
        $Checkpoint = Join-Path $ResultsRoot "$runFolder\main_seed42\checkpoints\best.pt"
    }
    else {
        $Checkpoint = Join-Path $repoRoot "artifacts\checkpoints\final\$($Model)_best.pt"
    }
}
if (-not (Test-Path -LiteralPath $Checkpoint)) {
    throw "Checkpoint not found: $Checkpoint"
}
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $repoRoot "artifacts\experiments\decision\predictions\$Model"
}

if ($Model -eq "segformer") {
    & $python -c "import transformers" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "SegFormer requires transformers. Run: .\.venv\Scripts\python.exe -m pip install transformers==5.0.0"
    }
}

$exportArgs = @(
    (Join-Path $toolDir "export_all_predictions.py"),
    "--model", $Model,
    "--checkpoint", $Checkpoint,
    "--dataset-root", $DatasetRoot,
    "--output-root", $OutputRoot,
    "--tile-batch-size", $TileBatchSize,
    "--splits"
)
$exportArgs += $Splits
if ($MaxImages -gt 0) {
    $exportArgs += @("--max-images", $MaxImages)
}
if ($Overwrite) {
    $exportArgs += "--overwrite"
}
if ($NoAmp) {
    $exportArgs += "--no-amp"
}

Write-Host "Exporting predictions for $Model on: $($Splits -join ', ')"
Write-Host "Existing prediction PNGs will be skipped, so this command can be resumed."
& $python @exportArgs
