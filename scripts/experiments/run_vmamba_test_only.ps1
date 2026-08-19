[CmdletBinding()]
param(
    [string]$VmambaCheckpoint = "",
    [string]$OutputRoot = "",
    [int]$TileBatchSize = 2
)

<# Export only VMamba Test probability maps. Existing PNGs are skipped, so the
command resumes safely after interruption. It intentionally does not touch
Validation and does not launch any ensemble experiment. #>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$WeightRoot = Join-Path $Root "artifacts\checkpoints\final"
if (-not $VmambaCheckpoint) { $VmambaCheckpoint = Join-Path $WeightRoot "vmamba_best.pt" }
if (-not $OutputRoot) { $OutputRoot = Join-Path $Root "artifacts\experiments\decision\predictions\vmamba" }
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Exporter = Join-Path $Root "scripts\experiments\export_probability_cache.py"
$Dataset = Join-Path $Root "data\3cad_ani"
$Log = Join-Path $OutputRoot "vmamba_test_export.log"
foreach ($Required in @($Python, $Exporter, $Dataset, $VmambaCheckpoint)) {
    if (-not (Test-Path -LiteralPath $Required)) { throw "Required path not found: $Required" }
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START · VMamba TEST only" |
    Tee-Object -FilePath $Log -Append
$Arguments = @(
    $Exporter, "--model", "vmamba", "--checkpoint", $VmambaCheckpoint,
    "--dataset-root", $Dataset, "--output-root", $OutputRoot,
    "--splits", "test", "--tile-size", "512", "--stride", "256",
    "--tile-batch-size", "$TileBatchSize"
)
& $Python @Arguments 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "VMamba Test export failed with exit code $LASTEXITCODE" }
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] COMPLETE · VMamba TEST only" |
    Tee-Object -FilePath $Log -Append
