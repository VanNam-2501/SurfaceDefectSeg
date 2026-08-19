[CmdletBinding()]
param(
    [int]$WaitForProcessId,
    [string]$VmambaPredictionRoot = "",
    [string]$ReportDir = ""
)

<# Wait for a resumable VMamba Validation/Test export, then run the complete
three-model report pipeline. The downstream script checks cache completeness
before any calibration, so it will stop safely if the export did not finish. #>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $VmambaPredictionRoot) { $VmambaPredictionRoot = Join-Path $Root "artifacts\experiments\decision\predictions\vmamba" }
if (-not $ReportDir) { $ReportDir = Join-Path $Root "artifacts\experiments\decision\three_model_experiment_report" }
$Pipeline = Join-Path $Root "scripts\experiments\run_three_model_experiments.ps1"
$Log = Join-Path $ReportDir "run_all_experiments.log"
if (-not (Test-Path -LiteralPath $Pipeline)) { throw "Pipeline not found: $Pipeline" }
New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null

function Write-Log([string]$Message) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message" |
        Tee-Object -FilePath $Log -Append
}

Write-Log "WAIT · VMamba export process $WaitForProcessId"
while (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}
Write-Log "VMamba export process ended · verify cache and run all experiments"
& $Pipeline -Action All -VmambaPredictionRoot $VmambaPredictionRoot -ReportDir $ReportDir 2>&1 |
    Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "Three-model experiment pipeline failed with exit code $LASTEXITCODE" }
Write-Log "COMPLETE · all report tables are ready"
