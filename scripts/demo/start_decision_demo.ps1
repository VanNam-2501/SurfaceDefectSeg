[CmdletBinding()]
param(
    [string]$UnetCheckpoint = "",
    [string]$SegformerCheckpoint = "",
    [string]$VmambaCheckpoint = "",
    [string]$DecisionPolicy = "",
    [string]$LearnedVerifierPolicy = "",
    [string]$LearnedVerifierModels = "",
    [int]$ApiPort = 8000
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$ProjectRoot = Join-Path $RepoRoot "src\threecad_segmentation"
$WebRoot = Join-Path $RepoRoot "apps\web_demo"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$WeightRoot = Join-Path $RepoRoot "artifacts\checkpoints\final"
if (-not $UnetCheckpoint) {
    $UnetCheckpoint = Join-Path $WeightRoot "unet_best.pt"
}
if (-not $SegformerCheckpoint) {
    $SegformerCheckpoint = Join-Path $WeightRoot "segformer_best.pt"
}
if (-not $VmambaCheckpoint) {
    $VmambaCheckpoint = Join-Path $WeightRoot "vmamba_best.pt"
}
if (-not $DecisionPolicy) {
    $DecisionPolicy = Join-Path $RepoRoot "artifacts\reports\final\decision_and_test_audit\spatial\unet_vmamba\policy\decision_policy.json"
}
if (-not $LearnedVerifierPolicy) {
    $LearnedVerifierPolicy = Join-Path $RepoRoot "artifacts\reports\final\decision_and_test_audit\hybrid_pairs\unet_vmamba\learned_policy.json"
}
if (-not $LearnedVerifierModels) {
    $LearnedVerifierModels = Join-Path $RepoRoot "artifacts\reports\final\decision_and_test_audit\hybrid_pairs\unet_vmamba\models"
}
foreach ($Path in @($Python, $ProjectRoot, $WebRoot, $UnetCheckpoint, $SegformerCheckpoint, $DecisionPolicy, $LearnedVerifierPolicy, $LearnedVerifierModels)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required path not found: $Path"
    }
}
if (-not (Test-Path -LiteralPath $VmambaCheckpoint)) {
    throw "VMamba checkpoint not found: $VmambaCheckpoint"
}

$env:SEGMENTATION_PROJECT_ROOT = $ProjectRoot
$env:UNET_CHECKPOINT = $UnetCheckpoint
$env:SEGFORMER_CHECKPOINT = $SegformerCheckpoint
$env:DECISION_POLICY = $DecisionPolicy
$env:LEARNED_VERIFIER_POLICY = $LearnedVerifierPolicy
$env:LEARNED_VERIFIER_MODELS = $LearnedVerifierModels
$env:VMAMBA_CHECKPOINT = $VmambaCheckpoint
$env:NEXT_PUBLIC_INFERENCE_API = "http://127.0.0.1:$ApiPort"

$LogRoot = Join-Path $RepoRoot "artifacts\experiments\decision\logs"
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
$Api = Start-Process -FilePath $Python -ArgumentList @(
    "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", [string]$ApiPort
) -WorkingDirectory $WebRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $LogRoot "api.stdout.log") `
    -RedirectStandardError (Join-Path $LogRoot "api.stderr.log")

$Npm = (Get-Command npm.cmd -ErrorAction Stop).Source
$Frontend = Start-Process -FilePath $Npm -ArgumentList @("run", "dev") `
    -WorkingDirectory $WebRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $LogRoot "web.stdout.log") `
    -RedirectStandardError (Join-Path $LogRoot "web.stderr.log")

Write-Host "Decision demo started." -ForegroundColor Green
Write-Host "API PID: $($Api.Id) · http://127.0.0.1:$ApiPort/health"
Write-Host "Web PID: $($Frontend.Id) · check $LogRoot\web.stdout.log for its URL"
Write-Host "Policy: $DecisionPolicy"
Write-Host "Learned verifier: $LearnedVerifierPolicy"
Write-Host "Stop later with: Stop-Process -Id $($Api.Id),$($Frontend.Id)"
