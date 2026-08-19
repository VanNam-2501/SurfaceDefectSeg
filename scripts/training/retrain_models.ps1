[CmdletBinding()]
param(
    [ValidateSet("all", "unet", "segformer", "vmamba")]
    [string]$Model = "all",

    # The default is the dataset shipped inside this training project.  To use
    # labels corrected in Data Review Studio, point this at an exported
    # `training_dataset` directory instead.
    [string]$DatasetRoot = "",
    [string]$OutputDir = "",
    [string]$RunName = "retrain_seed42",
    [int]$Seed = 42,
    [int]$Epochs = 50,
    [int]$NumWorkers = 2,
    [int]$PatchSize = 512,
    [int]$Stride = 256,
    [int]$ValTileBatchSize = 1,
    [switch]$SkipTest,
    [switch]$SkipProtocolCheck
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$projectRoot = Join-Path $repoRoot "src\threecad_segmentation"
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment not found: $python"
}

if (-not $DatasetRoot) {
    $DatasetRoot = Join-Path $repoRoot "data\3cad_ani"
}
if (-not (Test-Path -LiteralPath $DatasetRoot -PathType Container)) {
    throw "Dataset root not found: $DatasetRoot"
}
$DatasetRoot = (Resolve-Path -LiteralPath $DatasetRoot).Path

$splitsRoot = Join-Path $DatasetRoot "dataset_audit\splits"
$trainCsv = Join-Path $splitsRoot "train.csv"
$valCsv = Join-Path $splitsRoot "val.csv"
$testCsv = Join-Path $splitsRoot "test.csv"
foreach ($csv in @($trainCsv, $valCsv, $testCsv)) {
    if (-not (Test-Path -LiteralPath $csv -PathType Leaf)) {
        throw "Expected frozen split file not found: $csv"
    }
}

if (-not $OutputDir) {
    $OutputDir = Join-Path $repoRoot "artifacts\training"
}
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# A 4 GB RTX 3050 needs conservative batches.  Gradient accumulation keeps
# the protocol's effective batch size at four while full-resolution evaluation
# uses one tile at a time to avoid an avoidable CUDA OOM.
$specs = @(
    [PSCustomObject]@{ Key = "unet";      TrainScript = "train_unet.py";      ResultFolder = "unet_r18";        BatchSize = 1; GradAccum = 4 },
    [PSCustomObject]@{ Key = "segformer"; TrainScript = "train_segformer.py"; ResultFolder = "segformer_b0";     BatchSize = 1; GradAccum = 4 },
    [PSCustomObject]@{ Key = "vmamba";    TrainScript = "train_vmamba.py";    ResultFolder = "vmamba_t_s2l5"; BatchSize = 1; GradAccum = 4 }
)
$selected = if ($Model -eq "all") { $specs } else { @($specs | Where-Object { $_.Key -eq $Model }) }
if ($selected.Count -ne 1 -and $Model -ne "all") {
    throw "Unknown model selection: $Model"
}

# Never silently overwrite an earlier experiment.  Use a new -RunName for a
# fresh run, or run the underlying train_*.py command with --resume explicitly.
foreach ($spec in $selected) {
    $runDir = Join-Path (Join-Path $OutputDir $spec.ResultFolder) $RunName
    if (Test-Path -LiteralPath $runDir) {
        throw "Run already exists: $runDir`nChoose another -RunName so the prior experiment remains reproducible."
    }
}

Push-Location $repoRoot
try {
    Write-Host "Project root : $projectRoot"
    Write-Host "Dataset root : $DatasetRoot"
    Write-Host "Output root  : $OutputDir"
    Write-Host "Models       : $($selected.Key -join ', ')"

    if (-not $SkipProtocolCheck) {
        $protocolReport = Join-Path $OutputDir "protocol_check_$RunName.json"
        & $python (Join-Path $repoRoot "scripts\verification\check_protocol.py") `
            "--dataset-root" $DatasetRoot `
            "--train-csv" $trainCsv `
            "--val-csv" $valCsv `
            "--test-csv" $testCsv `
            "--save" $protocolReport
        if ($LASTEXITCODE -ne 0) { throw "Protocol preflight failed." }
    }

    # VMamba has a native CUDA extension.  Running its official smoke test
    # before any lengthy job gives an immediate, actionable failure when the
    # VMamba repository or a GPU-compatible mamba-ssm build is missing.
    if ($selected.Key -contains "vmamba") {
        & $python (Join-Path $repoRoot "tests\ml\test_vmamba_runtime.py")
        if ($LASTEXITCODE -ne 0) {
            throw "VMamba runtime check failed. Set up a GPU-compatible VMamba/mamba-ssm runtime before training VMamba. U-Net and SegFormer can still be run separately."
        }
    }

    foreach ($spec in $selected) {
        Write-Host "`n===== TRAIN $($spec.Key.ToUpper()) =====" -ForegroundColor Cyan
        $trainArgs = @(
            "--dataset-root", $DatasetRoot,
            "--train-csv", $trainCsv,
            "--val-csv", $valCsv,
            "--test-csv", $testCsv,
            "--output-dir", $OutputDir,
            "--run-name", $RunName,
            "--seed", $Seed,
            "--epochs", $Epochs,
            "--batch-size", $spec.BatchSize,
            "--grad-accum", $spec.GradAccum,
            "--num-workers", $NumWorkers,
            "--patch-size", $PatchSize,
            "--stride", $Stride,
            "--val-tile-batch-size", $ValTileBatchSize,
            "--device", "cuda"
        )
        & $python (Join-Path $PSScriptRoot $spec.TrainScript) @trainArgs
        if ($LASTEXITCODE -ne 0) { throw "Training failed for $($spec.Key)." }

        $checkpoint = Join-Path (Join-Path (Join-Path $OutputDir $spec.ResultFolder) $RunName) "checkpoints\best.pt"
        if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
            throw "Best checkpoint was not produced: $checkpoint"
        }

        Write-Host "`n===== VALIDATION / THRESHOLD SELECTION: $($spec.Key.ToUpper()) =====" -ForegroundColor Yellow
        $evaluationArgs = @(
            "--model", $spec.Key,
            "--checkpoint", $checkpoint,
            "--dataset-root", $DatasetRoot,
            "--train-csv", $trainCsv,
            "--val-csv", $valCsv,
            "--test-csv", $testCsv,
            "--tile-size", $PatchSize,
            "--stride", $Stride,
            "--tile-batch-size", $ValTileBatchSize,
            "--device", "cuda"
        )
        & $python (Join-Path $repoRoot "scripts\evaluation\evaluate_model.py") @evaluationArgs "--split" "val"
        if ($LASTEXITCODE -ne 0) { throw "Validation evaluation failed for $($spec.Key)." }

        if (-not $SkipTest) {
            Write-Host "`n===== TEST (FROZEN VALIDATION THRESHOLD): $($spec.Key.ToUpper()) =====" -ForegroundColor Green
            & $python (Join-Path $repoRoot "scripts\evaluation\evaluate_model.py") @evaluationArgs "--split" "test"
            if ($LASTEXITCODE -ne 0) { throw "Test evaluation failed for $($spec.Key)." }
        }
    }

    Write-Host "`nCompleted. Results are in: $OutputDir" -ForegroundColor Green
}
finally {
    Pop-Location
}
