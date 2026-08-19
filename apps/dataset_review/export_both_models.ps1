param(
    [string[]]$Splits = @("train", "val", "test"),
    [int]$TileBatchSize = 1
)

$toolDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$runner = Join-Path $toolDir "export_predictions.ps1"

Write-Host "Step 1/2 - UNet predictions"
& $runner -Model unet -Splits $Splits -TileBatchSize $TileBatchSize
if ($LASTEXITCODE -ne 0) {
    throw "UNet export failed. Re-run this command after resolving the error; completed images will be skipped."
}

Write-Host "Step 2/2 - SegFormer predictions"
& $runner -Model segformer -Splits $Splits -TileBatchSize $TileBatchSize
if ($LASTEXITCODE -ne 0) {
    throw "SegFormer export failed. Re-run this command; completed images will be skipped."
}

Write-Host "Both models are complete. Restart Data Review Studio and refresh the browser."

