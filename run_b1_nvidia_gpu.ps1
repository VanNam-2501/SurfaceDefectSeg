param(
    [int]$Repeats = 300,
    [switch]$PrepareOnly,
    [switch]$TrtOnly,
    [switch]$SkipTensorRT
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$study = Join-Path $PSScriptRoot 'scripts\research\benchmark_b1_nvidia_gpu.py'
$ortTarget = Join-Path $PSScriptRoot 'artifacts\b1_deployment\onnxruntime_gpu_1_26'
$trtTarget = Join-Path $PSScriptRoot 'artifacts\b1_deployment\tensorrt_cuda12_11_4'
$converterTarget = Join-Path $PSScriptRoot 'artifacts\b1_deployment\onnxconverter_common'
if (-not (Test-Path -LiteralPath $ortTarget)) {
    throw "Install ONNX Runtime CUDA with: .venv\Scripts\python.exe -m pip install --target artifacts\b1_deployment\onnxruntime_gpu_1_26 -r requirements\b1-nvidia-gpu-study.txt"
}
if (-not (Test-Path -LiteralPath $trtTarget)) {
    throw "Install TensorRT CUDA 12 with: .venv\Scripts\python.exe -m pip install --target artifacts\b1_deployment\tensorrt_cuda12_11_4 -r requirements\b1-tensorrt-cu12.txt"
}
if (-not (Test-Path -LiteralPath $converterTarget)) {
    throw "Install ONNX FP16 converter with: .venv\Scripts\python.exe -m pip install --target artifacts\b1_deployment\onnxconverter_common --no-deps onnxconverter-common==1.16.0"
}
$arguments = @($study, '--repeats', $Repeats)
if ($PrepareOnly) { $arguments += '--prepare-only' }
if ($TrtOnly) { $arguments += '--trt-only' }
if ($SkipTensorRT) { $arguments += '--skip-tensorrt' }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "NVIDIA GPU benchmark failed with exit code $LASTEXITCODE. Completed artifacts remain under artifacts\b1_nvidia_gpu." }
