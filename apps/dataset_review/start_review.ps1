param(
    [string]$DatasetRoot = "",
    [string[]]$ResultsRoot = @(),
    [int]$Port = 8765,
    [switch]$NoBrowser
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
if (-not (Test-Path -LiteralPath $DatasetRoot)) {
    throw "Dataset root not found: $DatasetRoot"
}

# Auto-load portable local prediction caches. The user can override this by
# passing one or more -ResultsRoot values for another experiment directory.
if ($ResultsRoot.Count -eq 0) {
    $knownResults = Join-Path $repoRoot "artifacts\experiments\decision\predictions"
    if (Test-Path -LiteralPath $knownResults) {
        $ResultsRoot = @($knownResults)
    }
}

$appArgs = @(
    (Join-Path $toolDir "app.py"),
    "--dataset-root", $DatasetRoot,
    "--port", $Port
)
foreach ($root in $ResultsRoot) {
    if (Test-Path -LiteralPath $root) {
        $appArgs += @("--results-root", $root)
    }
}
if (-not $NoBrowser) {
    $appArgs += "--open-browser"
}

Write-Host "Data Review Studio is starting at http://127.0.0.1:$Port"
Write-Host "Press Ctrl+C to stop. The source dataset remains read-only."
& $python @appArgs
