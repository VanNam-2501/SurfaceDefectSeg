[CmdletBinding()]
param(
    [ValidateSet('prepare','plan','train','calibrate','freeze','test','benchmark','report','smoke-data','lobo-data')]
    [string]$Action = 'plan',
    [ValidateSet('core','extended','smoke')][string]$Suite = 'core',
    [string]$Protocol = 'artifacts/straw_thesis/data_v2/iid',
    [string]$Output = 'artifacts/straw_thesis/core_v1',
    [string]$DataRoot = 'data/straw_defects_raw',
    [string]$Device = 'cuda',
    [string[]]$RunId = @(),
    [string[]]$Group = @()
)
$ErrorActionPreference = 'Stop'
$Python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $Python)) { $Python = 'python' }
Push-Location $PSScriptRoot
try {
    $CliArgs = @('-m', 'src.straw_defect_cls.thesis_runner', $Action, '--suite', $Suite,
                 '--protocol', $Protocol, '--output', $Output, '--data-root', $DataRoot, '--device', $Device)
    foreach ($Item in $RunId) { $CliArgs += @('--run-id', $Item) }
    foreach ($Item in $Group) { $CliArgs += @('--group', $Item) }
    & $Python @CliArgs
    if ($LASTEXITCODE -ne 0) { throw "Straw thesis $Action failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }
