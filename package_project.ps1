[CmdletBinding()]
param([ValidateSet('all','handover','source')][string]$Kind = 'all')
. (Join-Path $PSScriptRoot 'scripts\deployment\demo_common.ps1')
$python = Get-DemoPython
& $python (Join-Path $PSScriptRoot 'scripts\packaging\package_project.py') --kind $Kind
if ($LASTEXITCODE -ne 0) { throw 'Project packaging failed.' }
