[CmdletBinding()]
param(
    [switch]$NoBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = $PSScriptRoot
Push-Location $RepoRoot
try {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker was not found. Install Docker Desktop and start it first."
    }

    if ($NoBuild) {
        docker compose up -d
    } else {
        docker compose up -d --build
    }
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose failed with exit code $LASTEXITCODE"
    }

    Write-Host "Demo is starting." -ForegroundColor Green
    Write-Host "Web:     http://localhost:3000"
    Write-Host "API:     http://localhost:8000/health"
    Write-Host "Results: artifacts/reports/final/visualizations/index.html"
    Write-Host "First build downloads Python/Node dependencies and can take several minutes."
} finally {
    Pop-Location
}

