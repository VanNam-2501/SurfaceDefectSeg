[CmdletBinding()]
param(
    [ValidateSet('start','stop','logs','status')][string]$Action = 'start',
    [ValidateRange(1024,65535)][int]$Port = 8033,
    [ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9_.-]+$')][string]$Name = 'straw-qc-demo',
    [ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9_.-]+$')][string]$Volume = 'straw-qc-demo-data',
    [string]$WslDistro = ''
)
$ErrorActionPreference = 'Stop'
if ($WslDistro) {
    $runnerPath = (Join-Path $PSScriptRoot 'run_demo_docker.sh') -replace '\\', '/'
    $linuxScript = & wsl.exe -d $WslDistro -- wslpath -u $runnerPath
    if ($LASTEXITCODE -ne 0) { throw 'Cannot translate the runner path into WSL.' }
    & wsl.exe -d $WslDistro -- env "STRAW_PORT=$Port" "STRAW_CONTAINER=$Name" "STRAW_VOLUME=$Volume" bash $linuxScript $Action
    if ($LASTEXITCODE -ne 0) { throw 'Docker runner failed. Inspect the message above.' }
    return
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Install/start Docker Desktop with its WSL2 backend, or use -WslDistro for a Linux Docker Engine.' }
& docker info | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Docker daemon is not ready.' }
$image = 'straw-qc-demo:v3.5'
if ($Action -eq 'start') {
    & docker image inspect $image 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        $archive = Join-Path $PSScriptRoot 'straw-qc-demo-v3.5.tar.gz'
        if (-not (Test-Path -LiteralPath $archive) -or -not (Test-Path -LiteralPath ($archive + '.sha256'))) { throw 'Place the offline Docker image archive beside this script, or build the image from the source instructions.' }
        $expected = ((Get-Content -LiteralPath ($archive + '.sha256') -Raw).Trim() -split '\s+')[0]
        if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) { throw 'Docker image archive SHA256 mismatch.' }
        & docker load --input $archive
        if ($LASTEXITCODE -ne 0) { throw 'Docker image load failed.' }
    }
    $existing = & docker container inspect $Name 2>$null
    if ($LASTEXITCODE -eq 0) {
        $container = ($existing -join "`n" | ConvertFrom-Json)[0]
        if ($container.Config.Labels.'io.straw.demo' -ne 'container-v3.5') { throw 'Container name belongs to another application.' }
        $expectedImage = & docker image inspect --format '{{.Id}}' $image
        if ($container.Image -ne $expectedImage) { throw 'Existing container uses another image. Stop/remove that container first; keep its data volume.' }
        & docker start $Name
    } else {
        & docker run -d --init --gpus 1 --restart unless-stopped --name $Name --label io.straw.demo=container-v3.5 --shm-size=1g -p "127.0.0.1:${Port}:8033" --mount "type=volume,source=$Volume,target=/workspace/artifacts" $image
    }
    if ($LASTEXITCODE -ne 0) { throw 'Cannot start the GPU container. Check NVIDIA GPU support and whether the port is already in use.' }
    Write-Host "Demo: http://127.0.0.1:$Port/ (the first start builds an engine for this GPU)."
    Write-Host "Logs: docker logs -f $Name | Data volume: $Volume"
} else {
    $label = & docker inspect --format '{{index .Config.Labels "io.straw.demo"}}' $Name
    if ($LASTEXITCODE -ne 0 -or $label -ne 'container-v3.5') { throw 'Refusing another application container.' }
    switch ($Action) {
        stop { & docker stop --time 20 $Name }
        logs { & docker logs --tail 100 -f $Name }
        status { & docker inspect --format '{{json .State}}' $Name }
    }
    if ($LASTEXITCODE -ne 0) { throw 'Docker command failed.' }
}
