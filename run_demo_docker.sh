#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
action="${1:-start}"
image=straw-qc-demo:v3.5
name="${STRAW_CONTAINER:-straw-qc-demo}"
volume="${STRAW_VOLUME:-straw-qc-demo-data}"
port="${STRAW_PORT:-8033}"
[[ "$name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]+$ && "$volume" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]+$ ]] || exit 2
[[ "$port" =~ ^[0-9]+$ && "$port" -ge 1024 && "$port" -le 65535 ]] || exit 2
command -v docker >/dev/null || { echo 'Install Docker and the NVIDIA GPU runtime first.' >&2; exit 2; }
docker info >/dev/null
case "$action" in
  start)
    if ! docker image inspect "$image" >/dev/null 2>&1; then
      archive="$root/straw-qc-demo-v3.5.tar.gz"
      [[ -f "$archive" && -f "$archive.sha256" ]] || { echo 'Place the offline image archive here, or build the image as described in docs/STRAW_DOCKER_DEMO.md.' >&2; exit 2; }
      (cd "$root" && sha256sum -c "$(basename "$archive").sha256")
      docker load --input "$archive"
    fi
    if docker container inspect "$name" >/dev/null 2>&1; then
      [[ $(docker inspect --format '{{index .Config.Labels "io.straw.demo"}}' "$name") == container-v3.5 ]] || { echo 'Container name belongs to another application.' >&2; exit 2; }
      [[ $(docker inspect --format '{{.Image}}' "$name") == $(docker image inspect --format '{{.Id}}' "$image") ]] || { echo 'Existing container uses a different image. Stop/remove that container first; keep its data volume.' >&2; exit 2; }
      docker start "$name"
    else
      mounts=()
      # Linux Docker Engine inside WSL needs its WSL video driver libraries.
      for lib in libnvcuvid.so.1 libnvidia-encode.so.1; do
        if [[ -f "/usr/lib/wsl/lib/$lib" ]]; then
          mounts+=(--mount "type=bind,source=/usr/lib/wsl/lib/$lib,target=/usr/lib/x86_64-linux-gnu/$lib,readonly")
        fi
      done
      docker run -d --init --gpus all --restart unless-stopped --name "$name" \
        --label io.straw.demo=container-v3.5 --shm-size=1g \
        -p "127.0.0.1:$port:8033" --mount "type=volume,source=$volume,target=/workspace/artifacts" \
        "${mounts[@]}" "$image"
    fi
    echo "Demo: http://127.0.0.1:$port/ (first start builds the engine for your GPU)."
    echo "Logs: docker logs -f $name | Data volume: $volume"
    ;;
  stop|logs|status)
    [[ $(docker inspect --format '{{index .Config.Labels "io.straw.demo"}}' "$name") == container-v3.5 ]] || { echo 'Refusing another application container.' >&2; exit 2; }
    case "$action" in
      stop) docker stop --time 20 "$name" ;;
      logs) docker logs --tail 100 -f "$name" ;;
      status) docker inspect --format '{{json .State}}' "$name" ;;
    esac
    ;;
  *) echo 'Usage: bash run_demo_docker.sh [start|stop|logs|status]' >&2; exit 2 ;;
esac
