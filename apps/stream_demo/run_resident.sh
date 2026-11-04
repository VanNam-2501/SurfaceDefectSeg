#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
artifact="${1:-b1_stream_512}"
case "$artifact" in b1_stream|b1_stream_512|b1_stream_512_pose_v1) ;; *) echo 'Unsupported stream artifact' >&2; exit 2;; esac
out="$root/artifacts/$artifact"
# Docker must find the nested volume target inside the read-only workspace.
# A fresh handover contains only the selected artifact, not the legacy alias.
mkdir -p "$root/artifacts/b1_stream"
identity=$(python3 "$root/apps/stream_demo/serving_contract.py" "$out")
expected_contract=$(printf '%s\n' "$identity" | python3 -c 'import json,sys; print(json.load(sys.stdin)["contract_sha256"])')
size=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["size"])' "$out/manifest.json")
base=ghcr.io/nvidia/deepstream:9.2-samples-multiarch
suffix=$(printf '%s' "$root" | sha256sum | cut -c1-10)
lab="tttn-b1-deepstream-resident-$artifact-$suffix"
test -s "$out/b1_rtx3050_fp16.engine"
base_id=$(docker image inspect --format '{{.Id}}' "$base")
certified_base=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["context"]["sdk_image_id"])' "$out/engine_cache.json")
[[ "$base_id" == "$certified_base" ]] || { echo 'Engine SDK image differs from its certified build' >&2; exit 2; }
certified_gpu=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["context"]["gpu_identity"])' "$out/engine_cache.json")
current_gpu=$(nvidia-smi --query-gpu=name,driver_version,uuid --format=csv,noheader)
[[ "$current_gpu" == "$certified_gpu" ]] || { echo 'Engine GPU/driver differs from its certified build' >&2; exit 2; }
if docker container inspect "$lab" >/dev/null 2>&1; then
  [[ $(docker inspect --format '{{index .Config.Labels "io.tttn.experiment"}}' "$lab") == b1-rtsp ]] || { echo 'Refusing unrelated container'; exit 2; }
  [[ $(docker inspect --format '{{.Image}}' "$lab") == "$base_id" ]] || { echo 'Unexpected image digest'; exit 2; }
  mounted=$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/workspace/artifacts/b1_stream"}}{{.Source}}{{end}}{{end}}' "$lab")
  [[ "$mounted" == "$out" ]] || { echo "Refusing container mounted to another artifact: $mounted" >&2; exit 2; }
else
  docker create --init --name "$lab" --label io.tttn.experiment=b1-rtsp --network host --gpus all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video -e USE_NEW_NVSTREAMMUX=no \
    -e GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins \
    -v /usr/lib/wsl/lib/libnvcuvid.so.1:/usr/lib/x86_64-linux-gnu/libnvcuvid.so.1:ro \
    -v /usr/lib/wsl/lib/libnvidia-encode.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1:ro \
    -v "$root:/workspace:ro" -v "$out:/workspace/artifacts/b1_stream:rw" \
    --entrypoint /bin/bash "$base" -lc 'exec sleep infinity' >/dev/null
fi
current_contract=$(python3 -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(json.loads(p.read_text()).get("contract_sha256", "UNVERIFIED") if p.exists() else "")' "$out/resident_queue/runtime.json" 2>/dev/null || true)
if [[ -n "$current_contract" && "$current_contract" != "$expected_contract" ]]; then
  # This container is dedicated to this artifact and its mount was verified.
  # An old queue owner must release CUDA and the file lock before reloading.
  docker restart -t 3 "$lab" >/dev/null
else
  docker start "$lab" >/dev/null
fi
# Do not stop the container per selection: engine/context/GPU buffers stay loaded.
exec docker exec "$lab" python3 /workspace/apps/stream_demo/resident_worker.py \
  --engine /workspace/artifacts/b1_stream/b1_rtx3050_fp16.engine \
  --queue /workspace/artifacts/b1_stream/resident_queue --keepalive-seconds 1
