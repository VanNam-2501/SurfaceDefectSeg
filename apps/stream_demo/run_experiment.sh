#!/usr/bin/env bash
# Run as root in the isolated WSL distro. Does not modify the HTTP demo.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
out="$root/artifacts/${STRAW_STREAM_ARTIFACT:-b1_stream}"
mkdir -p "$root/artifacts/b1_stream"
lab="${STRAW_STREAM_LAB:-tttn-b1-deepstream-lab}"
size=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["size"])' "$out/manifest.json")
test -f "$out/manifest.json"
test -f "$out/quality.mp4"
test -f "$out/load_30fps.mp4"
sha256sum "$out/quality.mp4" "$out/load_30fps.mp4" "$out/b1_rawrgb${size}.onnx" "$out/nvinfer.txt" \
  apps/stream_demo/deepstream_probe.cpp apps/stream_demo/inside_build_engine.sh > "$out/run_input_hashes.txt"
base=ghcr.io/nvidia/deepstream:9.2-samples-multiarch
server_pid=''
monitor_pid=''
cleanup() {
  [[ -z "$server_pid" ]] || kill "$server_pid" 2>/dev/null || true
  [[ -z "$monitor_pid" ]] || kill "$monitor_pid" 2>/dev/null || true
  docker stop -t 3 "$lab" >/dev/null 2>&1 || true
}
base_id=$(docker image inspect --format '{{.Id}}' "$base")
if docker container inspect "$lab" >/dev/null 2>&1; then
  [[ $(docker inspect --format '{{index .Config.Labels "io.tttn.experiment"}}' "$lab") == b1-rtsp ]] || { echo 'Refusing unrelated container'; exit 2; }
  [[ $(docker inspect --format '{{.Image}}' "$lab") == "$base_id" ]] || { echo 'Lab base-image digest changed'; exit 2; }
  mounted=$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/workspace/artifacts/b1_stream"}}{{.Source}}{{end}}{{end}}' "$lab")
  [[ "$mounted" == "$out" ]] || { echo 'Lab artifact mount differs from this checkout' >&2; exit 2; }
else
  docker create --init --name "$lab" --label io.tttn.experiment=b1-rtsp --network host --gpus all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video -e USE_NEW_NVSTREAMMUX=no \
    -e GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins \
    -v /usr/lib/wsl/lib/libnvcuvid.so.1:/usr/lib/x86_64-linux-gnu/libnvcuvid.so.1:ro \
    -v /usr/lib/wsl/lib/libnvidia-encode.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1:ro \
    -v "$root:/workspace:ro" -v "$out:/workspace/artifacts/b1_stream:rw" \
    --entrypoint /bin/bash "$base" -lc 'exec sleep infinity' > "$out/lab_create.txt"
fi
docker start "$lab" >/dev/null
trap cleanup EXIT INT TERM
docker image inspect "$base" > "$out/base_image.json"
docker inspect "$lab" > "$out/lab_container.json"
nvidia-smi --query-gpu=name,driver_version,uuid --format=csv,noheader > "$out/gpu_identity.txt"
python3 apps/stream_demo/engine_cache.py check
if [[ ! -s "$out/b1_rtx3050_fp16.engine" ]]; then
  echo 'Converting build-only ONNX into an RTX 3050 TensorRT FP16 engine'
  docker exec "$lab" bash /workspace/apps/stream_demo/inside_build_engine.sh > "$out/tensorrt_build.log" 2>&1
fi
docker exec "$lab" bash /workspace/apps/stream_demo/inside_compile.sh > "$out/compile.log" 2>&1
{ cat /etc/os-release; docker version; nvidia-smi; ffmpeg -version | head -n 3; } > "$out/environment.txt"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw \
  --format=csv -l 1 > "$out/gpu_samples.csv" &
monitor_pid=$!
echo 'Running complete 300-frame file-quality check'
bash apps/stream_demo/run_container.sh file > "$out/file.log" 2>&1
python3 apps/stream_demo/engine_cache.py record
echo 'Starting localhost RTSP replay'
python3 apps/stream_demo/serve_rtsp.py "$out/load_30fps.mp4" > "$out/rtsp_server.log" 2>&1 &
server_pid=$!
for _ in {1..50}; do
  grep -q 'RTSP READY' "$out/rtsp_server.log" && break
  kill -0 "$server_pid" || { cat "$out/rtsp_server.log"; exit 3; }
  sleep .1
done
grep -q 'RTSP READY' "$out/rtsp_server.log"
echo 'Running short 8-second RTSP / GPU inference check'
STRAW_RTSP_URI=rtsp://127.0.0.1:8554/straw STRAW_TEST_SECONDS=8 \
  bash apps/stream_demo/run_container.sh rtsp > "$out/rtsp.log" 2>&1
echo 'Finished; run report_b1_stream.py in the Windows project venv'
