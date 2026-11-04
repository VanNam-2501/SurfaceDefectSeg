#!/usr/bin/env bash
# Start a short, browser-visible TensorRT/DeepStream preview from a repeated dataset clip.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
artifact="${2:-b1_stream_512_pose_v1}"
case "$artifact" in b1_stream|b1_stream_512|b1_stream_512_pose_v1) ;; *) echo 'Unsupported stream artifact' >&2; exit 2;; esac
out="$root/artifacts/$artifact"
mkdir -p "$root/artifacts/b1_stream"
python3 "$root/apps/stream_demo/serving_contract.py" "$out" >/dev/null
seconds="${1:-45}"
[[ "$seconds" =~ ^[0-9]+$ ]] && (( seconds >= 10 && seconds <= 300 )) || { echo 'Preview duration must be 10..300 seconds' >&2; exit 2; }
for file in "$out/b1_rtx3050_fp16.engine" "$out/nvinfer.txt" "$out/quality.mp4" "$out/manifest.json"; do
  test -s "$file" || { echo "Missing prepared TensorRT input: $file" >&2; exit 2; }
done
if grep -Eq '^[[:space:]]*onnx-file[[:space:]]*=' "$out/nvinfer.txt"; then
  echo 'Runtime must load the TensorRT engine only' >&2; exit 2
fi
base=ghcr.io/nvidia/deepstream:9.2-samples-multiarch
lab="tttn-b1-deepstream-preview-$artifact"
server_pid=''
cleanup() {
  [[ -z "$server_pid" ]] || kill "$server_pid" 2>/dev/null || true
  docker stop -t 3 "$lab" >/dev/null 2>&1 || true
}
base_id=$(docker image inspect --format '{{.Id}}' "$base")
if docker container inspect "$lab" >/dev/null 2>&1; then
  [[ $(docker inspect --format '{{index .Config.Labels "io.tttn.experiment"}}' "$lab") == b1-rtsp ]] || { echo 'Refusing unrelated container'; exit 2; }
  [[ $(docker inspect --format '{{.Image}}' "$lab") == "$base_id" ]] || { echo 'Lab base-image digest changed'; exit 2; }
  mounted=$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/workspace/artifacts/b1_stream"}}{{.Source}}{{end}}{{end}}' "$lab")
  [[ "$mounted" == "$out" ]] || { echo 'Container artifact mount mismatch' >&2; exit 2; }
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
docker exec "$lab" bash /workspace/apps/stream_demo/inside_compile.sh > "$out/preview_compile.log" 2>&1
loop_video="$out/preview_loop.mp4"
ffmpeg -hide_banner -loglevel warning -y -stream_loop -1 -i "$out/quality.mp4" -t "$seconds" -c copy -movflags +faststart "$loop_video"
python3 apps/stream_demo/serve_rtsp.py "$loop_video" > "$out/preview_rtsp_server.log" 2>&1 &
server_pid=$!
for _ in {1..100}; do
  grep -q 'RTSP READY' "$out/preview_rtsp_server.log" && break
  kill -0 "$server_pid" || { cat "$out/preview_rtsp_server.log"; exit 3; }
  sleep .1
done
grep -q 'RTSP READY' "$out/preview_rtsp_server.log"
preview_dir=/workspace/artifacts/b1_stream/preview
rm -f "$out/preview"/frame_*.jpg "$out/preview"/frame_*.json \
  "$out/preview"/frame_*.jpg.tmp "$out/preview"/frame_*.json.tmp \
  "$out/preview/session_running" "$out/preview/latest.jpg" "$out/preview/latest.json" \
  "$out/preview/latest.jpg.tmp" "$out/preview/latest.json.tmp"
echo "TensorRT browser preview active for ${seconds}s at http://127.0.0.1:8033/"
docker exec "$lab" bash /workspace/apps/stream_demo/inside_run.sh \
  rtsp://127.0.0.1:8554/straw /workspace/artifacts/b1_stream/nvinfer.txt \
  /workspace/artifacts/b1_stream/preview.jsonl "$seconds" "$preview_dir" \
  > "$out/preview_inference.log" 2>&1
echo "Preview finished; frame and final prediction remain in $out/preview/"
