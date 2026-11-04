#!/usr/bin/env bash
# Run one TensorRT/DeepStream inference for each selected still image.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
artifact="${2:-b1_stream_512_pose_v1}"
case "$artifact" in b1_stream|b1_stream_512|b1_stream_512_pose_v1) ;; *) echo 'Unsupported stream artifact' >&2; exit 2;; esac
out="$root/artifacts/$artifact"
python3 "$root/apps/stream_demo/serving_contract.py" "$out" >/dev/null
jobs="$out/image_jobs"
job_id="${1:-}"
[[ "$job_id" =~ ^[0-9a-f]{32}$ ]] || { echo 'Invalid image job id' >&2; exit 2; }
job="$jobs/$job_id"
manifest="$job/manifest.json"
test -s "$manifest" || { echo 'Image manifest is missing' >&2; exit 2; }
test -s "$out/b1_rtx3050_fp16.engine" || { echo 'TensorRT engine is missing' >&2; exit 2; }
test -s "$out/nvinfer.txt" || { echo 'nvinfer config is missing' >&2; exit 2; }
grep -Eq '^[[:space:]]*onnx-file[[:space:]]*=' "$out/nvinfer.txt" && { echo 'Runtime must load the serialized TensorRT engine only' >&2; exit 2; }
for tool in ffmpeg ffprobe docker python3; do command -v "$tool" >/dev/null || { echo "Missing required tool: $tool" >&2; exit 2; }; done

mapfile -t images < <(python3 - "$manifest" <<'PY'
import json, re, sys
data=json.load(open(sys.argv[1], encoding='utf-8'))
files=data.get('files', [])
if not files or len(files)>30: raise SystemExit('Manifest image count is invalid')
for item in files:
    name=item.get('stored', '')
    if not re.fullmatch(r'image_[0-9]{5}\.(?:jpg|jpeg|png|webp|bmp)', name):
        raise SystemExit('Unsafe manifest image name')
    print(name)
PY
)
count=${#images[@]}
(( count > 0 && count <= 30 )) || { echo 'No valid images in manifest' >&2; exit 2; }
for name in "${images[@]}"; do test -s "$job/$name" || { echo "Missing uploaded image: $name" >&2; exit 2; }; done

base=ghcr.io/nvidia/deepstream:9.2-samples-multiarch
lab="tttn-b1-deepstream-batch-$artifact"
server_pid=''
cleanup() {
  [[ -z "$server_pid" ]] || kill "$server_pid" 2>/dev/null || true
  docker stop -t 3 "$lab" >/dev/null 2>&1 || true
}
base_id=$(docker image inspect --format '{{.Id}}' "$base")
if docker container inspect "$lab" >/dev/null 2>&1; then
  [[ $(docker inspect --format '{{index .Config.Labels "io.tttn.experiment"}}' "$lab") == b1-rtsp ]] || { echo 'Refusing unrelated container' >&2; exit 2; }
  [[ $(docker inspect --format '{{.Image}}' "$lab") == "$base_id" ]] || { echo 'Lab base-image digest changed' >&2; exit 2; }
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
echo "Preparing $count image frame(s) for TensorRT inference"
frame_dir="$job/frames"
mkdir -p "$frame_dir"
rm -f "$frame_dir"/frame_*.png "$job/inference.jsonl" "$job/inference.jsonl.summary.json"
for ((i=0; i<count; i++)); do
  printf -v target 'frame_%05d.png' "$((i+1))"
  ffmpeg -hide_banner -loglevel error -y -i "$job/${images[$i]}" \
    -vf 'scale=640:640:flags=bicubic,format=rgb24' -frames:v 1 "$frame_dir/$target"
done
# RTSP file replay can close at EOS before the decoder publishes its final
# image. Add a short tail of the last still; the probe quits after exactly the
# selected-image count, so these tail frames can never become extra results.
tail_frames=5
for ((i=1; i<=tail_frames; i++)); do
  printf -v target 'frame_%05d.png' "$((count+i))"
  cp "$frame_dir/$(printf 'frame_%05d.png' "$count")" "$frame_dir/$target"
done
video="$job/input.mp4"
encoded_count=$((count+tail_frames))
ffmpeg -hide_banner -loglevel error -y -framerate 5 -start_number 1 -i "$frame_dir/frame_%05d.png" \
  -frames:v "$encoded_count" -c:v libx264 -preset ultrafast -crf 18 -pix_fmt yuv420p \
  -g 1 -keyint_min 1 -bf 0 -r 5 -movflags +faststart "$video"
actual=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$video")
[[ "$actual" == "$encoded_count" ]] || { echo "Encoded frame count mismatch: expected=$encoded_count got=$actual" >&2; exit 3; }

docker exec "$lab" bash /workspace/apps/stream_demo/inside_compile.sh
python3 "$root/apps/stream_demo/serve_rtsp.py" "$video" > "$job/rtsp.log" 2>&1 &
server_pid=$!
for _ in {1..100}; do
  grep -q 'RTSP READY' "$job/rtsp.log" && break
  kill -0 "$server_pid" || { cat "$job/rtsp.log"; exit 3; }
  sleep .1
done
grep -q 'RTSP READY' "$job/rtsp.log" || { cat "$job/rtsp.log"; exit 3; }
echo "RTSP replay ready; DeepStream TensorRT will emit exactly $count prediction(s) on GPU (with $tail_frames drain frames)"
docker exec "$lab" bash /workspace/apps/stream_demo/inside_run.sh \
  rtsp://127.0.0.1:8554/straw /workspace/artifacts/b1_stream/nvinfer.txt \
  "/workspace/artifacts/b1_stream/image_jobs/$job_id/inference.jsonl" 0 "" "$count"
predicted=$(python3 - "$job/inference.jsonl" <<'PY'
import json, sys
print(sum(1 for line in open(sys.argv[1], encoding='utf-8') if line.strip()))
PY
)
[[ "$predicted" == "$count" ]] || { echo "Prediction count mismatch: expected=$count got=$predicted" >&2; exit 3; }
echo "SUCCESS TensorRT predictions=$predicted/$count"
