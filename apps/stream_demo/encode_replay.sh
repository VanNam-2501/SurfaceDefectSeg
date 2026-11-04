#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
out="artifacts/${STRAW_STREAM_ARTIFACT:-b1_stream}"
test -f "$out/manifest.json"
ffmpeg -hide_banner -loglevel warning -y -framerate 30 -start_number 0 \
  -i "$out/frames/%04d.jpg" -frames:v 300 \
  -vf format=yuv420p -c:v libx264 -preset fast -crf 18 \
  -x264-params bframes=0:keyint=30:min-keyint=30:scenecut=0 \
  -movflags +faststart "$out/quality.mp4"
# User requested a short trial: same ten-second clip, no long replay.
cp "$out/quality.mp4" "$out/load_30fps.mp4"
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate,nb_read_frames \
  -of json "$out/quality.mp4" > "$out/quality_probe.json"
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate,nb_read_frames \
  -of json "$out/load_30fps.mp4" > "$out/load_probe.json"
