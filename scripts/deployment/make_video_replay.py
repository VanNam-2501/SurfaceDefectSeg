"""Make an explicitly synthetic conveyor clip for integration tests, not accuracy."""
from pathlib import Path
import argparse
import csv
import json
import shutil
import subprocess
import sys
from PIL import Image, ImageFilter, ImageDraw

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "demo_samples/conveyor_replay.mp4")
    args = parser.parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required to generate replay video")
    rows = list(csv.DictReader((ROOT / "artifacts/straw_global_local/data_v1/val.csv").open(encoding="utf-8-sig")))
    tiles, cases = [], []
    for index, name in enumerate(("good", "bent", "loose", "missing", "pierced")):
        row = next(r for r in rows if r["class_name"] == name)
        from PIL import ImageOps
        with Image.open(ROOT / "data/straw_defects_raw" / row["relative_path"]) as image:
            tile = ImageOps.exif_transpose(image).convert("RGB").resize((220, 220))
        condition = "clear"
        if index == 3:
            tile = tile.filter(ImageFilter.GaussianBlur(18))
            condition = "blurred"
        if index == 4:
            tile = tile.point(lambda v: int(v * .08))
            condition = "dark"
        tiles.append(tile)
        cases.append({"sample_id": row["sample_id"], "class_name": name, "condition": condition,
                      "source_role": "val", "expected_track": index + 1})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.stem + '.building.mp4')
    process = subprocess.Popen([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", "960x540", "-r", "30", "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-bf", "0", "-movflags", "+faststart", str(temporary)], stdin=subprocess.PIPE)
    fps, initial, travel = 30, 1.5, 4.0
    count = int((initial + len(tiles) * travel + .5) * fps)
    background = Image.new('RGB', (960,540), (204,214,222))
    draw = ImageDraw.Draw(background)
    draw.rectangle((0,115,959,416), fill=(115,128,139))
    draw.rectangle((0,105,959,120), fill=(78,96,111))
    draw.rectangle((0,410,959,427), fill=(67,84,100))
    draw.rectangle((0,428,959,486), fill=(155,170,182))
    for x in range(20,960,65):
        draw.rounded_rectangle((x,434,x+48,477),radius=9,fill=(182,194,204),outline=(130,149,164),width=2)
    try:
        for frame in range(count):
            image = background.copy()
            t = frame / fps - initial
            if t >= 0:
                item = int(t / travel)
                if item < len(tiles):
                    x = round(960 - (t % travel) / travel * 1200)
                    shadow = ImageDraw.Draw(image)
                    shadow.rounded_rectangle((x+4,160,x+224,380),radius=5,fill=(88,102,113))
                    image.paste(tiles[item], (x, 155))
            process.stdin.write(image.tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("FFmpeg replay encoding failed")
    temporary.replace(args.output)
    args.output.with_suffix(".json").write_text(json.dumps({
        "scope": "Synthetic moving validation-photo tiles; tests tracking, quality review and GPU integration only",
        "fps": fps, "frames": count, "seconds": count / fps, "initial_empty_seconds": initial,
        "render_speed_pixels_per_second": 1200 / travel,
        "expected_products": 5, "cases": cases,
        "not_an_accuracy_benchmark": True}, indent=2), encoding="utf-8")
    print(f"Synthetic replay: {args.output}, {count} frames, {count/fps:.1f} seconds")


if __name__ == "__main__":
    main()
