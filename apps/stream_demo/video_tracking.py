"""Foreground tracking for a fixed camera with multiple products in the ROI.

Calibration needs an empty belt. Each observed product is tracked separately.
"""
from __future__ import annotations
from collections import deque
from dataclasses import asdict, dataclass
import math
import time
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
try:
    from .image_quality import QualityConfig, assess
    from .tracking_math import associate, predicted_box
except ImportError:
    from image_quality import QualityConfig, assess
    from tracking_math import associate, predicted_box


@dataclass(frozen=True)
class GateConfig:
    roi: tuple = (.25, .08, .50, .84)
    line: float = .50
    direction: str = "right_to_left"
    threshold: float = 30
    min_area: int = 180
    calibration_frames: int = 8
    max_missed: int = 5
    crop_padding: int = 8
    content_inset: int = 0
    crossing_grace_ms: int = 900

    @classmethod
    def parse(cls, values=None):
        values = values or {}
        if not isinstance(values, dict) or set(values) - set(cls.__dataclass_fields__):
            raise ValueError("Unknown gate setting")
        cfg = cls(**values)
        if (len(cfg.roi) != 4 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in cfg.roi)
                or not all(0 <= v <= 1 for v in cfg.roi) or cfg.roi[2] < .1 or cfg.roi[3] < .1
                or cfg.roi[0] + cfg.roi[2] > 1.001 or cfg.roi[1] + cfg.roi[3] > 1.001):
            raise ValueError("ROI must fit inside the frame (x,y,width,height in 0..1)")
        if (not isinstance(cfg.line, (int, float)) or not math.isfinite(cfg.line)
                or not cfg.roi[0] < cfg.line < cfg.roi[0] + cfg.roi[2]
                or cfg.direction not in ("right_to_left", "left_to_right")
                or not isinstance(cfg.threshold, (int, float)) or not 5 <= cfg.threshold <= 120
                or type(cfg.min_area) is not int or not 30 <= cfg.min_area <= 20000
                or type(cfg.calibration_frames) is not int or not 3 <= cfg.calibration_frames <= 60
                or type(cfg.max_missed) is not int or not 1 <= cfg.max_missed <= 15
                or type(cfg.crop_padding) is not int or not 0 <= cfg.crop_padding <= 20
                or type(cfg.content_inset) is not int or not 0 <= cfg.content_inset <= 8
                or type(cfg.crossing_grace_ms) is not int or not 100 <= cfg.crossing_grace_ms <= 2000):
            raise ValueError("Invalid gate thresholds or direction")
        return cfg


def components(mask, minimum, with_spans=False):
    """Connected components over row spans (4-connectivity), bounded by 320x180."""
    parent, spans = [], []
    previous = []
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for y, row in enumerate(mask):
        edges = np.diff(np.pad(row.astype(np.int8), (1, 1)))
        current = []
        for left, right in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            label = len(parent)
            parent.append(label)
            spans.append((y, int(left), int(right)))
            for pleft, pright, plabel in previous:
                if left < pright and right > pleft:
                    parent[find(label)] = find(plabel)
            current.append((left, right, label))
        previous = current
    groups = {}
    for i, (y, left, right) in enumerate(spans):
        key = find(i)
        box = groups.setdefault(key, {"x": left, "y": y, "x2": right, "y2": y + 1, "area": 0})
        box["x"], box["y"] = min(box["x"], left), min(box["y"], y)
        box["x2"], box["y2"] = max(box["x2"], right), max(box["y2"], y + 1)
        box["area"] += right - left
        if with_spans:
            box.setdefault('spans', []).append((y, left, right))
    return [b for b in groups.values() if b["area"] >= minimum and (with_spans or (b["x2"] - b["x"] >= 6 and b["y2"] - b["y"] >= 10))]


def fill_holes(mask):
    """Fill enclosed background regions, preserving background connected to the image border."""
    result = mask.copy()
    height, width = mask.shape
    for region in components(~mask, 1, with_spans=True):
        if region['x'] == 0 or region['y'] == 0 or region['x2'] == width or region['y2'] == height:
            continue
        for y, left, right in region['spans']:
            result[y, left:right] = True
    return result


class VideoGate:
    width, height = 320, 180

    def __init__(self, config=None, quality=None):
        self.config = config or GateConfig()
        self.quality = quality or QualityConfig()
        self.background = None
        self.background_rgb = None
        self.calibration = []
        self.tracks = []
        self.next_id = 1
        self.frames = 0
        self.events = 0
        self.uninspected = 0
        self.previous_pts = None
        self.stage_counts = {name: 0 for name in ('grayscale', 'background_difference', 'threshold', 'morphology', 'hole_filling', 'components', 'quality')}
        self.stage_ms = {}
        self.peak_tracks = 0
        self.debug_arrays = None
        self.debug_pts = 0

    def _result(self, image, gray, difference, raw, clean, pts_ns, **values):
        self.debug_arrays = (image.resize((self.width, self.height)), gray, difference, raw, clean)
        self.debug_pts = int(pts_ns)
        values.update(frame_number=self.frames, stage_counts=dict(self.stage_counts), stage_ms=dict(self.stage_ms),
                      peak_simultaneous_tracks=self.peak_tracks)
        return values

    def debug_images(self):
        """Small, exact analysis-frame images, independent of live preview timing."""
        if self.debug_arrays is None:
            return {}
        source, gray, difference, raw, clean = self.debug_arrays
        def image(values):
            return Image.fromarray(np.clip(values, 0, 255).astype(np.uint8)).convert('RGB')
        objects = source.copy()
        draw = ImageDraw.Draw(objects)
        rx, ry, rw, rh = self.config.roi
        draw.rectangle((int(rx * self.width), int(ry * self.height), int((rx + rw) * self.width), int((ry + rh) * self.height)), outline='#5aafff', width=1)
        for track in self.tracks:
            if track['missed']:
                continue
            box = track['box']
            color = '#65edaa' if track['triggered'] else '#ffdc70'
            draw.rectangle((box['x'], box['y'], box['x2'], box['y2']), outline=color, width=2)
            draw.text((box['x'] + 2, max(0, box['y'] - 12)), '#' + str(track['id']), fill=color)
        return {'source': source, 'gray': image(gray), 'difference': image(difference),
                'threshold': image(raw * 255), 'morphology': image(clean * 255), 'objects': objects}

    def recalibrate(self):
        if self.tracks:
            raise ValueError("Wait until products leave the view before calibrating")
        self.background, self.background_rgb, self.calibration = None, None, []

    def process(self, image, pts_ns):
        started = time.perf_counter()
        step_seconds = max(.001, (pts_ns - self.previous_pts) / 1e9) if self.previous_pts is not None else .1
        self.previous_pts = pts_ns
        self.frames += 1
        small = image.resize((self.width, self.height)).convert("RGB")
        rgb = np.asarray(small, dtype=np.float32)
        gray = np.asarray(small.convert("L"), dtype=np.float32)
        self.stage_counts['grayscale'] += 1
        self.stage_ms = {'grayscale': (time.perf_counter() - started) * 1000}
        if self.background is None:
            self.calibration.append(rgb)
            if len(self.calibration) >= self.config.calibration_frames:
                self.background_rgb = np.median(self.calibration, axis=0)
                self.background = np.asarray(Image.fromarray(self.background_rgb.astype(np.uint8)).convert("L"), dtype=np.float32)
                self.calibration.clear()
            blank = np.zeros_like(gray)
            return self._result(image, gray, blank, blank, blank, pts_ns,
                    calibrating=self.background is None, calibration_count=len(self.calibration), tracks=[], events=[])
        stage_start = time.perf_counter()
        color_difference = rgb - self.background_rgb
        # Compensate uniform exposure/channel shifts without losing equal-luma colored products.
        channel_shift = np.median(color_difference.reshape(-1, 3), axis=0)
        shift = float(np.median(gray - self.background))
        difference = np.max(np.abs(color_difference - channel_shift), axis=2)
        self.stage_counts['background_difference'] += 1
        self.stage_ms['background_difference'] = (time.perf_counter() - stage_start) * 1000
        stage_start = time.perf_counter()
        mask = difference > self.config.threshold
        raw_mask = mask.copy()
        self.stage_counts['threshold'] += 1
        self.stage_ms['threshold'] = (time.perf_counter() - stage_start) * 1000
        stage_start = time.perf_counter()
        binary = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
        mask = np.asarray(binary) > 0
        mask = fill_holes(mask)
        self.stage_counts['hole_filling'] += 1
        self.stage_counts['morphology'] += 1
        self.stage_ms['morphology'] = (time.perf_counter() - stage_start) * 1000
        stage_start = time.perf_counter()
        blobs = sorted(components(mask, self.config.min_area), key=lambda b: -b["area"])[:16]
        if self.config.content_inset:
            # Framed-photo input: keep the external carrier contour. At the image edge
            # the carrier is open, so its enclosed photo content may be a separate blob.
            # This is contour selection, not grouping products or generating review events.
            blobs = [b for b in blobs if not any(other is not b and
                other['x'] <= b['x'] and other['y'] <= b['y'] and
                other['x2'] >= b['x2'] and other['y2'] >= b['y2'] and
                (other['x2']-other['x'])*(other['y2']-other['y']) > (b['x2']-b['x'])*(b['y2']-b['y'])
                for other in blobs)]
        self.stage_counts['components'] += 1
        self.stage_ms['components'] = (time.perf_counter() - stage_start) * 1000
        stage_start = time.perf_counter()
        detected_components = len(blobs)
        events = []
        matches = associate(self.tracks, blobs, pts_ns)
        rx, ry, rw, rh = self.config.roi
        roi = (int(rx * self.width), int(ry * self.height), int((rx + rw) * self.width), int((ry + rh) * self.height))
        for track in self.tracks:
            track["hits"].append(0)
            track["missed"] += 1
        # Global assignment uses predicted position at the source PTS, IoU, shape and lane proximity.
        line = self.config.line * self.width
        for blob_index, blob in enumerate(blobs):
            cx, cy = (blob["x"] + blob["x2"]) / 2, (blob["y"] + blob["y2"]) / 2
            track = matches.get(blob_index)
            if track is None:
                track = {"id": self.next_id, "cx": cx, "cy": cy, "vx": 0., "hits": deque([0] * 5, maxlen=5),
                         "missed": 0, "triggered": False, "best": None, "entered": False,
                         "last_pts_ns": int(pts_ns), "confirmed_once": False, "crossing_pts_ns": None,
                         "approach_seen": cx > line if self.config.direction == "right_to_left" else cx < line}
                self.next_id += 1
                self.tracks.append(track)
            previous_x = track["cx"]
            track["hits"][-1] = 1
            elapsed = max(.001, (pts_ns - track["last_pts_ns"]) / 1e9)
            measured_velocity = max(-3200., min(3200., (cx - previous_x) / elapsed))
            track.update(vx=.35 * track["vx"] + .65 * measured_velocity, cx=cx, cy=cy, missed=0, box=blob, last_pts_ns=int(pts_ns))
            track["confirmed_once"] = track["confirmed_once"] or sum(track["hits"]) >= 4
            overlap = int(mask[max(blob["y"], roi[1]):min(blob["y2"], roi[3]),
                               max(blob["x"], roi[0]):min(blob["x2"], roi[2])].sum()) if (
                min(blob["x2"], roi[2]) > max(blob["x"], roi[0]) and min(blob["y2"], roi[3]) > max(blob["y"], roi[1])) else 0
            fraction = min(1., overlap / max(1, blob["area"]))
            track["roi_overlap"] = fraction
            if fraction > 0:
                track["entered"] = True
            if not track["triggered"] and fraction > 0:
                sx, sy = image.width / self.width, image.height / self.height
                padding, inset = self.config.crop_padding, self.config.content_inset
                content_box = (max(0, int((blob["x"] + inset) * sx)), max(0, int((blob["y"] + inset) * sy)),
                               min(image.width, int((blob["x2"] - inset) * sx)), min(image.height, int((blob["y2"] - inset) * sy)))
                if content_box[2] <= content_box[0] or content_box[3] <= content_box[1]:
                    continue
                box = (max(0, content_box[0] - int(padding * sx)), max(0, content_box[1] - int(padding * sy)),
                       min(image.width, content_box[2] + int(padding * sx)), min(image.height, content_box[3] + int(padding * sy)))
                crop, content = image.crop(box), image.crop(content_box)
                self.stage_counts['quality'] += 1
                quality = assess(content, self.quality, clipped=blob["x"] <= 1 or blob["x2"] >= self.width - 1 or
                                 blob["y"] <= 1 or blob["y2"] >= self.height - 1)
                quality["assessment_region"] = "foreground content interior, excluding configured carrier border and inference padding"
                if fraction < .8:
                    quality["acceptable"] = False
                    quality["reasons"].append("outside_roi")
                candidate = {"image": crop, "quality": quality, "pts_ns": int(pts_ns), "crop_box": box,
                             "frame_number": self.frames, "roi_qualified": fraction >= .8}
                best = track["best"]
                if best is None or pts_ns - best['pts_ns'] > 800_000_000 or (candidate["roi_qualified"], quality["acceptable"], quality["score"]) > (
                        best["roi_qualified"], best["quality"]["acceptable"], best["quality"]["score"]):
                    track["best"] = candidate
            crossed = (previous_x > line >= cx) if self.config.direction == "right_to_left" else (previous_x < line <= cx)
            if crossed and track["approach_seen"] and track["crossing_pts_ns"] is None:
                track["crossing_pts_ns"] = int(pts_ns)
            if (cx > line if self.config.direction == "right_to_left" else cx < line):
                track["approach_seen"] = True
            pending = track["crossing_pts_ns"] is not None and 0 <= pts_ns - track["crossing_pts_ns"] <= self.config.crossing_grace_ms * 1_000_000
            if not track["triggered"] and pending and sum(track["hits"]) >= 4 and fraction >= .8:
                track["triggered"] = True
                self.events += 1
                best = track.pop("best")
                if best is None:
                    self.stage_counts['quality'] += 1
                    best = {"image": image.copy(), "quality": assess(image, self.quality, clipped=True),
                            "pts_ns": int(pts_ns), "crop_box": (0, 0, image.width, image.height), "frame_number": self.frames}
                events.append(dict(best, track_id=track["id"], gate={"crossed": True, "confirmed_frames": sum(track["hits"]),
                    "confirmation_window": 5, "roi_overlap": fraction, "crossing_pts_ns": track["crossing_pts_ns"]}, trigger_pts_ns=int(pts_ns)))
        for track in list(self.tracks):
            if track["missed"] > self.config.max_missed:
                if track["entered"] and not track["triggered"]:
                    self.uninspected += 1
                    best = track["best"]
                    if best:
                        best["quality"]["acceptable"] = False
                        best["quality"]["reasons"].append("roi_crossing_unconfirmed")
                        events.append(dict(best, track_id=track["id"], gate={"crossed": False}, trigger_pts_ns=int(pts_ns)))
                self.tracks.remove(track)
        if not blobs and not np.any(mask):
            self.background = .995 * self.background + .005 * gray
            self.background_rgb = .995 * self.background_rgb + .005 * rgb
        public = [{"id": t["id"], "box": predicted_box(t, pts_ns), "confirmed_frames": sum(t["hits"]),
                   "confirmed": t['confirmed_once'], "observed": t['missed'] == 0, "missed_frames": t['missed'],
                   "triggered": t["triggered"], "roi_overlap": t["roi_overlap"],
                   "velocity_x": t["vx"]} for t in self.tracks if t["missed"] == 0 or t['confirmed_once']]
        self.peak_tracks = max(self.peak_tracks, sum(t['confirmed'] and t['observed'] for t in public))
        self.stage_ms['tracking_quality'] = (time.perf_counter() - stage_start) * 1000
        self.stage_ms['total'] = (time.perf_counter() - started) * 1000
        return self._result(image, gray, difference, raw_mask, mask, pts_ns,
                calibrating=False, tracks=public, events=events, detected_components=detected_components,
                brightness_shift=shift, products_triggered=self.events, unconfirmed_exits=self.uninspected)

    def settings(self):
        return {"gate": asdict(self.config), "quality": asdict(self.quality)}
