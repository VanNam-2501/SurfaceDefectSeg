"""Pixel-rendered photo conveyor. Labels/schedules never enter the tracker."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from PIL import Image, ImageDraw, ImageOps


@dataclass(frozen=True)
class PhotoSettings:
    simultaneous: int = 2
    speed: int = 240
    rounds: int = 1

    @classmethod
    def parse(cls, value=None):
        value = value or {}
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError('Thiết lập băng chuyền ảnh không hợp lệ.')
        cfg = cls(**value)
        if (type(cfg.simultaneous) is not int or cfg.simultaneous not in (1, 2, 3)
                or type(cfg.speed) is not int or not 120 <= cfg.speed <= 600
                or type(cfg.rounds) is not int or not 1 <= cfg.rounds <= 3):
            raise ValueError('Chọn 1–3 sản phẩm, 120–600 px/s và 1–3 vòng.')
        return cfg


class PhotoConveyor:
    # Keep the 960x540 layout/speed seen by the operator, but retain at least
    # 256 source pixels per photo even in three-lane mode. Preview pixels must
    # never become the input to quality checks or inference.
    scale = 2
    preview_size = (960, 540)
    width, height, fps = 1920, 1080, 30
    border = 30
    revision = 'photo-source-v3.4'

    def __init__(self, root, image_ids=None, settings=None, direction='right_to_left'):
        self.root = Path(root).resolve()
        self.settings = PhotoSettings.parse(settings)
        self.direction = direction
        if direction not in ('right_to_left', 'left_to_right'):
            raise ValueError('Invalid direction')
        if not isinstance(image_ids, (list, type(None))) or len(image_ids or []) > 50:
            raise ValueError('Mỗi lượt chọn tối đa 50 ảnh.')
        paths = []
        if image_ids:
            for image_id in image_ids:
                if not isinstance(image_id, str) or not re.fullmatch(r'[0-9a-f]{32}', image_id):
                    raise ValueError('Mã ảnh băng chuyền không hợp lệ.')
                paths.append(self.root / 'artifacts/inspection/conveyor_inputs' / (image_id + '.jpg'))
        else:
            manifest_path = self.root / 'demo_samples/manifest.json'
            if not manifest_path.is_file():
                raise ValueError('Chọn ảnh từ máy; bộ ảnh mẫu không có trong gói source.')
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            paths = [self.root / row['path'] for row in manifest['cases']]
        self.tile_size = {1: 220, 2: 190, 3: 128}[self.settings.simultaneous] * self.scale
        self.render_speed = self.settings.speed * self.scale
        self.tiles = []
        for path in paths:
            if not path.resolve().is_relative_to(self.root) or not path.is_file():
                raise ValueError('Ảnh băng chuyền không còn khả dụng.')
            with Image.open(path) as source:
                # Keep all photo content; padding is part of the synthetic source.
                self.tiles.append(ImageOps.pad(ImageOps.exif_transpose(source).convert('RGB'),
                    (self.tile_size, self.tile_size), method=Image.Resampling.BILINEAR, color=(230, 230, 230)))
        if not self.tiles:
            raise ValueError('Cần ít nhất một ảnh.')
        items = list(range(len(self.tiles))) * self.settings.rounds
        lanes = self.settings.simultaneous
        self.groups = [items[i:i + lanes] for i in range(0, len(items), lanes)]
        self.travel_seconds = (self.width + self.tile_size + 60 * self.scale) / self.render_speed
        self.initial_seconds = 1.5
        self.duration = self.initial_seconds + len(self.groups) * self.travel_seconds + .7
        self.frame_count = math.ceil(self.duration * self.fps)
        self.positions = tuple(y * self.scale for y in
            {1: (155,), 2: (60, 290), 3: (33, 206, 379)}[self.settings.simultaneous])
        self.expected_products = len(items)  # Reference only; never passed into VideoGate.
        # Draw the original belt once, then scale its geometry along with the
        # photos. The timing and relative product size stay identical.
        belt = Image.new('RGB', self.preview_size, (199, 211, 220))
        draw = ImageDraw.Draw(belt)
        draw.rectangle((0, 30, 959, 510), fill=(113, 128, 139))
        draw.rectangle((0, 20, 959, 34), fill=(62, 85, 104))
        draw.rectangle((0, 506, 959, 522), fill=(62, 85, 104))
        for x in range(15, 960, 64):
            draw.rounded_rectangle((x, 525, x + 45, 539), radius=4, fill=(163, 184, 201))
        self.background = belt.resize((self.width, self.height), Image.Resampling.NEAREST)

    def render(self, frame_number):
        image = self.background.copy()
        elapsed = frame_number / self.fps - self.initial_seconds
        if elapsed < 0:
            return image
        group_index = int(elapsed / self.travel_seconds)
        if group_index >= len(self.groups):
            return image
        x = self.width - (elapsed % self.travel_seconds) * self.render_speed
        positions = self.positions
        for lane, item in enumerate(self.groups[group_index]):
            y = positions[lane]
            px = x if self.direction == 'right_to_left' else self.width - self.tile_size - x
            px, y = round(px), round(y)
            ImageDraw.Draw(image).rectangle((px-self.border, y-self.border, px+self.tile_size+self.border-1, y+self.tile_size+self.border-1), fill=(30, 44, 58))
            image.paste(self.tiles[item], (px, y))
        return image

    def gate_settings(self):
        return {"content_inset": self.border * 320 // self.width, "crop_padding": 0}

    def description(self):
        return {'image_count': len(self.tiles), 'expected_products': self.expected_products,
                'revision': self.revision, 'source_size': [self.width, self.height],
                'preview_size': list(self.preview_size), 'photo_content_pixels': self.tile_size,
                'simultaneous': self.settings.simultaneous, 'carrier_border_pixels': self.border,
                'speed_pixels_per_second': self.settings.speed, 'speed_coordinate_width': self.preview_size[0],
                'source_speed_pixels_per_second': self.render_speed, 'rounds': self.settings.rounds,
                'seconds': round(self.duration, 2), 'fps': self.fps,
                'scope': 'Synthetic photos rendered on CPU; actual pixels go through the gate and TensorRT. Expected count is a reference, not detector input.'}
