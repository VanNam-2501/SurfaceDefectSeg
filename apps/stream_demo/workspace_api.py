"""Video/photo inspection and persistent history HTTP routes."""
from __future__ import annotations
import io
import json
import mimetypes
from pathlib import Path
import re
import socket
import uuid
from urllib.parse import unquote
from PIL import Image, ImageOps
try:
    from .inspection_service import InspectionService
    from .video_session import VideoSession
    from .process_metrics import ProcessMetrics
    from .performance_monitor import PerformanceMonitor
except ImportError:
    from inspection_service import InspectionService
    from video_session import VideoSession
    from process_metrics import ProcessMetrics
    from performance_monitor import PerformanceMonitor


class WorkspaceAPI:
    def __init__(self, root, resident, distro, wsl_root, artifact):
        self.root = Path(root)
        self.service = InspectionService(root, resident)
        self.video = VideoSession(root, resident, self.service, distro, wsl_root, artifact)
        self.metrics = ProcessMetrics()
        self.performance = PerformanceMonitor(self.video, self.service.store, self.metrics)

    def get(self, h, route, query):
        def value(name, default=""):
            return query.get(name, [default])[0]
        filters = {key: value(key) for key in ("session_id", "decision", "class_name", "search")}
        if route in ("/", "/video", "/images", "/history", "/performance"):
            h._send(200, "text/html; charset=utf-8", Path(__file__).with_name("workspace.html").read_bytes())
        elif route in ("/workspace.css", "/workspace.js", "/workspace_diagnostics.js", "/performance.js"):
            path = Path(__file__).with_name(route[1:])
            h._send(200, (mimetypes.guess_type(path.name)[0] or "text/plain") + "; charset=utf-8", path.read_bytes())
        elif route == "/api/history":
            h._json(200, self.service.store.query(**filters, limit=int(value("limit", "50")), offset=int(value("offset", "0"))))
        elif route == "/api/history/summary":
            h._json(200, self.service.store.summary())
        elif route == "/api/history/export.csv":
            h._send(200, "text/csv; charset=utf-8", self.service.store.export_csv(**filters))
        elif route.startswith("/api/history/"):
            match = re.fullmatch(r"/api/history/([0-9a-f]{32})(/image)?", route)
            row = self.service.store.get(match[1]) if match else None
            if not row:
                h._json(404, {"error": "Không tìm thấy bản ghi."})
            elif match[2]:
                path = (self.root / row["image_path"]).resolve()
                if not path.is_relative_to(self.root.resolve()) or not path.is_file():
                    h._json(404, {"error": "Ảnh lưu trữ không còn khả dụng."})
                else:
                    if value('thumb') == '1':
                        folder = self.root / 'artifacts/inspection/thumbnails'
                        folder.mkdir(parents=True, exist_ok=True)
                        thumbnail = folder / (row['id'] + '.jpg')
                        if not thumbnail.is_file():
                            with Image.open(path) as source:
                                if source.format == 'JPEG':
                                    source.draft('RGB', (256, 256))
                                image = ImageOps.exif_transpose(source).convert('RGB')
                                image.thumbnail((160, 160))
                                stream = io.BytesIO()
                                image.save(stream, format='JPEG', quality=82)
                                thumbnail.write_bytes(stream.getvalue())
                        h._send(200, 'image/jpeg', thumbnail.read_bytes())
                    else:
                        h._send(200, mimetypes.guess_type(path.name)[0] or "image/jpeg", path.read_bytes())
            else:
                h._json(200, row)
        elif route == "/api/video":
            h._json(200, self.video.snapshot())
        elif route == '/api/performance':
            h._json(200, self.performance.snapshot())
        elif route == '/api/conveyor/selection':
            path = self.root / 'artifacts/inspection/conveyor_selection.json'
            items = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else []
            folder = self.root / 'artifacts/inspection/conveyor_inputs'
            h._json(200, {'items': [dict(item, url='/api/conveyor/images/' + item['image_id']) for item in items
                if re.fullmatch(r'[0-9a-f]{32}', item.get('image_id', '')) and (folder / (item['image_id'] + '.jpg')).is_file()]})
        elif route.startswith('/api/video/debug/'):
            match = re.fullmatch(r'/api/video/debug/(source|gray|difference|threshold|morphology|objects|montage)\.jpg', route)
            jpeg = self.video.debug_frame(value('session_id'), match[1], int(value('frame', '0'))) if match else None
            if jpeg:
                h._send(200, 'image/jpeg', jpeg)
            else:
                h._json(404, {'error': 'Khung phân tích đã hết bộ nhớ đệm; hãy lấy khung mới.'})
        elif route.startswith('/api/conveyor/images/'):
            image_id = route.rsplit('/', 1)[-1]
            path = self.root / 'artifacts/inspection/conveyor_inputs' / (image_id + '.jpg')
            if re.fullmatch(r'[0-9a-f]{32}', image_id) and path.is_file():
                h._send(200, 'image/jpeg', path.read_bytes())
            else:
                h._json(404, {'error': 'Ảnh băng chuyền không còn khả dụng.'})
        elif route == '/api/video/stream.mjpg':
            self.stream(h, value('session_id'))
        elif route == "/api/video/frame":
            jpeg = self.video.frame(int(value("frame", "0")))
            h._send(200, "image/jpeg", jpeg) if jpeg else h._json(404, {"error": "Chưa có frame."})
        else:
            return False
        return True

    def stream(self, h, session_id):
        with self.video.condition:
            if session_id != self.video.state.get('id'):
                h._json(404, {'error': 'Phiên video không còn hiện hành.'})
                return
            if self.video.stream_clients >= 4:
                h._json(429, {'error': 'Đã có bốn cửa sổ preview; hãy đóng bớt cửa sổ.'})
                return
            self.video.stream_clients += 1
        try:
            h.send_response(200)
            h.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            h.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            h.send_header('Connection', 'close')
            h.end_headers()
            h.close_connection = True
            h.connection.settimeout(5)
            h.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            h.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
            last = 0
            while True:
                with self.video.condition:
                    self.video.condition.wait_for(lambda: self.video.state.get('id') != session_id or
                        self.video.state.get('frame_number', 0) != last or self.video.capture_done, timeout=1)
                    if self.video.state.get('id') != session_id:
                        break
                    number = self.video.state.get('frame_number', 0)
                    jpeg = self.video.jpeg
                    done = self.video.capture_done
                if jpeg and number != last:
                    h.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
                        str(len(jpeg)).encode() + b'\r\nX-Frame-Number: ' + str(number).encode() + b'\r\n\r\n' + jpeg + b'\r\n')
                    h.wfile.flush()
                    last = number
                elif done:
                    break
            h.wfile.write(b'--frame--\r\n')
        except (BrokenPipeError, ConnectionError, OSError):
            pass
        finally:
            with self.video.condition:
                self.video.stream_clients -= 1

    def post(self, h, route):
        if route == '/api/conveyor/selection':
            length = h._content_length(64 * 1024)
            payload = json.loads(h.rfile.read(length)) if length else {}
            items = payload.get('items') if isinstance(payload, dict) else None
            if not isinstance(items, list) or len(items) > 50:
                raise ValueError('Danh sách ảnh cần có từ 0 đến 50 ảnh.')
            folder = self.root / 'artifacts/inspection/conveyor_inputs'
            selected, seen = [], set()
            for item in items:
                image_id = item.get('image_id', '') if isinstance(item, dict) else ''
                if (not isinstance(image_id, str) or not re.fullmatch(r'[0-9a-f]{32}', image_id)
                        or image_id in seen or not (folder / (image_id + '.jpg')).is_file()):
                    raise ValueError('Ảnh đã chọn không hợp lệ hoặc không còn khả dụng.')
                seen.add(image_id)
                selected.append({'image_id': image_id, 'name': str(item.get('name', 'Ảnh băng chuyền'))[:150]})
            folder.parent.mkdir(parents=True, exist_ok=True)
            target = folder.parent / 'conveyor_selection.json'
            temporary = folder.parent / ('selection_' + uuid.uuid4().hex + '.part')
            try:
                temporary.write_text(json.dumps(selected, ensure_ascii=False), encoding='utf-8')
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            h._json(200, {'saved': len(selected)})
        elif route == '/api/conveyor/images':
            length = h._content_length(30 * 1024 * 1024)
            if not length:
                raise ValueError('Ảnh không được quá 30 MB.')
            raw = h.rfile.read(length)
            if len(raw) != length:
                raise ValueError('Ảnh tải lên chưa đầy đủ.')
            with Image.open(io.BytesIO(raw)) as opened:
                if opened.format not in ('JPEG', 'PNG', 'WEBP', 'BMP') or opened.width * opened.height > 16_000_000:
                    raise ValueError('Chọn JPG, PNG, WebP hoặc BMP, tối đa 16 triệu pixel.')
                image = ImageOps.exif_transpose(opened).convert('RGB')
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            image_id = uuid.uuid4().hex
            folder = self.root / 'artifacts/inspection/conveyor_inputs'
            folder.mkdir(parents=True, exist_ok=True)
            image.save(folder / (image_id + '.jpg'), quality=95)
            name = unquote(h.headers.get('X-Image-Name', 'Ảnh băng chuyền'))[:150]
            h._json(201, {'image_id': image_id, 'name': name, 'url': '/api/conveyor/images/' + image_id})
        elif route == "/api/video/upload":
            length = h._content_length(512 * 1024 * 1024)
            if not length:
                raise ValueError("Video cần có kích thước từ 1 byte đến 512 MB.")
            upload_id = uuid.uuid4().hex
            folder = self.root / "artifacts/inspection/uploads"
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f"{upload_id}.video"
            temporary = target.with_suffix(".part")
            try:
                with temporary.open("wb") as stream:
                    remaining = length
                    while remaining:
                        chunk = h.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError("Kết nối bị ngắt khi tải video.")
                        stream.write(chunk)
                        remaining -= len(chunk)
                temporary.replace(target)
                target.with_suffix('.json').write_text(json.dumps({
                    'name': unquote(h.headers.get('X-Video-Name', 'Video tải lên'))[:150]}, ensure_ascii=False), encoding='utf-8')
            finally:
                temporary.unlink(missing_ok=True)
            h._json(201, {"upload_id": upload_id, "bytes": length})
        elif route == "/api/photo":
            length = h._content_length(30 * 1024 * 1024)
            if not length:
                raise ValueError("Ảnh không được quá 30 MB.")
            raw = h.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Ảnh tải lên chưa đầy đủ.")
            with Image.open(io.BytesIO(raw)) as image:
                if image.width * image.height > 16_000_000:
                    raise ValueError("Ảnh vượt quá 16 triệu pixel.")
                suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp"}.get(image.format)
                if not suffix:
                    raise ValueError("Chọn ảnh JPG, PNG, WebP hoặc BMP.")
                image.verify()
            name = unquote(h.headers.get("X-Image-Name", "Ảnh tải lên"))[:150]
            session_id = self.service.store.session("photo", name)
            folder = self.root / "artifacts/inspection/photos"
            folder.mkdir(parents=True, exist_ok=True)
            image_path = folder / (session_id + suffix)
            image_path.write_bytes(raw)
            row = self.service.submit(image_path, "photo:" + session_id, session_id, 1,
                                      {"source_kind": "photo", "source_label": name})
            h._json(202, row)
        elif route in ("/api/video/start", "/api/video/stop", "/api/video/calibrate"):
            length = h._content_length(32 * 1024)
            payload = json.loads(h.rfile.read(length)) if length else {}
            if route.endswith("/start"):
                if not isinstance(payload, dict):
                    raise ValueError("Invalid video settings")
                h._json(201, self.video.start(payload))
            elif route.endswith("/stop"):
                h._json(200, self.video.stop())
            else:
                self.video.recalibrate()
                h._json(200, {"status": "calibrating"})
        else:
            return False
        return True

    def close(self):
        self.performance.close()
        self.video.stop()
        self.service.close()
        close = getattr(self.service.resident, 'close', None)
        if close:
            close()
