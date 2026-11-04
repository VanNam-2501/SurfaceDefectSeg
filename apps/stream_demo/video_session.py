"""DeepStream GPU video capture -> fixed-camera gate -> resident TensorRT."""
from __future__ import annotations
import io
from dataclasses import replace
import json
from collections import deque
import re
from pathlib import Path
import subprocess
import threading
import os
import time
from urllib.parse import urlsplit, urlunsplit
from PIL import Image, ImageDraw
try:
    from .image_quality import QualityConfig
    from .video_tracking import GateConfig, VideoGate
    from .photo_conveyor import PhotoConveyor
    from .runtime_backend import native_runtime, host_command
except ImportError:
    from image_quality import QualityConfig
    from video_tracking import GateConfig, VideoGate
    from photo_conveyor import PhotoConveyor
    from runtime_backend import native_runtime, host_command


def redact_source(value):
    parts = urlsplit(value)
    host = parts.hostname or "camera"
    if ":" in host:
        host = "[" + host + "]"
    return urlunsplit((parts.scheme, host + (f":{parts.port}" if parts.port else ""), parts.path, "", ""))


class VideoSession:
    def __init__(self, root, resident, inspections, distro, wsl_root, artifact):
        self.root, self.resident, self.inspections = Path(root), resident, inspections
        self.distro, self.wsl_root, self.artifact = distro, wsl_root, artifact
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.analysis_lock = threading.RLock()
        self.latest_packet = None
        self.capture_done = False
        self.analysis_error = ''
        self.stream_clients = 0
        self.stop_event = threading.Event()
        self.process, self.thread = None, None
        self.jpeg = None
        self.frame_cache = deque(maxlen=12)
        self.debug_cache = deque(maxlen=8)
        self.gate = None
        self.state = {"status": "idle", "frames_received": 0, "tracks": [], "events": 0}

    def start(self, spec):
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError("Stop the active video session before opening another source")
            kind = spec.get("kind", "demo")
            if (not self.wsl_root and kind != 'photos') or self.resident.status()["state"] != "ready":
                raise ValueError("Wait for the TensorRT GPU worker to be ready")
            gate, quality = GateConfig.parse(spec.get("gate")), QualityConfig.parse(spec.get("quality"))
            photo_source = None
            if kind == "demo":
                video = self.root / "demo_samples/conveyor_replay.mp4"
                label = "Video băng chuyền tổng hợp · ảnh validation"
            elif kind == "file":
                upload = str(spec.get("upload_id", ""))
                if not re.fullmatch(r"[0-9a-f]{32}", upload):
                    raise ValueError("Invalid video upload")
                video = self.root / "artifacts/inspection/uploads" / f"{upload}.video"
                label = "Video tải lên"
                if video.with_suffix('.json').is_file():
                    label += ' · ' + json.loads(video.with_suffix('.json').read_text(encoding='utf-8')).get('name', '')
            elif kind == "rtsp":
                uri = str(spec.get("uri", "")).strip()
                if len(uri) > 2048 or any(c in uri for c in "\r\n\x00") or urlsplit(uri).scheme not in ("rtsp", "rtsps"):
                    raise ValueError("Enter a valid rtsp:// or rtsps:// source")
                if not urlsplit(uri).hostname:
                    raise ValueError("RTSP host is missing")
                label = redact_source(uri)
            elif kind == 'photos':
                photo_source = PhotoConveyor(self.root, spec.get('image_ids'), spec.get('photos'), gate.direction)
                gate = replace(gate, **photo_source.gate_settings())
                label = 'Băng chuyền ảnh · ' + str(len(photo_source.tiles)) + ' ảnh · ' + str(photo_source.settings.simultaneous) + ' đồng thời'
            else:
                raise ValueError("Unsupported video source")
            if kind not in ('rtsp', 'photos'):
                if not video.is_file():
                    raise ValueError("Video is unavailable; upload a video or use the full handover bundle")
                uri = "file:///workspace/" + video.relative_to(self.root).as_posix()
            self.gate = VideoGate(gate, quality)
            session_config = self.gate.settings()
            if photo_source:
                session_config['source'] = photo_source.description()
            session_id = self.inspections.store.session(kind, label, session_config)
            self.stop_event = threading.Event()
            self.jpeg = None
            self.frame_cache.clear()
            self.debug_cache.clear()
            self.latest_packet = None
            self.capture_done = False
            self.analysis_error = ''
            self.state = {"id": session_id, "status": "starting", "kind": kind, "source": label,
                          "frames_received": 0, "frame_number": 0, "analysis_steps": 0,
                          "tracks": [], "events": 0, "settings": self.gate.settings(),
                          "backend": "DeepStream nvurisrcbin/NVDEC + CPU gate + resident TensorRT",
                          "scope": "Fixed camera, separated products, empty-belt calibration; replay is not a factory test"}
            if photo_source:
                self.state.update(backend='CPU photo renderer + CPU gate + resident TensorRT', synthetic=photo_source.description(), render_frames=0,
                    preview_width=photo_source.preview_size[0], preview_height=photo_source.preview_size[1])
            self.thread = threading.Thread(target=self._run_photos if photo_source else self._run,
                args=(session_id, photo_source) if photo_source else (session_id, uri), daemon=True, name="inspection-video")
            self.thread.start()
            return dict(self.state)

    @staticmethod
    def _read_exact(stream, size):
        chunks, remaining = [], size
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                raise EOFError("Incomplete GPU capture frame")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _analyse(self, session_id):
        last_started = 0.
        last_debug = 0.
        source_frames = {}
        try:
            while not self.stop_event.is_set():
                with self.condition:
                    while self.latest_packet is None and not self.capture_done and not self.stop_event.is_set():
                        self.condition.wait(.2)
                    if self.stop_event.is_set() or (self.latest_packet is None and self.capture_done):
                        break
                    remaining = .1 - (time.perf_counter() - last_started)
                    if remaining > 0:
                        self.condition.wait(remaining)
                        continue
                    capture_number, pts_ns, pixels = self.latest_packet
                    self.latest_packet = None
                last_started = time.perf_counter()
                if isinstance(pixels, Image.Image):
                    image = pixels  # Full source frame; the MJPEG preview is separate.
                else:
                    with Image.open(io.BytesIO(pixels)) as opened:
                        image = opened.convert('RGB')
                with self.analysis_lock:
                    analysis = self.gate.process(image, pts_ns)
                    source_frames[analysis['frame_number']] = capture_number
                    if len(source_frames) > 100:
                        del source_frames[min(source_frames)]
                    debug = self.gate.debug_images() if last_started - last_debug >= .2 else None
                if debug:
                    self._publish_debug(session_id, capture_number, analysis['frame_number'], pts_ns, debug)
                    last_debug = last_started
                events = analysis.pop('events')
                analysis_number = analysis.pop('frame_number')
                with self.condition:
                    self.state.update(status='calibrating' if analysis['calibrating'] else 'running',
                        width=image.width, height=image.height, analysis_frame_number=capture_number,
                        analysis_source_pts_ns=pts_ns, analysis_updated_at=time.time(),
                        analysis_steps=analysis_number, **analysis)
                for event in events:
                    folder = self.root / 'artifacts/inspection/sessions' / session_id
                    folder.mkdir(parents=True, exist_ok=True)
                    image_path = folder / f"track_{event['track_id']:06d}.jpg"
                    event['image'].save(image_path, quality=95)
                    metadata = {'source_pts_ns': event['pts_ns'], 'trigger_pts_ns': event['trigger_pts_ns'],
                        'frame_number': source_frames[event['frame_number']], 'analysis_frame_number': event['frame_number'],
                        'crop_box': event['crop_box'], 'gate': event['gate'],
                        'source_size': [image.width, image.height],
                        'source_kind': self.state['kind'], 'source_label': self.state['source']}
                    if self.state.get('synthetic'):
                        metadata['photo_source_revision'] = self.state['synthetic']['revision']
                    row = self.inspections.submit(image_path, f"video:{session_id}:{event['track_id']}", session_id,
                        event['track_id'], metadata, event['quality'])
                    with self.condition:
                        self.state['events'] += 1
                        self.state['last_inspection_id'] = row['id']
                with self.condition:
                    self.state['host_gate_ms'] = (time.perf_counter() - last_started) * 1000
        except Exception as exc:
            with self.condition:
                self.analysis_error = str(exc)
                self.stop_event.set()
                self.condition.notify_all()

    def _terminate_capture(self, session_id):
        if native_runtime():
            with self.lock:
                if self.process and self.process.poll() is None:
                    self.process.terminate()
            return
        try:
            subprocess.run(['wsl.exe', '-d', self.distro, '-u', 'root', '--', 'bash',
                f'{self.wsl_root}/apps/stream_demo/stop_capture.sh', self.artifact, session_id],
                capture_output=True, timeout=10, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except (OSError, subprocess.SubprocessError):
            pass

    def _publish_debug(self, session_id, capture_number, step, pts_ns, images):
        encoded = {}
        montage = Image.new('RGB', (640, 636), '#10263b')
        draw = ImageDraw.Draw(montage)
        labels = {'source': '1. Analysis source', 'gray': '2. Grayscale', 'difference': '3. Compensated RGB background difference',
                  'threshold': '4. Threshold mask', 'morphology': '5. Open + close + fill enclosed holes', 'objects': '6. Components + tracks + ROI'}
        for index, (name, image) in enumerate(images.items()):
            stream = io.BytesIO()
            image.save(stream, format='JPEG', quality=86)
            encoded[name] = stream.getvalue()
            x, y = index % 2 * 320, index // 2 * 204
            draw.text((x + 7, y + 5), labels[name], fill='#c6dced')
            montage.paste(image, (x, y + 24))
        draw.text((7, 616), f'Source frame {capture_number} | Analysis {step} | PTS {pts_ns / 1e9:.3f}s', fill='#d9e8f3')
        stream = io.BytesIO()
        montage.save(stream, format='JPEG', quality=90)
        encoded['montage'] = stream.getvalue()
        with self.lock:
            self.debug_cache.append((capture_number, encoded))
            self.state['debug'] = {'frame_number': capture_number, 'analysis_step': step, 'pts_ns': pts_ns,
                'urls': {name: f'/api/video/debug/{name}.jpg?session_id={session_id}&frame={capture_number}' for name in encoded}}

    def debug_frame(self, session_id, stage, number):
        with self.lock:
            if session_id != self.state.get('id'):
                return None
            return next((images.get(stage) for frame, images in self.debug_cache if frame == number), None)

    def _publish_packet(self, jpeg, pts_ns, arrivals, *, source_image=None):
        now = time.perf_counter()
        arrivals.append(now)
        with self.condition:
            capture_number = self.state['frames_received'] + 1
            self.jpeg = jpeg
            self.frame_cache.append((capture_number, jpeg))
            self.latest_packet = (capture_number, pts_ns, source_image if source_image is not None else jpeg)
            self.state.update(frames_received=capture_number, frame_number=capture_number,
                source_pts_ns=pts_ns, updated_at=time.time(), preview_fps=(len(arrivals)-1)/(now-arrivals[0]) if len(arrivals)>1 and now>arrivals[0] else 0)
            self.condition.notify_all()

    def _run_photos(self, session_id, source):
        analyser = threading.Thread(target=self._analyse, args=(session_id,), daemon=True, name='photo-analysis-10hz')
        analyser.start()
        arrivals = deque(maxlen=60)
        try:
            started = time.perf_counter()
            index = 0
            while index < source.frame_count and not self.stop_event.is_set():
                wait = started + index / source.fps - time.perf_counter()
                if wait > 0 and self.stop_event.wait(wait):
                    break
                # Skip late render slots instead of playing a growing queue in slow motion.
                index = max(index, int((time.perf_counter() - started) * source.fps))
                if index >= source.frame_count:
                    break
                image = source.render(index)
                stream = io.BytesIO()
                image.resize(source.preview_size, Image.Resampling.BILINEAR).save(stream, format='JPEG', quality=90)
                self._publish_packet(stream.getvalue(), round(index / source.fps * 1e9), arrivals, source_image=image)
                with self.lock:
                    self.state['render_frames'] += 1
                index += 1
            with self.condition:
                self.capture_done = True
                self.condition.notify_all()
            analyser.join(timeout=5)
            if analyser.is_alive():
                raise RuntimeError('Photo conveyor analysis did not drain')
            if self.analysis_error:
                raise RuntimeError(self.analysis_error)
            with self.lock:
                self.state['status'] = 'stopped' if self.stop_event.is_set() else 'completed'
        except Exception as exc:
            with self.lock:
                self.state.update(status='error', error=str(exc))
        finally:
            with self.condition:
                self.capture_done = True
                self.condition.notify_all()
            analyser.join(timeout=2)
            self.inspections.store.end_session(session_id, self.state['status'])

    def _run(self, session_id, uri):
        errors, arrivals = [], deque(maxlen=60)
        process, analyser = None, None
        try:
            command = ([os.environ.get('STRAW_CAPTURE_BINARY', '/opt/straw/capture_probe')] if native_runtime() else
                host_command(self.distro, 'bash', f'{self.wsl_root}/apps/stream_demo/run_capture.sh', self.artifact, session_id))
            process = subprocess.Popen(command,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=dict(os.environ, STRAW_CAPTURE_URI=uri) if native_runtime() else None,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            with self.lock:
                self.process = process
            if not native_runtime():
                process.stdin.write((uri + '\n').encode())
            process.stdin.close()
            def stderr_reader():
                for line in iter(process.stderr.readline, b''):
                    message = line.decode('utf-8', errors='replace').replace(uri, '[source]')
                    message = re.sub(r'rtsps?://\S+', '[RTSP source]', message)[:300]
                    errors.append(message.strip())
                    del errors[:-8]
            threading.Thread(target=stderr_reader, daemon=True).start()
            analyser = threading.Thread(target=self._analyse, args=(session_id,), daemon=True, name='video-analysis-10hz')
            analyser.start()
            while not self.stop_event.is_set():
                line = process.stdout.readline(1024)
                if not line:
                    break
                match = re.fullmatch(rb'STRW (\d+) (\d+)\n', line)
                if not match:
                    continue
                pts_ns, length = map(int, match.groups())
                if not 1 <= length <= 8 * 1024 * 1024:
                    raise ValueError('Invalid GPU JPEG packet size')
                jpeg = self._read_exact(process.stdout, length)
                self._publish_packet(jpeg, pts_ns, arrivals)
            if self.stop_event.is_set():
                self._terminate_capture(session_id)
            code = process.wait(timeout=25)
            with self.condition:
                self.capture_done = True
                self.condition.notify_all()
            analyser.join(timeout=5)
            if analyser.is_alive():
                raise RuntimeError('Video analysis did not drain')
            with self.condition:
                if self.analysis_error:
                    raise RuntimeError(self.analysis_error)
                if self.stop_event.is_set():
                    self.state['status'] = 'stopped'
                elif code or not self.state['frames_received']:
                    self.state['diagnostics'] = errors
                    raise RuntimeError('Không mở được nguồn video. Kiểm tra địa chỉ, codec và kết nối nguồn.')
                else:
                    self.state['status'] = 'completed'
        except Exception as exc:
            with self.condition:
                self.state.update(status='stopped' if self.stop_event.is_set() and not self.analysis_error else 'error',
                                  error=str(exc).replace(uri, '[source]'))
        finally:
            with self.condition:
                self.capture_done = True
                self.condition.notify_all()
            if process and process.poll() is None:
                self._terminate_capture(session_id)
                if process.poll() is None:
                    process.terminate()
            if analyser:
                analyser.join(timeout=2)
            try:
                (self.root/'artifacts'/self.artifact/'capture_pids'/(session_id+'.pid')).unlink(missing_ok=True)
            except OSError:
                pass
            with self.lock:
                state = self.state['status']
            self.inspections.store.end_session(session_id, state)

    def snapshot(self):
        with self.lock:
            result = dict(self.state)
        if result.get("id"):
            result["inspections"] = self.inspections.store.query(session_id=result["id"], limit=50)["items"]
            result["counts"] = self.inspections.store.session_counts(result["id"])
            result['gpu_calls'] = self.inspections.store.session_gpu_counts(result['id'])
        return result

    def frame(self, number=0):
        with self.lock:
            if not number:
                return self.jpeg
            return next((jpeg for frame_number, jpeg in self.frame_cache if frame_number == number), None)

    def recalibrate(self):
        with self.analysis_lock:
            with self.lock:
                if not self.gate or self.state["status"] not in ("running", "calibrating"):
                    raise ValueError("Start a video before calibrating")
            self.gate.recalibrate()

    def stop(self):
        with self.lock:
            if not self.thread or not self.thread.is_alive():
                return self.snapshot()
            self.stop_event.set()
            self.condition.notify_all()
            session_id = self.state["id"]
            is_photos = self.state['kind'] == 'photos'
        try:
            if is_photos:
                self.thread.join(timeout=5)
                return self.snapshot()
            subprocess.run(["wsl.exe", "-d", self.distro, "-u", "root", "--", "bash",
                f"{self.wsl_root}/apps/stream_demo/stop_capture.sh", self.artifact, session_id],
                capture_output=True, timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.SubprocessError):
            pass
        self.thread.join(timeout=3)
        if self.thread.is_alive() and self.process and self.process.poll() is None:
            self.process.terminate()
        return self.snapshot()
