"""Bounded, measured performance samples retained while navigating the demo."""
from collections import deque
import sqlite3
import threading
import time


class PerformanceMonitor:
    def __init__(self, video, store, process_metrics, limit=300):
        self.video, self.store, self.metrics = video, store, process_metrics
        self.samples = deque(maxlen=limit)
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self.session_id = None
        self.source = ''
        self.status = 'idle'
        self.error = ''
        self.thread = threading.Thread(target=self._run, daemon=True, name='performance-sampler')
        self.thread.start()

    def sample_once(self):
        with self.video.lock:
            state = self.video.state
            session_id = state.get('id')
            source, status = state.get('source', ''), state.get('status', 'idle')
            tracks = list(state.get('tracks', []))
            point = {'timestamp': time.time(), 'analysis_steps': state.get('analysis_steps', 0),
                'cpu_frame_ms': state.get('host_gate_ms'), 'frames_received': state.get('frames_received', 0),
                'active_tracks': sum(t.get('confirmed', False) and t.get('observed', True) for t in tracks),
                'predicted_tracks': sum(t.get('observed') is False for t in tracks)}
        gpu = self.store.session_gpu_counts(session_id) if session_id else {'submitted': 0, 'completed': 0, 'pending': 0, 'failed': 0}
        process = self.metrics.snapshot()
        point.update(cpu_one_core_percent=process.get('cpu_one_core_percent'), ram_mib=process.get('rss_mib'),
            gpu_submitted=gpu['submitted'], gpu_completed=gpu['completed'], gpu_pending=gpu['pending'], gpu_failed=gpu['failed'])
        with self.lock:
            if self.session_id == session_id and self.samples and self.samples[-1]['analysis_steps'] == point['analysis_steps']:
                point['cpu_frame_ms'] = None  # No new frame: do not invent a processing sample.
            if session_id != self.session_id:
                self.samples.clear()
            self.session_id, self.source, self.status = session_id, source, status
            self.samples.append(point)
            self.error = ''

    def _run(self):
        while not self.closed.is_set():
            try:
                self.sample_once()
            except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
                with self.lock:
                    self.error = str(exc)
            self.closed.wait(1)

    def snapshot(self):
        with self.lock:
            return {'schema': 1, 'session_id': self.session_id, 'source': self.source, 'status': self.status,
                'sample_interval_seconds': 1, 'history_seconds': self.samples.maxlen,
                'samples': list(self.samples), 'error': self.error,
                'scope': 'Measured Python web-process CPU/RAM and current-session counters. CPU 100% equals one logical core; GPU calls are product requests, not GPU utilization. History starts when the server runs and resets for a new session.'}

    def close(self):
        self.closed.set()
        self.thread.join(timeout=2)
