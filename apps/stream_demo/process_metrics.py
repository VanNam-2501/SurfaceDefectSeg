"""Live host-process CPU/RAM without an extra runtime dependency."""
import ctypes
import os
from pathlib import Path
import threading
import time


def memory_bytes():
    if os.name == 'nt':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), 'GetProcessMemoryInfo')
        return counters.WorkingSetSize, counters.PeakWorkingSetSize
    values = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:')):
            key, value = line.split(':', 1)
            values[key] = int(value.split()[0]) * 1024
    return values['VmRSS'], values['VmHWM']


class ProcessMetrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.previous = (time.perf_counter(), time.process_time())
        self.cached = {}

    def snapshot(self):
        with self.lock:
            now, cpu = time.perf_counter(), time.process_time()
            elapsed = now - self.previous[0]
            if self.cached and elapsed < 1:
                return dict(self.cached)
            usage = max(0., (cpu - self.previous[1]) / elapsed * 100) if elapsed > .1 else None
            self.previous = now, cpu
            try:
                resident, peak = memory_bytes()
            except (OSError, KeyError, AttributeError, ValueError):
                resident = peak = None
            self.cached = {'pid': os.getpid(), 'rss_mib': resident / 1048576 if resident is not None else None,
                'peak_rss_mib': peak / 1048576 if peak is not None else None,
                'cpu_one_core_percent': usage, 'logical_cpus': os.cpu_count(), 'updated_at': time.time(),
                'scope': 'Python web process only; CPU 100% = one logical core. Excludes browser and separate GPU worker. Host RAM and GPU memory are reported separately.'}
            return dict(self.cached)
