"""Native node telemetry: samples CPU utilization and RSS in a background thread.

Degrades to a no-op sampler when psutil is unavailable (e.g. inside a stock
framework image), so trainers never need a runtime install.
"""
import threading
import time

try:
    import psutil
except ImportError:
    psutil = None


class NodeSampler:
    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.samples = []          # (t, cpu_percent, rss_bytes)
        self._stop = threading.Event()
        self._proc = psutil.Process() if psutil else None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        psutil.cpu_percent(None)   # prime the counter
        t0 = time.perf_counter()
        while not self._stop.is_set():
            time.sleep(self.interval)
            self.samples.append((
                time.perf_counter() - t0,
                psutil.cpu_percent(None),           # whole-node CPU %
                self._proc.memory_info().rss,
            ))

    def __enter__(self):
        if psutil:
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()

    def summary(self):
        if not self.samples:
            return {}
        cpu = [s[1] for s in self.samples]
        rss = [s[2] for s in self.samples]
        return {
            "cpu_util_mean_pct": round(sum(cpu) / len(cpu), 1),
            "cpu_util_peak_pct": round(max(cpu), 1),
            "rss_peak_mb": round(max(rss) / 2**20, 1),
            "n_samples": len(self.samples),
            "series": [(round(t, 3), c, round(r / 2**20, 1)) for t, c, r in self.samples],
        }
