"""Stage timer for per-stage latency instrumentation."""
import time
from contextlib import contextmanager


class StageTimer:
    """Collects wall-clock durations per named stage.

    Usage:
        timer = StageTimer()
        with timer.stage("retrieval"):
            ...
        timer.as_millis()  # {"retrieval_ms": 123, "total_ms": 123}
    """

    def __init__(self) -> None:
        self._durations: dict[str, float] = {}
        self._start = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._durations[name] = self._durations.get(name, 0.0) + (time.perf_counter() - t0)

    def as_millis(self) -> dict[str, int]:
        out = {f"{k}_ms": int(v * 1000) for k, v in self._durations.items()}
        out["total_ms"] = int((time.perf_counter() - self._start) * 1000)
        return out
