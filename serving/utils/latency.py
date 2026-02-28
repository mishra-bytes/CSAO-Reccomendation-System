from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class LatencyTracker:
    stages_ms: dict[str, float] = field(default_factory=dict)
    _start_ts: float = field(default_factory=time.perf_counter)

    @contextmanager
    def track(self, stage: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages_ms[stage] = (time.perf_counter() - start) * 1000.0

    def finalize(self) -> dict[str, float]:
        self.stages_ms["total"] = (time.perf_counter() - self._start_ts) * 1000.0
        return self.stages_ms

