# ga/ga_timing.py
import time
from typing import List, Tuple

class StopWatch:
    def __init__(self):
        self._marks: List[Tuple[str, float]] = []
        self._t0 = time.perf_counter()

    def tick(self, label: str) -> None:
        self._marks.append((label, time.perf_counter()))

    def split_durations(self) -> List[Tuple[str, float]]:
        """Return (label, seconds since previous tick) for each tick."""
        out: List[Tuple[str, float]] = []
        prev = self._t0
        for label, t in self._marks:
            out.append((label, t - prev))
            prev = t
        return out

    def total(self) -> float:
        if not self._marks:
            return 0.0
        return self._marks[-1][1] - self._t0

    def pretty_line(self, prefix: str = "") -> str:
        parts = []
        for label, dt in self.split_durations():
            parts.append(f"{label}={dt:.2f}s")
        parts.append(f"total={self.total():.2f}s")
        return (prefix + " " if prefix else "") + " | ".join(parts)
