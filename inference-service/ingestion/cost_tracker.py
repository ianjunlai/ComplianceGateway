"""Indexing-cost accounting.

Accounting protocol:
  - SHARED costs (one extraction pass feeds all three paradigms): LLM wall-time
    and token counts — reported once.
  - PARADIGM-SPECIFIC costs: embedding counts, build wall-time, storage bytes —
    reported per paradigm (hybrid / light_rag / hippo_rag).

Cost is reported in tokens, not currency: token counts are objective and
reproducible, whereas per-provider prices change and are not comparable across
vendors. Convert to currency separately if needed, using the provider's
pricing at a stated date.

Report written to artifacts/indexing_cost_report.json.
"""
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import config


class CostTracker:
    """Every mutator holds a lock: extraction may run across a thread pool, and
    `bucket["prompt"] += n` is a read-modify-write that silently loses counts
    when two threads interleave. Cost is a reported metric, so an
    under-count here would be a quiet error in the results, not a crash.
    """

    def __init__(self) -> None:
        self.wall_time_s: dict[str, float] = {}
        self.tokens: dict[str, dict[str, int]] = {}
        self.embedding_counts: dict[str, int] = {}
        self.storage_bytes: dict[str, int] = {}
        self.notes: dict[str, object] = {}
        self._lock = threading.Lock()

    @contextmanager
    def llm_call(self, phase: str):
        """NOTE: under concurrency this sums per-call API time, which exceeds
        elapsed time by roughly the worker count. `notes` records the
        concurrency so the figure stays interpretable."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            with self._lock:
                self.wall_time_s[phase] = self.wall_time_s.get(phase, 0.0) + elapsed

    @contextmanager
    def build_phase(self, paradigm: str):
        key = f"build:{paradigm}"
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            with self._lock:
                self.wall_time_s[key] = self.wall_time_s.get(key, 0.0) + elapsed

    def add_tokens(self, phase: str, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            bucket = self.tokens.setdefault(phase, {"prompt": 0, "completion": 0})
            bucket["prompt"] += prompt_tokens
            bucket["completion"] += completion_tokens

    def add_embeddings(self, paradigm: str, count: int) -> None:
        with self._lock:
            self.embedding_counts[paradigm] = self.embedding_counts.get(paradigm, 0) + count

    def add_storage(self, paradigm: str, num_bytes: int) -> None:
        with self._lock:
            self.storage_bytes[paradigm] = self.storage_bytes.get(paradigm, 0) + num_bytes

    def note(self, key: str, value: object) -> None:
        """Run conditions that make the numbers above interpretable."""
        with self._lock:
            self.notes[key] = value

    def report(self) -> dict:
        tokens = {
            phase: {**t, "total": t["prompt"] + t["completion"]}
            for phase, t in self.tokens.items()
        }
        return {
            # Which model produced these token counts -- they are not
            # comparable across providers without it.
            "extraction_provider": config.EXTRACTION_PROVIDER,
            "extraction_model": config.EXTRACTION_MODEL,
            "extraction_profile": config.EXTRACTION_PROFILE,
            "wall_time_s": {k: round(v, 2) for k, v in self.wall_time_s.items()},
            "tokens": tokens,
            "embedding_counts": self.embedding_counts,
            "storage_bytes": self.storage_bytes,
            "notes": self.notes,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.report(), indent=2), encoding="utf-8")
