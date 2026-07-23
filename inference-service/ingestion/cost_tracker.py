"""Indexing-cost accounting.

Accounting protocol:
  - SHARED costs (one extraction pass feeds all three paradigms): LLM wall-time,
    token counts, USD estimate — reported once.
  - PARADIGM-SPECIFIC costs: embedding counts, build wall-time, storage bytes —
    reported per paradigm (hybrid / light_rag / hippo_rag).

Report written to artifacts/indexing_cost_report.json.
"""
import json
import time
from contextlib import contextmanager
from pathlib import Path

# USD per 1M tokens; update to current pricing and record the retrieval date
PRICE_PER_M_INPUT = 2.50   # gpt-4o input
PRICE_PER_M_OUTPUT = 10.00  # gpt-4o output


class CostTracker:
    def __init__(self) -> None:
        self.wall_time_s: dict[str, float] = {}
        self.tokens: dict[str, dict[str, int]] = {}
        self.embedding_counts: dict[str, int] = {}
        self.storage_bytes: dict[str, int] = {}

    @contextmanager
    def llm_call(self, phase: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.wall_time_s[phase] = self.wall_time_s.get(phase, 0.0) + time.perf_counter() - t0

    @contextmanager
    def build_phase(self, paradigm: str):
        key = f"build:{paradigm}"
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.wall_time_s[key] = self.wall_time_s.get(key, 0.0) + time.perf_counter() - t0

    def add_tokens(self, phase: str, prompt_tokens: int, completion_tokens: int) -> None:
        bucket = self.tokens.setdefault(phase, {"prompt": 0, "completion": 0})
        bucket["prompt"] += prompt_tokens
        bucket["completion"] += completion_tokens

    def add_embeddings(self, paradigm: str, count: int) -> None:
        self.embedding_counts[paradigm] = self.embedding_counts.get(paradigm, 0) + count

    def add_storage(self, paradigm: str, num_bytes: int) -> None:
        self.storage_bytes[paradigm] = self.storage_bytes.get(paradigm, 0) + num_bytes

    def report(self) -> dict:
        usd = {
            phase: round(
                t["prompt"] / 1e6 * PRICE_PER_M_INPUT + t["completion"] / 1e6 * PRICE_PER_M_OUTPUT,
                4,
            )
            for phase, t in self.tokens.items()
        }
        return {
            "wall_time_s": {k: round(v, 2) for k, v in self.wall_time_s.items()},
            "tokens": self.tokens,
            "estimated_usd": usd,
            "embedding_counts": self.embedding_counts,
            "storage_bytes": self.storage_bytes,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.report(), indent=2), encoding="utf-8")
