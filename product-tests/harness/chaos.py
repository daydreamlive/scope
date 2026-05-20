"""ChaosDriver — seeded simulation of chaotic user behavior.

Samples actions (stop/start stream, switch input, change workflow, spam
parameter) from a weighted distribution using a seeded ``random.Random``,
so runs are reproducible given the same ``--chaos-seed``.

Every action is logged to ``timeline.jsonl`` with a ``test_initiated`` flag
so ``FailureWatcher`` can attribute session closes.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ChaosAction:
    name: str
    weight: float
    fn: Callable[[], None]


@dataclass
class ChaosDriver:
    seed: str
    report_dir: Path
    actions: list[ChaosAction] = field(default_factory=list)
    tick_min_ms: int = 200
    tick_max_ms: int = 2000
    _rng: random.Random = field(init=False)
    _timeline_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._timeline_path = self.report_dir / "timeline.jsonl"
        # Truncate the timeline for this run.
        self._timeline_path.write_text("")

    def register(self, name: str, weight: float, fn: Callable[[], None]) -> None:
        self.actions.append(ChaosAction(name=name, weight=weight, fn=fn))

    def run(self, duration_sec: float) -> None:
        """Fire actions for ``duration_sec`` seconds, pacing per tick."""
        if not self.actions:
            raise RuntimeError("ChaosDriver has no registered actions")
        weights = [a.weight for a in self.actions]
        end = time.monotonic() + duration_sec
        while time.monotonic() < end:
            action = self._rng.choices(self.actions, weights=weights, k=1)[0]
            started = time.time()
            error: str | None = None
            try:
                action.fn()
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
            self._log(
                {
                    "t": started,
                    "action": action.name,
                    "error": error,
                    "test_initiated": True,
                }
            )
            tick_ms = self._rng.randint(self.tick_min_ms, self.tick_max_ms)
            time.sleep(tick_ms / 1000.0)

    def _log(self, event: dict) -> None:
        with open(self._timeline_path, "a") as fh:
            fh.write(json.dumps(event) + "\n")
