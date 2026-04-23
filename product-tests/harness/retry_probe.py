"""RetryProbe — query instrumented retry counters on the Scope server.

Talks to ``/api/v1/_debug/retry_stats`` (only available when the server is
launched with ``SCOPE_TEST_INSTRUMENTATION=1``).

Semantics:
  - ``snapshot()`` returns the current counts dict
  - ``events()`` returns the recorded events (with context)
  - ``assert_zero()`` raises if any counter > 0 (hard failure)
  - ``reset()`` zeros counters (used between phases of a test)

The zero-retry gate is the entire point of this system. Any retry counter
ticking up during a scenario is a hard fail, not a "flaky test that passed
eventually".
"""

from __future__ import annotations

from dataclasses import dataclass

import requests


class RetryAssertionError(AssertionError):
    """Raised when any instrumented retry counter is non-zero at checkpoint."""


@dataclass
class RetryProbe:
    base_url: str
    timeout: float = 5.0

    def snapshot(self) -> dict[str, int]:
        r = requests.get(
            f"{self.base_url}/api/v1/_debug/retry_stats", timeout=self.timeout
        )
        r.raise_for_status()
        return r.json().get("counts", {})

    def events(self) -> list[dict]:
        r = requests.get(
            f"{self.base_url}/api/v1/_debug/retry_stats", timeout=self.timeout
        )
        r.raise_for_status()
        return r.json().get("events", [])

    def reset(self) -> None:
        r = requests.post(
            f"{self.base_url}/api/v1/_debug/retry_stats/reset", timeout=self.timeout
        )
        r.raise_for_status()

    def assert_zero(self, *, allow: tuple[str, ...] = ()) -> None:
        """Raise if any counter not in ``allow`` is non-zero.

        ``allow`` is for counters that are legitimately expected to tick (e.g.
        ``cloud_connect_attempts`` — one attempt is fine, a retry is not).
        """
        counts = self.snapshot()
        nonzero = {k: v for k, v in counts.items() if v > 0 and k not in allow}
        if nonzero:
            evts = self.events()
            raise RetryAssertionError(
                f"Retry counters non-zero: {nonzero}\n"
                f"Events: {evts}\n"
                "A retry or drop was observed; the product considers this a hard fail."
            )
