"""Cross-cutting contracts applied to every product-test run.

A "contract" is a product-wide invariant that is NOT the test's primary
assertion but must never be violated regardless. The conftest autouse
fixture calls every contract at teardown; a violation hard-fails the
test even if the test body's own assertions passed.

These exist because the three failure modes we're gating on (unexplained
retries, unexpected session closes, UI errors) can happen silently —
they'd show up as a flake or a brief log line rather than an assertion
failure. Contracts convert "silent flake" into "loud red".
"""

from __future__ import annotations

from .no_retries import NoRetries, NoRetriesViolation
from .no_unexpected_session_close import (
    NoUnexpectedSessionClose,
    NoUnexpectedSessionCloseViolation,
)

__all__ = [
    "NoRetries",
    "NoRetriesViolation",
    "NoUnexpectedSessionClose",
    "NoUnexpectedSessionCloseViolation",
]
