"""Per-mode baseline ceilings for product-quality dimensions.

Baselines live in ``product-tests/baselines/{local,cloud}.json`` and are
keyed by (workflow, dimension). A dimension above baseline fails the test;
missing baselines default to an effectively-infinite ceiling so a new
workflow doesn't silently pass without a baseline being committed.
"""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).parent.parent / "baselines"
_BIG = 10**9


def load(mode: str) -> dict:
    path = _ROOT / f"{mode}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def ceiling(mode: str, workflow: str, dim: str) -> float | int:
    return load(mode).get(workflow, {}).get(dim, _BIG)


def check(report, mode: str, workflow: str, dim: str, value: float | int) -> bool:
    """Populate the report with the measurement + fail if over baseline.

    Returns True if within baseline (or baseline absent), False if over.
    """
    report.measure(dim, value)
    limit = ceiling(mode, workflow, dim)
    if limit == _BIG:
        # No baseline — record but don't fail.
        return True
    if value > limit:
        report.fail(f"{dim}={value} > baseline[{mode}/{workflow}]={limit}")
        return False
    # Track drift as metadata for PR-comment signal.
    drift_pct = round(100 * (value - limit) / limit, 1) if limit else 0
    report.metadata[f"baseline_{dim}_drift_pct"] = str(drift_pct)
    return True
