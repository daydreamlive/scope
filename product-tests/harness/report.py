"""TestReport — uniform JSON + Markdown emission for every test.

The report is keyed to product-quality dimensions, not just pass/fail:

    dimensions:
      first_frame_time_ms
      parameter_round_trip_ms_p95
      session_stability_rate
      retry_count              (must be 0)
      unexpected_close_count   (must be 0)
      ui_error_events          (must be 0)

A summary.md across all tests is rendered after the run for PR comments.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TestReport:
    __test__ = False  # prevent pytest from collecting this as a test class

    test: str
    mode: str  # "local" | "cloud"
    report_dir: Path
    dimensions: dict[str, float | int] = field(default_factory=dict)
    hard_fails: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    _start_ts: float = field(default_factory=time.time, init=False)

    @property
    def passed(self) -> bool:
        return not self.hard_fails

    def fail(self, reason: str) -> None:
        self.hard_fails.append(reason)

    def measure(self, name: str, value: float | int) -> None:
        self.dimensions[name] = value

    def add_artifact(self, path: Path | str) -> None:
        self.artifacts.append(str(path))

    def emit(self) -> Path:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        path = self.report_dir / "report.json"
        payload = {
            "test": self.test,
            "mode": self.mode,
            "pass": self.passed,
            "duration_sec": round(time.time() - self._start_ts, 3),
            "hard_fails": self.hard_fails,
            "dimensions": self.dimensions,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
        }
        path.write_text(json.dumps(payload, indent=2))
        return path


def aggregate_summary(reports_root: Path) -> Path:
    """Walk reports_root and emit a summary.md suitable for PR comments."""
    rows: list[dict] = []
    for p in sorted(reports_root.rglob("report.json")):
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:
            continue

    lines = ["# product-tests summary", ""]
    total = len(rows)
    passed = sum(1 for r in rows if r.get("pass"))
    lines.append(f"**{passed}/{total} passed**")
    lines.append("")
    lines.append(
        "| test | mode | pass | first_frame_ms | retries | unexpected_closes |"
    )
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        d = r.get("dimensions", {})
        lines.append(
            "| {test} | {mode} | {p} | {ff} | {rc} | {uc} |".format(
                test=r.get("test", "?"),
                mode=r.get("mode", "?"),
                p="✅" if r.get("pass") else "❌",
                ff=d.get("first_frame_time_ms", "—"),
                rc=d.get("retry_count", "—"),
                uc=d.get("unexpected_close_count", "—"),
            )
        )

    failed = [r for r in rows if not r.get("pass")]
    if failed:
        lines.append("")
        lines.append("## Hard failures")
        for r in failed:
            lines.append(f"- **{r['test']}**: {', '.join(r.get('hard_fails', []))}")

    summary = reports_root / "summary.md"
    summary.write_text("\n".join(lines))
    return summary
