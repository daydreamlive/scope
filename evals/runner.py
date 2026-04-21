"""Runner: execute cases, grade proposals, print a summary, dump artifacts."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from .case import Case, discover_cases, load_case
from .driver import DriveResult, run_case
from .grader import run_check

logger = logging.getLogger("evals")


EVALS_ROOT = Path(__file__).resolve().parent
DEFAULT_CASES_DIR = EVALS_ROOT / "cases"
DEFAULT_OUTPUT_DIR = EVALS_ROOT / "outputs"


@dataclass
class RunResult:
    case_name: str
    run_index: int
    passed: bool
    failures: list[str] = field(default_factory=list)
    drive: DriveResult | None = None
    wall_seconds: float = 0.0


@dataclass
class CaseSummary:
    case: Case
    runs: list[RunResult]

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.runs if r.passed)

    @property
    def rate_pct(self) -> float:
        if not self.runs:
            return 0.0
        return 100.0 * self.pass_count / len(self.runs)

    @property
    def grouped_failures(self) -> list[str]:
        """Human-readable failure labels grouped by run index."""
        return [
            f"r{r.run_index}: {'; '.join(r.failures)}"
            for r in self.runs
            if not r.passed and r.failures
        ]


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def _grade(case: Case, drive: DriveResult) -> tuple[bool, list[str]]:
    """Return ``(passed, failure_reasons)`` for a single run."""
    failures: list[str] = []

    if drive.error and drive.proposal is None:
        return False, [f"driver error: {drive.error.strip()[:200]}"]

    if drive.proposal is None:
        return False, [
            "no workflow_proposal SSE event seen — agent likely gave a "
            "text-only response or failed before proposing"
        ]

    graph = drive.proposal

    def _fail(name: str, arg: object, detail: str) -> None:
        arg_repr = json.dumps(arg, default=str) if not isinstance(arg, str) else arg
        failures.append(f"{name}({arg_repr}): {detail}")

    for spec in case.expect:
        res = run_check(spec.name, graph, spec.arg)
        if not res.ok:
            _fail(spec.name, spec.arg, res.detail)

    # `forbid`: check returning ok=True means the forbidden pattern was
    # NOT present, which is the success condition. Checks in forbid are
    # the same named functions as in expect; we invert nothing — the
    # `bad_handle_prefix` etc. are themselves phrased as "ok if absent".
    for spec in case.forbid:
        res = run_check(spec.name, graph, spec.arg)
        if not res.ok:
            _fail(f"forbid.{spec.name}", spec.arg, res.detail)

    return (not failures), failures


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------


def _write_artifacts(
    output_dir: Path, case_name: str, run_index: int, run: RunResult
) -> None:
    out = output_dir / case_name / f"r{run_index:02d}"
    out.mkdir(parents=True, exist_ok=True)
    drive = run.drive or DriveResult()
    (out / "proposal.json").write_text(
        json.dumps(drive.proposal or {}, indent=2, default=str)
    )
    (out / "meta.json").write_text(
        json.dumps(
            {
                "case": case_name,
                "run_index": run_index,
                "passed": run.passed,
                "failures": run.failures,
                "rationale": drive.rationale,
                "proposal_id": drive.proposal_id,
                "session_id": drive.session_id,
                "wall_seconds": round(run.wall_seconds, 3),
                "error": drive.error,
            },
            indent=2,
            default=str,
        )
    )
    # SSE trace as JSONL for easy grepping.
    with (out / "trace.jsonl").open("w") as f:
        for evt in drive.trace:
            f.write(json.dumps(evt, default=str) + "\n")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def _run_single(
    app,
    case: Case,
    run_index: int,
    *,
    model_override: str | None,
    provider_override: str | None,
) -> RunResult:
    t0 = perf_counter()
    drive = await run_case(
        app,
        case.prompt,
        model_override=model_override,
        provider_override=provider_override,
    )
    wall = perf_counter() - t0
    passed, failures = _grade(case, drive)
    return RunResult(
        case_name=case.name,
        run_index=run_index,
        passed=passed,
        failures=failures,
        drive=drive,
        wall_seconds=wall,
    )


async def run_cases(
    cases: list[Case],
    *,
    runs_override: int | None = None,
    model_override: str | None = None,
    provider_override: str | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[CaseSummary]:
    # Import here so a plain ``python -m evals --help`` doesn't pay the
    # Scope import cost.
    from scope.server.app import app  # noqa: PLC0415

    summaries: list[CaseSummary] = []
    for case in cases:
        n = runs_override or case.runs
        run_results: list[RunResult] = []
        for i in range(1, n + 1):
            logger.info(f"[{case.name}] run {i}/{n}...")
            rr = await _run_single(
                app,
                case,
                i,
                model_override=model_override,
                provider_override=provider_override,
            )
            _write_artifacts(output_dir, case.name, i, rr)
            run_results.append(rr)
            status = "PASS" if rr.passed else "FAIL"
            detail = "" if rr.passed else f" — {'; '.join(rr.failures)[:160]}"
            logger.info(
                f"[{case.name}] run {i}/{n} {status} ({rr.wall_seconds:.1f}s){detail}"
            )
        summaries.append(CaseSummary(case=case, runs=run_results))
    return summaries


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_summary(summaries: list[CaseSummary], output_dir: Path) -> tuple[int, int]:
    """Return ``(total_pass, total_runs)``."""
    # Column widths
    name_w = max((len(s.case.name) for s in summaries), default=4)
    name_w = max(name_w, 4)

    header = f"{'case'.ljust(name_w)}  runs  pass  rate   failures"
    print(header)
    total_pass = total_runs = 0
    for s in summaries:
        failures = "; ".join(s.grouped_failures)[:200]
        total_pass += s.pass_count
        total_runs += len(s.runs)
        print(
            f"{s.case.name.ljust(name_w)}  "
            f"{len(s.runs):>4}  "
            f"{s.pass_count:>4}  "
            f"{s.rate_pct:>4.0f}%  "
            f"{failures}"
        )
    rule_w = max(len(header), 60)
    print("─" * rule_w)
    overall_rate = 100.0 * total_pass / total_runs if total_runs else 0.0
    print(
        f"{'overall'.ljust(name_w)}  {total_runs:>4}  {total_pass:>4}  "
        f"{overall_rate:>4.0f}%"
    )
    print(f"\nArtifacts: {output_dir}/<case>/<run>/{{proposal.json,trace.jsonl}}")
    return total_pass, total_runs


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _resolve_cases(cases_dir: Path, selected: list[str] | None) -> list[Case]:
    if not selected:
        return discover_cases(cases_dir)
    out: list[Case] = []
    for s in selected:
        candidate = cases_dir / (s if s.endswith((".yaml", ".yml")) else f"{s}.yaml")
        if not candidate.exists():
            raise FileNotFoundError(f"no such case: {candidate}")
        out.append(load_case(candidate))
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(
        level=os.environ.get("EVALS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    p = argparse.ArgumentParser(prog="python -m evals")
    p.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case name (with or without .yaml). Repeatable. Omit for all cases.",
    )
    p.add_argument("--runs", type=int, default=None, help="Override runs per case.")
    p.add_argument("--model", default=None, help="Override model id.")
    p.add_argument("--provider", default=None, help="Override provider.")
    p.add_argument(
        "--cases-dir",
        default=str(DEFAULT_CASES_DIR),
        help="Directory containing case YAMLs.",
    )
    p.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Where to write per-run artifacts.",
    )
    p.add_argument(
        "--fail-threshold",
        type=float,
        default=None,
        help="Exit non-zero if overall pass-rate < this percentage.",
    )
    args = p.parse_args(argv)

    cases_dir = Path(args.cases_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        cases = _resolve_cases(cases_dir, args.case)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if not cases:
        print(f"no cases found in {cases_dir}", file=sys.stderr)
        return 2

    summaries = asyncio.run(
        run_cases(
            cases,
            runs_override=args.runs,
            model_override=args.model,
            provider_override=args.provider,
            output_dir=output_dir,
        )
    )
    print()
    total_pass, total_runs = print_summary(summaries, output_dir)

    if args.fail_threshold is not None and total_runs:
        rate = 100.0 * total_pass / total_runs
        if rate < args.fail_threshold:
            print(
                f"\nFAIL: overall {rate:.1f}% < threshold {args.fail_threshold:.1f}%",
                file=sys.stderr,
            )
            return 1
    return 0
