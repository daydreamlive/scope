"""Generated constants for every ``data-testid`` declared in ``frontend/src``.

This module is the single source of truth for testids in pytest. Tests should
import from here instead of hardcoding strings:

    from harness import testids
    ctx.click(testids.STREAM_RUN_STOP)
    ctx.click(testids.workflow_card("local-passthrough"))

The module has two halves:

1. **Static constants** — one UPPER_SNAKE_CASE constant per literal
   ``data-testid="..."`` in the frontend. A test that references a testid that
   no longer exists in the frontend will fail to import, which is the signal.

2. **Dynamic factories** — one snake_case helper per ``data-testid={`foo-${x}`}``
   template. The template bodies are parsed by the sync script; the Python
   helper accepts the template variables as kwargs and returns the final
   testid string.

The **static** section below is auto-generated; **do not edit by hand**. Run:

    uv run python -m harness.testids --sync

…after any change to ``frontend/src/**/*.{ts,tsx,jsx,js}`` that adds, removes,
or renames a ``data-testid``. The CI PR gate fails if this file drifts from
the actual frontend scan (see ``.github/workflows/product-tests.yml``).

The **dynamic** section is maintained by hand — dynamic testids are rare
enough (two in the codebase today) that a hand-curated list is clearer than a
template-expression parser that's inevitably wrong on edge cases. Add a new
factory here when you introduce a new templated testid in the frontend.
"""

from __future__ import annotations

# fmt: off
# -----------------------------------------------------------------------------
# BEGIN AUTO-GENERATED  (do not edit; regenerate via `python -m harness.testids --sync`)
# -----------------------------------------------------------------------------

CLOUD_TOGGLE = "cloud-toggle"
INFERENCE_MODE_CONTINUE = "inference-mode-continue"
SINK_VIDEO = "sink-video"
START_STREAM_BUTTON = "start-stream-button"
STREAM_RUN_STOP = "stream-run-stop"
TELEMETRY_ACCEPT = "telemetry-accept"
TELEMETRY_DECLINE = "telemetry-decline"
TOUR_NEXT = "tour-next"
TOUR_SKIP = "tour-skip"
WORKFLOW_GET_STARTED = "workflow-get-started"
WORKFLOW_IMPORT_LOAD = "workflow-import-load"

# -----------------------------------------------------------------------------
# END AUTO-GENERATED
# -----------------------------------------------------------------------------
# fmt: on


# -----------------------------------------------------------------------------
# Dynamic factories (hand-maintained). Mirror a frontend template like
# ``data-testid={`inference-mode-${mode}`}``.
# -----------------------------------------------------------------------------


def inference_mode(mode: str) -> str:
    """E.g. ``inference-mode-local`` / ``inference-mode-cloud``.

    Frontend: ``frontend/src/components/onboarding/InferenceModeStep.tsx`` —
    ``data-testid={`inference-mode-${mode}`}``.
    """
    return f"inference-mode-{mode}"


def workflow_card(workflow_id: str) -> str:
    """E.g. ``workflow-card-local-passthrough`` / ``workflow-card-starter-mythical-creature``.

    Frontend: ``frontend/src/components/onboarding/WorkflowPickerStep.tsx`` —
    ``data-testid={`workflow-card-${wf.id}`}``.
    """
    return f"workflow-card-{workflow_id}"


# -----------------------------------------------------------------------------
# CLI entry point: `uv run python -m harness.testids --sync` regenerates the
# auto-generated block above. `--check` exits non-zero if the file is stale,
# which is what CI runs.
# -----------------------------------------------------------------------------

_AUTOGEN_BEGIN = (
    "# BEGIN AUTO-GENERATED  "
    "(do not edit; regenerate via `python -m harness.testids --sync`)"
)
_AUTOGEN_END = "# END AUTO-GENERATED"


def _scan_frontend(frontend_src: str | None = None) -> list[str]:
    """Return the sorted, unique list of literal ``data-testid`` values found
    in ``frontend/src``. Dynamic template testids (``data-testid={`foo-${x}`}``)
    are ignored here — they live in the hand-maintained factories section.
    """
    import re
    from pathlib import Path

    if frontend_src is None:
        # Default: assumes invocation from product-tests/ with repo root one up.
        here = Path(__file__).resolve()
        # product-tests/harness/testids.py → repo root is 2 up from harness/.
        repo_root = here.parent.parent.parent
        frontend_src_path = repo_root / "frontend" / "src"
    else:
        frontend_src_path = Path(frontend_src)

    if not frontend_src_path.exists():
        raise FileNotFoundError(f"frontend/src not found at {frontend_src_path}")

    # Match only literal forms: data-testid="foo" or data-testid='foo'.
    # Template-literal forms (data-testid={`foo-${x}`}) are deliberately excluded.
    pattern = re.compile(r'data-testid=["\']([^"\']+)["\']')

    found: set[str] = set()
    for ext in ("ts", "tsx", "js", "jsx"):
        for path in frontend_src_path.rglob(f"*.{ext}"):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for match in pattern.finditer(text):
                found.add(match.group(1))

    return sorted(found)


def _constant_name(testid: str) -> str:
    """Convert ``sink-video`` → ``SINK_VIDEO``."""
    return testid.replace("-", "_").replace(".", "_").upper()


def _render_autogen_block(testids: list[str]) -> str:
    lines = [_AUTOGEN_BEGIN, "# " + "-" * 77]
    lines.append("")
    for tid in testids:
        name = _constant_name(tid)
        lines.append(f'{name} = "{tid}"')
    lines.append("")
    lines.append("# " + "-" * 77)
    lines.append(_AUTOGEN_END)
    return "\n".join(lines)


def _splice(current: str, new_block: str) -> str:
    """Replace the auto-generated block inside ``current`` with ``new_block``."""
    start = current.index(_AUTOGEN_BEGIN)
    end = current.index(_AUTOGEN_END) + len(_AUTOGEN_END)
    return current[:start] + new_block + current[end:]


def _sync(write: bool) -> int:
    """Regenerate or verify the auto-generated block. Returns an exit code."""
    from pathlib import Path

    self_path = Path(__file__).resolve()
    current = self_path.read_text(encoding="utf-8")
    testids = _scan_frontend()
    new_block = _render_autogen_block(testids)
    updated = _splice(current, new_block)

    if updated == current:
        print(f"harness/testids.py up to date ({len(testids)} testids).")
        return 0

    if write:
        self_path.write_text(updated, encoding="utf-8")
        print(
            f"harness/testids.py updated ({len(testids)} testids). "
            "Commit the change along with the frontend diff."
        )
        return 0

    print(
        "harness/testids.py is OUT OF SYNC with frontend/src. "
        "Run:\n\n    uv run python -m harness.testids --sync\n\n"
        "…and commit the change.",
    )
    return 1


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="harness.testids",
        description="Regenerate or verify harness/testids.py from frontend/src.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--sync", action="store_true", help="Rewrite the auto-generated block."
    )
    group.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the file drifts (for CI).",
    )
    args = parser.parse_args()

    if args.sync:
        return _sync(write=True)
    # Default is --check so CI can run `python -m harness.testids` bare.
    return _sync(write=False)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
