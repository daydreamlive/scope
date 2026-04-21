"""YAML → Case dataclass loader for the eval harness.

A case file looks like::

    name: starter-mythical-creature
    description: |
      Reproduces the Mythical Creature teaching starter.
    prompt: |
      I want a slime creature ...
    runs: 5
    expect:
      - pipelines_equal: [longlive]
      - wire_present: { kind: vace_to_pipeline }
    forbid:
      - bad_handle_prefix: "parameter:"

Each entry under ``expect`` / ``forbid`` is a single-key mapping whose key is
the name of a check in :mod:`evals.grader` and whose value is the check
argument. We deliberately keep the format flat and declarative so adding a
case is just dropping a new YAML file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CheckSpec:
    """One graded check: ``(name, arg)`` where ``name`` resolves to a function
    in :mod:`evals.grader`."""

    name: str
    arg: Any


@dataclass
class Case:
    name: str
    prompt: str
    description: str = ""
    runs: int = 5
    expect: list[CheckSpec] = field(default_factory=list)
    forbid: list[CheckSpec] = field(default_factory=list)
    source_path: Path | None = None


def _parse_check_list(raw: list[Any], context: str) -> list[CheckSpec]:
    """Convert a list of single-key mappings to ``CheckSpec``s."""
    out: list[CheckSpec] = []
    for idx, entry in enumerate(raw or []):
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(
                f"{context}[{idx}] must be a single-key mapping, got: {entry!r}"
            )
        ((name, arg),) = entry.items()
        if not isinstance(name, str):
            raise ValueError(f"{context}[{idx}] check name must be a string")
        out.append(CheckSpec(name=name, arg=arg))
    return out


def load_case(path: Path) -> Case:
    """Load a single case YAML file into a :class:`Case`."""
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at top level")

    name = data.get("name") or path.stem
    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{path}: 'prompt' is required and must be a non-empty string")

    runs = data.get("runs", 5)
    if not isinstance(runs, int) or runs < 1:
        raise ValueError(f"{path}: 'runs' must be a positive integer")

    return Case(
        name=str(name),
        prompt=prompt,
        description=str(data.get("description") or ""),
        runs=runs,
        expect=_parse_check_list(data.get("expect") or [], f"{path}:expect"),
        forbid=_parse_check_list(data.get("forbid") or [], f"{path}:forbid"),
        source_path=path,
    )


def discover_cases(cases_dir: Path) -> list[Case]:
    """Load every ``*.yaml`` / ``*.yml`` case in ``cases_dir``, alpha-sorted."""
    paths = sorted(
        p for p in cases_dir.iterdir() if p.suffix in (".yaml", ".yml") and p.is_file()
    )
    return [load_case(p) for p in paths]
