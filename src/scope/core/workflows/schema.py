"""Scope workflow schema — the `.scope-workflow.json` file format.

This module defines the Pydantic models that represent a shareable workflow.
The schema is a long-term contract: once workflows are shared publicly we
cannot easily rename or remove fields.  New *optional* fields may be added
in future versions; consumers MUST tolerate unknown fields (``extra="ignore"``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

from scope.core.lora.manifest import LoRAProvenance as _LoRAProvenance

WORKFLOW_FORMAT_VERSION = "1.0"


class WorkflowLoRAProvenance(_LoRAProvenance, extra="ignore"):
    """Forward-compatible wrapper around :class:`LoRAProvenance`.

    The upstream model does not set ``extra="ignore"``, so unknown fields
    added by a newer Scope version would cause validation errors.  This
    subclass adds that tolerance for the workflow schema while keeping the
    manifest model strict.
    """


class WorkflowMetadata(BaseModel, extra="ignore"):
    """Authorship and tooling metadata."""

    name: str
    description: str = ""
    author: str = ""
    created_at: datetime
    scope_version: str


class WorkflowPipelineSource(BaseModel, extra="ignore"):
    """Where a pipeline comes from."""

    type: Literal["builtin", "pypi", "git", "local"]
    plugin_name: str | None = None
    plugin_version: str | None = None
    package_spec: str | None = None


class WorkflowLoRA(BaseModel, extra="ignore"):
    """A LoRA adapter used by the workflow."""

    id: str | None = None
    filename: str
    weight: float = 1.0
    merge_mode: str = "permanent_merge"
    provenance: WorkflowLoRAProvenance | None = None
    expected_sha256: str | None = None


class WorkflowPipeline(BaseModel, extra="ignore"):
    """A single pipeline within the workflow."""

    pipeline_id: str
    pipeline_version: str
    source: WorkflowPipelineSource
    loras: list[WorkflowLoRA] = []
    params: dict[str, Any] = {}


class ScopeWorkflow(BaseModel, extra="ignore"):
    """Root schema for a ``.scope-workflow.json`` file."""

    format: Literal["scope-workflow"] = "scope-workflow"
    format_version: str = WORKFLOW_FORMAT_VERSION
    metadata: WorkflowMetadata
    pipelines: list[WorkflowPipeline]
