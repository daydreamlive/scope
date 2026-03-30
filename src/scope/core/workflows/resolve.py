"""Dependency resolution for imported workflows.

Given a :class:`WorkflowRequest`, check which pipelines, plugins, and LoRAs
are available locally and produce a :class:`WorkflowResolutionPlan` that the
frontend can display in a trust-gate dialog before any side-effects occur.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from scope.core.plugins.manager import PluginManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal request models -- only the fields the backend actually accesses.
# All models use extra="ignore" so the frontend can send the full workflow
# JSON and the backend silently drops fields it doesn't care about.
# ---------------------------------------------------------------------------


class WorkflowLoRAProvenance(BaseModel, extra="ignore"):
    source: str
    repo_id: str | None = None
    hf_filename: str | None = None
    model_id: str | None = None
    version_id: str | None = None
    url: str | None = None


class WorkflowLoRA(BaseModel, extra="ignore"):
    filename: str = Field(max_length=255)
    provenance: WorkflowLoRAProvenance | None = None


class WorkflowPipelineSource(BaseModel, extra="ignore"):
    type: Literal["builtin", "pypi", "git", "local"]
    plugin_name: str | None = Field(default=None, min_length=1)
    plugin_version: str | None = None
    package_spec: str | None = None


class WorkflowPipeline(BaseModel, extra="ignore"):
    pipeline_id: str = Field(min_length=1)
    source: WorkflowPipelineSource
    loras: list[WorkflowLoRA] = Field(default=[], max_length=100)


class WorkflowRequest(BaseModel, extra="ignore"):
    pipelines: list[WorkflowPipeline] = Field(max_length=50)
    min_scope_version: str | None = None


# ---------------------------------------------------------------------------
# Resolution models
# ---------------------------------------------------------------------------


class ResolutionItem(BaseModel):
    """A single dependency check result."""

    kind: Literal["pipeline", "plugin", "lora"]
    name: str
    status: Literal["ok", "missing", "version_mismatch"]
    detail: str | None = None
    action: str | None = None
    can_auto_resolve: bool = False


class WorkflowResolutionPlan(BaseModel):
    """Full resolution result for a workflow import."""

    can_apply: bool
    items: list[ResolutionItem]
    warnings: list[str] = []


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def _resolve_builtin(pipeline_id: str) -> ResolutionItem:
    from scope.core.pipelines.registry import PipelineRegistry

    if PipelineRegistry.is_registered(pipeline_id):
        return ResolutionItem(kind="pipeline", name=pipeline_id, status="ok")

    import torch

    if not torch.cuda.is_available():
        detail = (
            f"Built-in pipeline '{pipeline_id}' not found. "
            "Your hardware may not support this pipeline — a compatible GPU is required."
        )
    else:
        detail = f"Built-in pipeline '{pipeline_id}' not found"

    return ResolutionItem(
        kind="pipeline",
        name=pipeline_id,
        status="missing",
        detail=detail,
    )


def _resolve_plugin(
    wp: WorkflowPipeline,
    plugins: list[dict],
) -> ResolutionItem:
    plugin_name = wp.source.plugin_name
    if plugin_name is None:
        return ResolutionItem(
            kind="plugin",
            name=wp.pipeline_id,
            status="missing",
            detail="No plugin name specified in workflow",
        )

    installed = next((p for p in plugins if p.get("name") == plugin_name), None)
    if installed is None:
        return _missing_plugin_item(wp.source, plugin_name)

    return _check_plugin_version(
        plugin_name,
        installed.get("version"),
        wp.source.plugin_version,
    )


def _missing_plugin_item(
    source: WorkflowPipelineSource,
    plugin_name: str,
) -> ResolutionItem:
    if source.type == "git" and source.package_spec:
        action = f"Install from git: {source.package_spec}"
    else:
        spec = plugin_name
        if source.plugin_version:
            spec += f">={source.plugin_version}"
        action = f"Install {spec} from PyPI"

    return ResolutionItem(
        kind="plugin",
        name=plugin_name,
        status="missing",
        detail=f"Plugin '{plugin_name}' is not installed",
        action=action,
        can_auto_resolve=True,
    )


def _check_plugin_version(
    plugin_name: str,
    installed_version: str | None,
    workflow_version: str | None,
) -> ResolutionItem:
    if not workflow_version:
        return ResolutionItem(kind="plugin", name=plugin_name, status="ok")
    if not installed_version:
        return ResolutionItem(
            kind="plugin",
            name=plugin_name,
            status="ok",
            detail=f"Could not verify version (no version metadata for {plugin_name})",
        )
    try:
        if Version(installed_version) < Version(workflow_version):
            return ResolutionItem(
                kind="plugin",
                name=plugin_name,
                status="version_mismatch",
                detail=(
                    f"Installed {installed_version},"
                    f" workflow expects {workflow_version}"
                ),
                action=f"Upgrade {plugin_name} to >={workflow_version}",
                can_auto_resolve=True,
            )
        return ResolutionItem(kind="plugin", name=plugin_name, status="ok")
    except InvalidVersion:
        return ResolutionItem(
            kind="plugin",
            name=plugin_name,
            status="ok",
            detail=f"Could not compare versions (installed={installed_version})",
        )


def _resolve_lora(lora: WorkflowLoRA, lora_dir: Path) -> ResolutionItem:
    # Guard against path traversal: ensure the resolved path stays within lora_dir
    target = (lora_dir / lora.filename).resolve()
    if not target.is_relative_to(lora_dir.resolve()):
        return ResolutionItem(
            kind="lora",
            name=lora.filename,
            status="missing",
            detail="Invalid LoRA filename",
        )

    if target.exists():
        return ResolutionItem(kind="lora", name=lora.filename, status="ok")

    prov = lora.provenance
    has_provenance = prov is not None and prov.source != "local"

    action = None
    if has_provenance:
        if prov.source == "huggingface":
            action = f"Download from HuggingFace: {prov.repo_id}"
        elif prov.source == "civitai":
            action = f"Download from CivitAI (model {prov.model_id})"
        elif prov.url:
            action = f"Download from {prov.url}"
        else:
            action = "Download from source"

    return ResolutionItem(
        kind="lora",
        name=lora.filename,
        status="missing",
        detail=f"LoRA '{lora.filename}' not found locally",
        action=action,
        can_auto_resolve=has_provenance,
    )


def _check_min_scope_version(
    min_version_str: str,
    warnings: list[str],
) -> None:
    """Compare *min_version_str* against the installed Scope version."""
    import importlib.metadata

    try:
        current = Version(importlib.metadata.version("daydream-scope"))
        required = Version(min_version_str)
        if current < required:
            warnings.append(
                f"This workflow requires Scope >= {min_version_str} "
                f"(installed: {current})"
            )
    except (InvalidVersion, importlib.metadata.PackageNotFoundError):
        warnings.append(f"Could not verify min_scope_version '{min_version_str}'")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def resolve_workflow(
    workflow: WorkflowRequest,
    plugin_manager: PluginManager,
    lora_dir: Path,
) -> WorkflowResolutionPlan:
    """Resolve all dependencies for *workflow*.

    This is a **read-only** operation -- no downloads, no installs.
    """
    items: list[ResolutionItem] = []
    warnings: list[str] = []
    all_pipelines_ok = True

    try:
        plugins = plugin_manager.list_plugins_sync()
    except Exception:
        logger.warning(
            "Failed to list plugins; treating all plugins as missing", exc_info=True
        )
        plugins = []
        warnings.append(
            "Could not read installed plugins. Plugin status may be inaccurate."
        )

    for wp in workflow.pipelines:
        if wp.source.type == "builtin":
            item = _resolve_builtin(wp.pipeline_id)
        else:
            item = _resolve_plugin(wp, plugins)

        items.append(item)
        if item.status in ("missing", "version_mismatch"):
            all_pipelines_ok = False

        for lora in wp.loras:
            items.append(_resolve_lora(lora, lora_dir))

    if workflow.min_scope_version:
        _check_min_scope_version(workflow.min_scope_version, warnings)

    return WorkflowResolutionPlan(
        can_apply=all_pipelines_ok,
        items=items,
        warnings=warnings,
    )
