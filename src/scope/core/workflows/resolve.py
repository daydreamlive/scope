"""Dependency resolution for imported workflows.

Given a :class:`ScopeWorkflow`, check which pipelines, plugins, and LoRAs
are available locally and produce a :class:`WorkflowResolutionPlan` that the
frontend can display in a trust-gate dialog before any side-effects occur.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from packaging.version import InvalidVersion, Version
from pydantic import BaseModel

from scope.core.pipelines.registry import PipelineRegistry

from .schema import ScopeWorkflow

logger = logging.getLogger(__name__)


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


def is_load_param(config_class: type, field_name: str) -> bool:
    """Check whether *field_name* is a load param on *config_class*."""
    field_info = config_class.model_fields.get(field_name)
    if field_info is None:
        return False
    extra = field_info.json_schema_extra
    if callable(extra):
        return False
    if isinstance(extra, dict):
        ui = extra.get("ui", {})
        return ui.get("is_load_param", False)
    return False


def _check_settings(
    config_class: type,
    params: dict[str, Any],
) -> list[str]:
    """Validate *params* against *config_class* fields, return warnings.

    Parameters not present in the pipeline config schema are silently
    ignored — they are frontend runtime params that get returned via
    ``runtime_params`` on apply.
    """
    warnings: list[str] = []
    # Future: validate known fields have compatible types, etc.
    return warnings


def _check_min_scope_version(
    min_version_str: str,
    warnings: list[str],
) -> None:
    """Compare *min_version_str* against the installed Scope version."""
    import importlib.metadata

    try:
        current_version = Version(importlib.metadata.version("daydream-scope"))
        min_version = Version(min_version_str)
        if current_version < min_version:
            warnings.append(
                f"This workflow requires Scope >= {min_version_str} "
                f"(installed: {current_version})"
            )
    except (InvalidVersion, importlib.metadata.PackageNotFoundError):
        warnings.append(f"Could not verify min_scope_version '{min_version_str}'")


def resolve_workflow(
    workflow: ScopeWorkflow,
    plugin_manager: Any,
    models_dir: Path,
) -> WorkflowResolutionPlan:
    """Resolve all dependencies for *workflow*.

    This is a **read-only** operation — no downloads, no installs.

    Parameters
    ----------
    workflow:
        The parsed workflow to resolve.
    plugin_manager:
        The running ``PluginManager`` instance (used for plugin checks).
    models_dir:
        Root models directory (LoRAs live under ``models_dir / "lora"``).
    """

    from ._utils import find_plugin_info, get_plugin_list

    items: list[ResolutionItem] = []
    settings_warnings: list[str] = []
    all_pipelines_ok = True

    plugins = get_plugin_list(plugin_manager)

    lora_dir = models_dir / "lora"

    for wp in workflow.pipelines:
        # --- Pipeline / plugin resolution ---
        if wp.source.type == "builtin":
            if PipelineRegistry.is_registered(wp.pipeline_id):
                items.append(
                    ResolutionItem(
                        kind="pipeline",
                        name=wp.pipeline_id,
                        status="ok",
                    )
                )
            else:
                items.append(
                    ResolutionItem(
                        kind="pipeline",
                        name=wp.pipeline_id,
                        status="missing",
                        detail=f"Built-in pipeline '{wp.pipeline_id}' not found",
                    )
                )
                all_pipelines_ok = False
        else:
            # Plugin-provided pipeline
            plugin_name = wp.source.plugin_name
            if plugin_name is None:
                items.append(
                    ResolutionItem(
                        kind="plugin",
                        name=wp.pipeline_id,
                        status="missing",
                        detail="No plugin name specified in workflow",
                    )
                )
                all_pipelines_ok = False
                continue

            # Find the plugin in installed list
            installed_info = find_plugin_info(plugins, plugin_name)

            if installed_info is None:
                # Build install action
                if wp.source.type == "git" and wp.source.package_spec:
                    action = f"Install from git: {wp.source.package_spec}"
                else:
                    spec = plugin_name
                    if wp.source.plugin_version:
                        spec += f">={wp.source.plugin_version}"
                    action = f"Install {spec} from PyPI"

                items.append(
                    ResolutionItem(
                        kind="plugin",
                        name=plugin_name,
                        status="missing",
                        detail=f"Plugin '{plugin_name}' is not installed",
                        action=action,
                        can_auto_resolve=True,
                    )
                )
                all_pipelines_ok = False
            else:
                # Plugin installed — check version if workflow specifies one
                installed_version = installed_info.get("version")
                workflow_version = wp.source.plugin_version

                if workflow_version and installed_version:
                    try:
                        iv = Version(installed_version)
                        wv = Version(workflow_version)
                        if iv < wv:
                            items.append(
                                ResolutionItem(
                                    kind="plugin",
                                    name=plugin_name,
                                    status="version_mismatch",
                                    detail=f"Installed {installed_version}, workflow expects {workflow_version}",
                                    action=f"Upgrade {plugin_name} to >={workflow_version}",
                                    can_auto_resolve=True,
                                )
                            )
                            # Version mismatch doesn't block — pipeline might still work
                        else:
                            items.append(
                                ResolutionItem(
                                    kind="plugin",
                                    name=plugin_name,
                                    status="ok",
                                )
                            )
                    except InvalidVersion:
                        # Can't parse version — treat as ok with a warning
                        items.append(
                            ResolutionItem(
                                kind="plugin",
                                name=plugin_name,
                                status="ok",
                                detail=f"Could not compare versions (installed={installed_version})",
                            )
                        )
                else:
                    items.append(
                        ResolutionItem(
                            kind="plugin",
                            name=plugin_name,
                            status="ok",
                        )
                    )

        # --- LoRA resolution ---
        for lora in wp.loras:
            lora_path = lora_dir / lora.filename
            if lora_path.exists():
                items.append(
                    ResolutionItem(
                        kind="lora",
                        name=lora.filename,
                        status="ok",
                    )
                )
            else:
                has_provenance = (
                    lora.provenance is not None and lora.provenance.source != "local"
                )
                if has_provenance:
                    prov = lora.provenance
                    if prov.source == "huggingface":
                        action = f"Download from HuggingFace: {prov.repo_id}"
                    elif prov.source == "civitai":
                        action = f"Download from CivitAI (model {prov.model_id})"
                    else:
                        action = (
                            f"Download from {prov.url}"
                            if prov.url
                            else "Download from source"
                        )
                else:
                    action = None

                items.append(
                    ResolutionItem(
                        kind="lora",
                        name=lora.filename,
                        status="missing",
                        detail=f"LoRA '{lora.filename}' not found locally",
                        action=action,
                        can_auto_resolve=has_provenance,
                    )
                )

        # --- Settings validation ---
        config_class = PipelineRegistry.get_config_class(wp.pipeline_id)
        if config_class is not None:
            settings_warnings.extend(_check_settings(config_class, wp.params))

    # --- min_scope_version check ---
    if workflow.min_scope_version:
        _check_min_scope_version(workflow.min_scope_version, settings_warnings)

    return WorkflowResolutionPlan(
        can_apply=all_pipelines_ok,
        items=items,
        warnings=settings_warnings,
    )
