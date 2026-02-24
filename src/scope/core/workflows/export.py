"""Build a ScopeWorkflow from the current server state."""

from __future__ import annotations

import importlib.metadata
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schema import (
    ScopeWorkflow,
    WorkflowLoRA,
    WorkflowLoRAProvenance,
    WorkflowMetadata,
    WorkflowPipeline,
    WorkflowPipelineSource,
)

logger = logging.getLogger(__name__)

# Maps plugin_manager source strings to WorkflowPipelineSource.type values.
_SOURCE_TYPE_MAP: dict[str, str] = {
    "pypi": "pypi",
    "git": "git",
    "local": "local",
}


def build_workflow(
    *,
    name: str,
    description: str,
    author: str,
    pipeline_manager: Any,
    plugin_manager: Any,
    lora_dir: Path,
    frontend_params: dict[str, dict[str, Any]] | None = None,
) -> ScopeWorkflow:
    """Snapshot the currently-loaded pipelines into a :class:`ScopeWorkflow`.

    Parameters
    ----------
    name, description, author:
        User-supplied metadata for the workflow.
    pipeline_manager:
        The running ``PipelineManager`` instance.
    plugin_manager:
        The running ``PluginManager`` instance.
    lora_dir:
        Absolute path to the LoRA directory (used to relativise LoRA paths
        and to load the manifest for provenance data).
    frontend_params:
        Optional dict keyed by pipeline_id mapping to that pipeline's
        frontend-supplied runtime parameters.  Each pipeline's params are
        merged into its ``params`` dict.  The ``"loras"`` and
        ``"lora_merge_mode"`` keys are stripped (they are promoted to the
        top-level ``WorkflowLoRA`` list).
    """
    from scope.core.lora.manifest import load_manifest
    from scope.core.pipelines.registry import PipelineRegistry

    manifest = load_manifest(lora_dir)

    # Resolve plugin info once
    plugin_list: list[dict[str, Any]] | None = None

    def _plugin_info(package_name: str) -> dict[str, Any] | None:
        nonlocal plugin_list
        if plugin_list is None:
            plugin_list = plugin_manager.list_plugins_sync()
        for info in plugin_list:
            if info.get("name") == package_name:
                return info
        return None

    snapshot = pipeline_manager.get_load_snapshot()

    # If no pipelines are loaded, seed from frontend_params keys so
    # the export still captures the user's selected pipeline/settings.
    if not snapshot and frontend_params:
        snapshot = {pid: {} for pid in frontend_params}

    pipelines: list[WorkflowPipeline] = []

    for pipeline_id, load_params in snapshot.items():
        load_params = dict(load_params)  # work on a copy

        # --- pipeline version ---
        config_class = PipelineRegistry.get_config_class(pipeline_id)
        pipeline_version = config_class.pipeline_version if config_class else "unknown"

        # --- source ---
        package_name = plugin_manager.get_plugin_for_pipeline(pipeline_id)
        if package_name is None:
            source = WorkflowPipelineSource(type="builtin")
        else:
            info = _plugin_info(package_name)
            plugin_source = info.get("source", "") if info else ""
            source = WorkflowPipelineSource(
                type=_SOURCE_TYPE_MAP.get(plugin_source, "pypi"),
                plugin_name=package_name,
                plugin_version=info.get("version") if info else None,
                package_spec=info.get("package_spec") if info else None,
            )

        # --- params (merge frontend_params first so LoRAs are available) ---
        params = dict(load_params)
        if frontend_params and pipeline_id in frontend_params:
            params.update(frontend_params[pipeline_id])

        # --- LoRAs (extract from merged params) ---
        raw_loras: list[dict[str, Any]] = params.pop("loras", None) or []
        lora_merge_mode: str = params.pop("lora_merge_mode", "permanent_merge")
        workflow_loras: list[WorkflowLoRA] = []

        for lora in raw_loras:
            lora_path = Path(lora.get("path", ""))
            # Relativise against lora_dir when possible
            try:
                filename = str(lora_path.relative_to(lora_dir))
            except ValueError:
                filename = lora_path.name or str(lora_path)

            # Normalise to forward-slash
            filename = filename.replace("\\", "/")

            entry = manifest.entries.get(filename)
            wl = WorkflowLoRA(
                filename=filename,
                weight=lora.get("weight", 1.0),
                merge_mode=lora_merge_mode,
                provenance=(
                    WorkflowLoRAProvenance.model_validate(entry.provenance.model_dump())
                    if entry
                    else None
                ),
                expected_sha256=entry.sha256 if entry else None,
            )
            workflow_loras.append(wl)

        pipelines.append(
            WorkflowPipeline(
                pipeline_id=pipeline_id,
                pipeline_version=pipeline_version,
                source=source,
                loras=workflow_loras,
                params=params,
            )
        )

    scope_version = importlib.metadata.version("daydream-scope")

    return ScopeWorkflow(
        metadata=WorkflowMetadata(
            name=name,
            description=description,
            author=author,
            created_at=datetime.now(UTC),
            scope_version=scope_version,
        ),
        pipelines=pipelines,
    )
