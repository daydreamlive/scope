"""Build a ScopeWorkflow from frontend-supplied pipeline configuration."""

from __future__ import annotations

import importlib.metadata
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ._utils import find_plugin_info, get_plugin_list
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
    pipelines_input: list[dict[str, Any]],
    plugin_manager: Any,
    lora_dir: Path,
    lora_merge_mode: str = "permanent_merge",
) -> ScopeWorkflow:
    """Build a :class:`ScopeWorkflow` from frontend-supplied pipeline data.

    Parameters
    ----------
    name:
        User-supplied name for the workflow.
    pipelines_input:
        List of dicts, each with ``pipeline_id``, ``params``, and ``loras``.
        The frontend is the source of truth for what the user configured.
    plugin_manager:
        The running ``PluginManager`` instance (for source enrichment).
    lora_dir:
        Absolute path to the LoRA directory (used to relativise LoRA paths
        and to load the manifest for provenance data).
    lora_merge_mode:
        LoRA merge strategy applied to all pipelines.
    """
    from scope.core.lora.manifest import load_manifest
    from scope.core.pipelines.registry import PipelineRegistry

    manifest = load_manifest(lora_dir)

    plugins = get_plugin_list(plugin_manager)

    pipelines: list[WorkflowPipeline] = []

    for pipeline_input in pipelines_input:
        pipeline_id = pipeline_input["pipeline_id"]
        params = dict(pipeline_input.get("params", {}))
        raw_loras: list[dict[str, Any]] = pipeline_input.get("loras", [])

        # --- pipeline version ---
        config_class = PipelineRegistry.get_config_class(pipeline_id)
        if config_class is None:
            logger.warning("Unknown pipeline %r; cannot validate params", pipeline_id)
        pipeline_version = config_class.pipeline_version if config_class else None

        # --- filter params against pipeline config schema ---
        if config_class is not None and hasattr(config_class, "model_fields"):
            known_keys = set(config_class.model_fields)
            unknown_keys = set(params) - known_keys
            if unknown_keys:
                logger.debug(
                    "Dropping unknown params for %s: %s", pipeline_id, unknown_keys
                )
                params = {k: v for k, v in params.items() if k in known_keys}

        # --- source ---
        package_name = plugin_manager.get_plugin_for_pipeline(pipeline_id)
        if package_name is None:
            source = WorkflowPipelineSource(type="builtin")
        else:
            info = find_plugin_info(plugins, package_name)
            plugin_source = info.get("source", "") if info else ""
            source = WorkflowPipelineSource(
                type=_SOURCE_TYPE_MAP.get(plugin_source, "pypi"),
                plugin_name=package_name,
                plugin_version=info.get("version") if info else None,
                package_spec=info.get("package_spec") if info else None,
            )

        # --- LoRAs ---
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
                weight=lora.get("scale", 1.0),
                merge_mode=lora.get("merge_mode") or lora_merge_mode,
                provenance=(
                    WorkflowLoRAProvenance.model_validate(entry.provenance.model_dump())
                    if entry
                    else None
                ),
                sha256=entry.sha256 if entry else None,
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
            created_at=datetime.now(UTC),
            scope_version=scope_version,
        ),
        pipelines=pipelines,
    )
