"""Apply a resolved workflow to the running server.

After :func:`resolve_workflow` produces a :class:`WorkflowResolutionPlan`,
this module converts the workflow's settings into the ``load_pipelines()``
call and returns runtime params for the frontend to push via WebRTC.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from scope.core.pipelines.registry import PipelineRegistry

from .resolve import WorkflowResolutionPlan, is_load_param
from .schema import ScopeWorkflow

logger = logging.getLogger(__name__)


class ApplyResult(BaseModel):
    """Result of applying a workflow."""

    applied: bool
    pipeline_ids: list[str]
    skipped_loras: list[str] = []
    runtime_params: dict[str, Any] = {}
    restart_required: bool = False
    message: str = ""


async def apply_workflow(
    workflow: ScopeWorkflow,
    plan: WorkflowResolutionPlan,
    pipeline_manager: Any,
    plugin_manager: Any,
    models_dir: Path,
    *,
    install_missing_plugins: bool = False,
    skip_missing_loras: bool = True,
) -> ApplyResult:
    """Apply *workflow* to the running server.

    Parameters
    ----------
    workflow:
        The parsed workflow.
    plan:
        Resolution plan from :func:`resolve_workflow`.
    pipeline_manager:
        The running ``PipelineManager`` instance.
    plugin_manager:
        The running ``PluginManager`` instance.
    models_dir:
        Root models directory.
    install_missing_plugins:
        If True, install missing plugins before loading.
    skip_missing_loras:
        If True, skip LoRAs that aren't available locally.
    """

    if not plan.can_apply and not install_missing_plugins:
        return ApplyResult(
            applied=False,
            pipeline_ids=[],
            message="Cannot apply: missing required pipelines or plugins",
        )

    # --- Install missing plugins if requested ---
    restart_required = False
    if install_missing_plugins:
        for item in plan.items:
            if item.kind == "plugin" and item.status == "missing":
                # Determine package spec
                matching_pipeline = next(
                    (
                        wp
                        for wp in workflow.pipelines
                        if wp.source.plugin_name == item.name
                    ),
                    None,
                )
                if matching_pipeline:
                    source = matching_pipeline.source
                    if source.type == "git" and source.package_spec:
                        package = source.package_spec
                    else:
                        package = source.plugin_name or item.name
                    try:
                        await plugin_manager.install_plugin_async(package)
                        restart_required = True
                    except Exception:
                        logger.exception(f"Failed to install plugin '{package}'")
                        return ApplyResult(
                            applied=False,
                            pipeline_ids=[],
                            message=f"Failed to install plugin '{package}'",
                        )

    if restart_required:
        return ApplyResult(
            applied=False,
            pipeline_ids=[wp.pipeline_id for wp in workflow.pipelines],
            restart_required=True,
            message="Plugins installed. Server restart required to load them.",
        )

    # --- Build load_params and identify runtime_params ---
    # ``load_pipelines()`` accepts a single ``load_params`` dict shared across
    # all pipelines.  We merge params from every pipeline in order so that the
    # primary (last) pipeline's values win on conflict — this matches the
    # existing frontend behaviour where the main pipeline's settings dominate.
    lora_dir = models_dir / "lora"

    load_params: dict[str, Any] = {}
    runtime_params: dict[str, Any] = {}
    skipped_loras: list[str] = []
    loaded_merge_modes: list[str] = []

    for wp in workflow.pipelines:
        config_class = PipelineRegistry.get_config_class(wp.pipeline_id)

        # Classify params
        for key, value in wp.params.items():
            if config_class and is_load_param(config_class, key):
                load_params[key] = value
            else:
                runtime_params[key] = value

        # Build LoRA list (accumulate across pipelines)
        loras_for_load: list[dict[str, Any]] = load_params.get("loras", [])
        for lora in wp.loras:
            lora_path = lora_dir / lora.filename
            if lora_path.exists():
                # Verify SHA256 if the workflow specifies an expected hash
                if lora.sha256:
                    from scope.core.lora.manifest import compute_sha256

                    actual = compute_sha256(lora_path)
                    if actual != lora.sha256:
                        logger.warning(
                            "SHA256 mismatch for LoRA '%s' (expected %s..., got %s...)",
                            lora.filename,
                            lora.sha256[:12],
                            actual[:12],
                        )

                loras_for_load.append(
                    {
                        "path": str(lora_path),
                        "weight": lora.weight,
                    }
                )
                loaded_merge_modes.append(lora.merge_mode)
            elif skip_missing_loras:
                skipped_loras.append(lora.filename)
            else:
                return ApplyResult(
                    applied=False,
                    pipeline_ids=[],
                    message=f"Missing LoRA: {lora.filename}",
                )

        if loras_for_load:
            load_params["loras"] = loras_for_load

    # Determine merge mode from actually-loaded LoRAs only
    if loaded_merge_modes:
        unique_modes = set(loaded_merge_modes)
        if len(unique_modes) > 1:
            logger.warning(
                "Workflow contains LoRAs with conflicting merge modes: %s. "
                "Using '%s' (from first loaded LoRA).",
                unique_modes,
                loaded_merge_modes[0],
            )
        load_params["lora_merge_mode"] = loaded_merge_modes[0]

    pipeline_ids = [wp.pipeline_id for wp in workflow.pipelines]

    # --- Load pipelines ---
    try:
        success = await pipeline_manager.load_pipelines(pipeline_ids, load_params)
    except Exception:
        logger.exception("Failed to load pipelines from workflow")
        return ApplyResult(
            applied=False,
            pipeline_ids=pipeline_ids,
            message="Failed to load pipelines",
        )

    if not success:
        return ApplyResult(
            applied=False,
            pipeline_ids=pipeline_ids,
            message="Pipeline loading returned failure",
        )

    msg_parts = [f"Loaded {len(pipeline_ids)} pipeline(s)"]
    if skipped_loras:
        msg_parts.append(f", skipped {len(skipped_loras)} missing LoRA(s)")

    return ApplyResult(
        applied=True,
        pipeline_ids=pipeline_ids,
        skipped_loras=skipped_loras,
        runtime_params=runtime_params,
        message="".join(msg_parts),
    )
