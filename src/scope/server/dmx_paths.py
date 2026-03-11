"""DMX path inventory: numeric-only runtime parameters for DMX channel mapping.

Reuses the same pipeline-schema introspection as OSC docs but filters to only
numeric types (float, number, integer) since DMX channels carry 0-255 values
that must be scaled to a numeric range.
"""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pipeline_manager import PipelineManager

logger = logging.getLogger(__name__)

_NUMERIC_TYPES = {"float", "number", "integer"}

# Runtime params that are numeric and useful to control via DMX.
# Mirrors the curated list in osc_docs._RUNTIME_PARAMS but only numeric entries.
_RUNTIME_PARAMS: list[dict[str, Any]] = [
    {
        "key": "noise_scale",
        "type": "float",
        "description": "Noise scale",
        "min": 0.0,
        "max": 1.0,
    },
    {
        "key": "kv_cache_attention_bias",
        "type": "float",
        "description": "KV-cache attention bias",
        "min": 0.01,
        "max": 1.0,
    },
    {
        "key": "vace_context_scale",
        "type": "float",
        "description": "VACE hint injection scale",
        "min": 0.0,
        "max": 2.0,
    },
    {
        "key": "transition_steps",
        "type": "integer",
        "description": "Prompt transition steps (0 = instant)",
        "min": 0,
        "max": 100,
    },
]


def _extract_numeric_paths_from_schema(
    config_schema: dict,
    pipeline_id: str,
) -> list[dict[str, Any]]:
    """Extract numeric runtime-controllable paths from a pipeline config schema."""
    paths: list[dict[str, Any]] = []
    properties = config_schema.get("properties", {})
    for key, prop in properties.items():
        ui = prop.get("ui", {})
        if ui.get("is_load_param", True):
            continue
        prop_type = prop.get("type", "any")
        if prop_type not in _NUMERIC_TYPES:
            continue

        entry: dict[str, Any] = {
            "key": key,
            "type": prop_type,
            "description": prop.get("description", ""),
            "pipeline_id": pipeline_id,
        }
        if "minimum" in prop:
            entry["min"] = prop["minimum"]
        if "maximum" in prop:
            entry["max"] = prop["maximum"]

        # Provide sensible defaults when schema lacks explicit bounds
        if "min" not in entry:
            entry["min"] = 0 if prop_type == "integer" else 0.0
        if "max" not in entry:
            entry["max"] = 100 if prop_type == "integer" else 1.0

        paths.append(entry)
    return paths


def _collect_pipeline_paths() -> dict[str, list[dict[str, Any]]]:
    from scope.core.pipelines.registry import PipelineRegistry

    pipeline_paths: dict[str, list[dict[str, Any]]] = {}
    for pid in PipelineRegistry.list_pipelines():
        config_class = PipelineRegistry.get_config_class(pid)
        if not config_class:
            continue
        schema_data = config_class.get_schema_with_metadata()
        config_schema = schema_data.get("config_schema", {})
        paths = _extract_numeric_paths_from_schema(config_schema, pid)
        if paths:
            pipeline_paths[pid] = paths
    return pipeline_paths


def get_dmx_paths(
    pipeline_manager: "PipelineManager | None",
) -> dict[str, Any]:
    """Build the numeric DMX path inventory split into active / available."""
    active_pipeline_ids: list[str] = []
    if pipeline_manager:
        active_pipeline_ids = pipeline_manager.get_loaded_pipeline_ids()

    pipeline_paths = _collect_pipeline_paths()

    active_groups: dict[str, list[dict[str, Any]]] = {}
    available_groups: dict[str, list[dict[str, Any]]] = {}

    # Runtime params always active
    active_groups["Runtime"] = list(_RUNTIME_PARAMS)

    for pid, paths in pipeline_paths.items():
        target = active_groups if pid in active_pipeline_ids else available_groups
        target[pid] = paths

    return {
        "active": active_groups,
        "available": available_groups,
        "active_pipeline_ids": active_pipeline_ids,
    }


def get_all_numeric_paths(
    pipeline_manager: "PipelineManager | None",
) -> dict[str, dict[str, Any]]:
    """Flat dict mapping every known numeric key to its path metadata."""
    data = get_dmx_paths(pipeline_manager)
    result: dict[str, dict[str, Any]] = {}
    for groups in (data["active"], data["available"]):
        for paths in groups.values():
            for p in paths:
                result[p["key"]] = p
    return result
