"""Tests for agent_tool_impls helpers.

Focuses on the pure-function validator and pipeline-handle deriver so we
don't need to stand up a FastAPI app.
"""

from __future__ import annotations

from typing import Any

from scope.server.agent_tool_impls import (
    _derive_pipeline_handles,
    _validate_proposal,
)

# ---------------------------------------------------------------------------
# Fixtures — compact pipeline handle dicts that mirror _derive_pipeline_handles
# output. Using these directly keeps validator tests isolated from schema
# shape churn.
# ---------------------------------------------------------------------------


def _vace_pipeline_handles() -> dict[str, Any]:
    return {
        "pipeline_id": "krea-realtime-video",
        "supports_prompts": True,
        "supports_vace": True,
        "supports_lora": True,
        "stream_inputs": [
            "stream:video",
            "stream:vace_input_frames",
            "stream:vace_input_masks",
        ],
        "stream_outputs": ["stream:video"],
        "param_inputs": [
            {"handle": "param:noise_scale", "field": "noise_scale", "type": "number"},
            {
                "handle": "param:denoising_steps",
                "field": "denoising_steps",
                "type": "list_number",
            },
            {"handle": "param:__prompt", "aggregate": True, "type": "string"},
            {"handle": "param:__vace", "aggregate": True, "type": "vace"},
            {"handle": "param:__loras", "aggregate": True, "type": "lora"},
        ],
    }


def _simple_pipeline_handles() -> dict[str, Any]:
    return {
        "pipeline_id": "longlive",
        "supports_prompts": True,
        "supports_vace": False,
        "supports_lora": False,
        "stream_inputs": ["stream:video"],
        "stream_outputs": ["stream:video"],
        "param_inputs": [
            {"handle": "param:noise_scale", "field": "noise_scale", "type": "number"},
            {"handle": "param:__prompt", "aggregate": True, "type": "string"},
        ],
    }


def _minimal_backend(pipeline_id: str = "longlive") -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "input", "type": "source"},
            {"id": "pipe", "type": "pipeline", "pipeline_id": pipeline_id},
            {"id": "output", "type": "sink"},
        ],
        "edges": [
            {
                "from": "input",
                "from_port": "video",
                "to_node": "pipe",
                "to_port": "video",
                "kind": "stream",
            },
            {
                "from": "pipe",
                "from_port": "video",
                "to_node": "output",
                "to_port": "video",
                "kind": "stream",
            },
        ],
    }


# ---------------------------------------------------------------------------
# _validate_proposal
# ---------------------------------------------------------------------------


def test_validate_rejects_invalid_handle_prefix():
    graph = _minimal_backend()
    graph["ui_state"] = {
        "nodes": [
            {
                "id": "slider_noise",
                "type": "slider",
                "data": {"value": 0.7},
            },
        ],
        "edges": [
            {
                "id": "e1",
                "source": "slider_noise",
                "sourceHandle": "parameter:value",  # INVALID prefix
                "target": "pipe",
                "targetHandle": "param:noise_scale",
            }
        ],
    }
    issues = _validate_proposal(graph, {"longlive": _simple_pipeline_handles()})
    errors = [i for i in issues if i["severity"] == "error"]
    assert any("parameter:" in i["message"] for i in errors), errors
    assert any(i.get("edge_id") == "e1" for i in errors)


def test_validate_rejects_missing_target_node():
    graph = _minimal_backend()
    graph["ui_state"] = {
        "nodes": [
            {"id": "slider_noise", "type": "slider", "data": {}},
        ],
        "edges": [
            {
                "id": "e_missing",
                "source": "slider_noise",
                "sourceHandle": "param:value",
                "target": "nonexistent_pipe",
                "targetHandle": "param:noise_scale",
            }
        ],
    }
    issues = _validate_proposal(graph, {"longlive": _simple_pipeline_handles()})
    errors = [i for i in issues if i["severity"] == "error"]
    assert any(
        "does not exist" in i["message"] and i.get("edge_id") == "e_missing"
        for i in errors
    ), errors


def test_validate_rejects_unknown_pipeline_handle():
    graph = _minimal_backend()
    graph["ui_state"] = {
        "nodes": [
            {"id": "slider_wild", "type": "slider", "data": {}},
        ],
        "edges": [
            {
                "id": "e_bad_handle",
                "source": "slider_wild",
                "sourceHandle": "param:value",
                "target": "pipe",
                "targetHandle": "param:does_not_exist_on_pipeline",
            }
        ],
    }
    issues = _validate_proposal(graph, {"longlive": _simple_pipeline_handles()})
    errors = [i for i in issues if i["severity"] == "error"]
    assert any("no input handle" in i["message"] for i in errors), (
        f"expected unknown-handle error, got: {errors}"
    )


def test_validate_accepts_good_graph():
    graph = _minimal_backend()
    graph["ui_state"] = {
        "nodes": [
            {"id": "slider_noise", "type": "slider", "data": {}},
            {
                "id": "prompt_text",
                "type": "primitive",
                "data": {"valueType": "string"},
            },
        ],
        "edges": [
            {
                "id": "e_noise",
                "source": "slider_noise",
                "sourceHandle": "param:value",
                "target": "pipe",
                "targetHandle": "param:noise_scale",
            },
            {
                "id": "e_prompt",
                "source": "prompt_text",
                "sourceHandle": "param:value",
                "target": "pipe",
                "targetHandle": "param:__prompt",
            },
        ],
    }
    issues = _validate_proposal(graph, {"longlive": _simple_pipeline_handles()})
    errors = [i for i in issues if i["severity"] == "error"]
    assert errors == [], errors


def test_validate_warns_on_unreached_vace():
    graph = _minimal_backend("krea-realtime-video")
    graph["ui_state"] = {
        "nodes": [
            {"id": "vace_1", "type": "vace", "data": {}},
            {"id": "ref_img", "type": "image", "data": {}},
        ],
        "edges": [
            # Image → VACE only. No VACE → pipeline wire.
            {
                "id": "e_img_vace",
                "source": "ref_img",
                "sourceHandle": "param:value",
                "target": "vace_1",
                "targetHandle": "param:ref_image",
            },
        ],
    }
    issues = _validate_proposal(
        graph, {"krea-realtime-video": _vace_pipeline_handles()}
    )
    warnings = [i for i in issues if i["severity"] == "warning"]
    assert any("param:__vace" in i["message"] for i in warnings), (
        f"expected VACE wiring warning, got: {warnings}"
    )


def test_validate_warns_on_unreached_prompt_input():
    graph = _minimal_backend()
    graph["ui_state"] = {"nodes": [], "edges": []}
    issues = _validate_proposal(graph, {"longlive": _simple_pipeline_handles()})
    warnings = [i for i in issues if i["severity"] == "warning"]
    assert any("param:__prompt" in i["message"] for i in warnings), (
        f"expected unreached-prompt warning, got: {warnings}"
    )


def test_validate_subgraph_internal_consistency():
    graph = _minimal_backend()
    graph["ui_state"] = {
        "nodes": [
            {
                "id": "sg",
                "type": "subgraph",
                "data": {
                    "subgraphNodes": [
                        {"id": "inner_primitive", "type": "primitive", "data": {}},
                        {"id": "inner_control", "type": "control", "data": {}},
                    ],
                    "subgraphEdges": [
                        {
                            "id": "se1",
                            "source": "inner_primitive",
                            "sourceHandle": "param:value",
                            "target": "missing_inner",  # not in subgraphNodes
                            "targetHandle": "param:str_0",
                        }
                    ],
                    "subgraphInputs": [
                        {
                            "name": "trigger_a",
                            "portType": "param",
                            "paramType": "number",
                            "innerNodeId": "nonexistent_inner",  # bad
                            "innerHandleId": "param:item_0",
                        }
                    ],
                    "subgraphOutputs": [
                        {
                            "name": "prompt",
                            "portType": "param",
                            "paramType": "string",
                            "innerNodeId": "inner_control",
                            "innerHandleId": "param:value",
                        }
                    ],
                },
            }
        ],
        "edges": [],
    }
    issues = _validate_proposal(graph, {"longlive": _simple_pipeline_handles()})
    errors = [i for i in issues if i["severity"] == "error"]
    assert any(
        "subgraphEdge references missing target" in i["message"] for i in errors
    ), errors
    assert any(
        "innerNodeId 'nonexistent_inner' is not in subgraphNodes" in i["message"]
        for i in errors
    ), errors


def test_validate_subgraph_external_edge_must_match_declared_port():
    graph = _minimal_backend()
    graph["ui_state"] = {
        "nodes": [
            {
                "id": "sg",
                "type": "subgraph",
                "data": {
                    "subgraphNodes": [
                        {"id": "inner_ctrl", "type": "control", "data": {}},
                    ],
                    "subgraphEdges": [],
                    "subgraphInputs": [],
                    "subgraphOutputs": [
                        {
                            "name": "prompt",
                            "portType": "param",
                            "paramType": "string",
                            "innerNodeId": "inner_ctrl",
                            "innerHandleId": "param:value",
                        }
                    ],
                },
            },
        ],
        "edges": [
            # External edge refers to a port that the subgraph does not expose.
            {
                "id": "e_bad_port",
                "source": "sg",
                "sourceHandle": "param:does_not_exist",
                "target": "pipe",
                "targetHandle": "param:__prompt",
            },
        ],
    }
    issues = _validate_proposal(graph, {"longlive": _simple_pipeline_handles()})
    errors = [i for i in issues if i["severity"] == "error"]
    assert any("no declared output" in i["message"] for i in errors), errors


# ---------------------------------------------------------------------------
# _derive_pipeline_handles
# ---------------------------------------------------------------------------


def test_derive_handles_includes_aggregates_when_supported():
    schema = {
        "supports_prompts": True,
        "supports_vace": True,
        "supports_lora": True,
        "produces_video": True,
        "config_schema": {
            "properties": {
                "noise_scale": {
                    "type": "number",
                    "ui": {
                        "category": "configuration",
                        "is_load_param": False,
                    },
                },
                # Fields with these components are aggregated and skipped.
                "vace_context_scale": {
                    "type": "number",
                    "ui": {"component": "vace"},
                },
                "manage_cache": {
                    "type": "boolean",
                    "ui": {"component": "cache"},
                },
            }
        },
    }
    result = _derive_pipeline_handles("krea-realtime-video", schema)
    handles = {p["handle"] for p in result["param_inputs"]}
    assert "param:noise_scale" in handles
    assert "param:vace_context_scale" not in handles
    assert "param:manage_cache" not in handles
    assert "param:__prompt" in handles
    assert "param:__vace" in handles
    assert "param:__loras" in handles
    assert "stream:vace_input_frames" in result["stream_inputs"]
    assert "stream:vace_input_masks" in result["stream_inputs"]


def test_derive_handles_omits_aggregates_when_unsupported():
    schema = {
        "supports_prompts": False,
        "supports_vace": False,
        "supports_lora": False,
        "produces_video": True,
        "config_schema": {
            "properties": {
                "noise_scale": {"type": "number", "ui": {"is_load_param": False}},
            }
        },
    }
    result = _derive_pipeline_handles("plain", schema)
    handles = {p["handle"] for p in result["param_inputs"]}
    assert "param:noise_scale" in handles
    assert "param:__prompt" not in handles
    assert "param:__vace" not in handles
    assert "param:__loras" not in handles
    assert "stream:vace_input_frames" not in result["stream_inputs"]


def test_derive_handles_ignores_fields_without_ui():
    schema = {
        "supports_prompts": False,
        "supports_vace": False,
        "supports_lora": False,
        "produces_video": True,
        "config_schema": {
            "properties": {
                "internal_only": {"type": "number"},  # no ui metadata
            }
        },
    }
    result = _derive_pipeline_handles("internal", schema)
    handles = {p["handle"] for p in result["param_inputs"]}
    assert "param:internal_only" not in handles
