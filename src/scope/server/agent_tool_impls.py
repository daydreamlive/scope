"""Shared tool implementations for the Scope agent.

This module exposes Scope's capabilities as agent tools. It uses an in-process
httpx client (via ``httpx.ASGITransport``) to call the FastAPI app's existing
HTTP endpoints. This matches the pattern used by ``mcp_server.py`` (which hits
the same endpoints over loopback HTTP), keeping behavior consistent between
external MCP clients and the in-app agent.

Each tool method:
- Returns a JSON-serializable dict (plus base64 image for capture_frame).
- Keeps payloads small so tool results stay well under provider context limits.

The loop (``agent_loop.py``) owns provider/session/SSE concerns. This module
is intentionally side-effect-light and does not know about Anthropic or SSE.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI

from .agent_state import AgentSession, WorkflowProposal
from .graph_schema import GraphConfig

logger = logging.getLogger(__name__)


# Maximum size (bytes) for a tool-result JSON payload sent to the model.
# Truncate more aggressively on anything that lists files.
MAX_PAYLOAD_BYTES = 16 * 1024


def _truncate_list(items: list, cap: int = 40) -> tuple[list, dict | None]:
    """Cap a list and return (items, info_or_none)."""
    if len(items) <= cap:
        return items, None
    return items[:cap], {"truncated": True, "total": len(items), "shown": cap}


def _canonical_graph_hash(graph: dict) -> str:
    """Stable sha256 of a canonical JSON form. Used to detect graph edits."""
    blob = json.dumps(graph, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class AgentTools:
    """Callable bundle of tools the agent can invoke.

    One instance per AgentLoop turn. Holds a short-lived httpx client wired to
    the FastAPI app via ASGITransport so calls go in-process (no socket hop).
    """

    def __init__(self, app: FastAPI, session: AgentSession) -> None:
        self._app = app
        self._session = session
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> AgentTools:
        transport = httpx.ASGITransport(app=self._app)
        self._client = httpx.AsyncClient(
            transport=transport,
            base_url="http://scope-agent.local",
            timeout=300.0,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _c(self) -> httpx.AsyncClient:
        assert self._client is not None, "AgentTools must be used as async ctx mgr"
        return self._client

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    async def list_pipelines(self) -> dict:
        """Return all registered pipelines with minimal metadata.

        Includes plugin-provided pipelines (e.g. ltx2, helios) — they register
        against the same registry and appear here with no special-casing.
        """
        resp = await self._c().get("/api/v1/pipelines/schemas")
        resp.raise_for_status()
        data = resp.json()

        schemas = data.get("pipelines", {}) or {}
        pipelines = []
        for pipeline_id, schema in schemas.items():
            pipelines.append(
                {
                    "id": pipeline_id,
                    "name": schema.get("name"),
                    "description": schema.get("description"),
                    "supports_prompts": schema.get("supports_prompts", False),
                    "supports_lora": schema.get("supports_lora", False),
                    "supports_vace": schema.get("supports_vace", False),
                    "supported_modes": schema.get("supported_modes", []),
                }
            )
        return {"pipelines": pipelines, "count": len(pipelines)}

    async def get_pipeline_schema(self, pipeline_id: str) -> dict:
        """Return the full Pydantic schema for a pipeline, including UI hints.

        This is the authoritative source of truth for what fields the agent
        can safely set via update_parameters or include in a workflow.
        """
        resp = await self._c().get("/api/v1/pipelines/schemas")
        resp.raise_for_status()
        schemas = resp.json().get("pipelines", {}) or {}
        schema = schemas.get(pipeline_id)
        if schema is None:
            return {
                "error": f"pipeline '{pipeline_id}' not found",
                "available": sorted(schemas.keys()),
            }
        return schema

    async def list_loras(self) -> dict:
        resp = await self._c().get("/api/v1/loras")
        resp.raise_for_status()
        data = resp.json()
        files = data.get("files", []) or []
        files, info = _truncate_list(files, cap=50)
        summary = {
            "loras": [
                {"name": f.get("name"), "path": f.get("path"), "size": f.get("size")}
                for f in files
            ]
        }
        if info:
            summary["pagination"] = info
        return summary

    async def list_assets(self) -> dict:
        resp = await self._c().get("/api/v1/assets")
        resp.raise_for_status()
        data = resp.json()
        assets = data.get("assets", data.get("files", [])) or []
        assets, info = _truncate_list(assets, cap=50)
        summary = {
            "assets": [
                {
                    "name": a.get("name"),
                    "path": a.get("path"),
                    "type": a.get("type"),
                    "size": a.get("size"),
                }
                for a in assets
            ]
        }
        if info:
            summary["pagination"] = info
        return summary

    async def list_plugins(self) -> dict:
        resp = await self._c().get("/api/v1/plugins")
        resp.raise_for_status()
        data = resp.json()
        plugins = data.get("plugins", []) or []
        return {
            "plugins": [
                {
                    "name": p.get("name"),
                    "version": p.get("version"),
                    "pipelines": p.get("pipelines", []),
                }
                for p in plugins
            ]
        }

    async def list_blueprints(self) -> dict:
        """List frontend composable blueprints (workflow fragments).

        Reads from ``frontend/src/data/blueprints/`` in dev installs; returns
        an empty list with a note in packaged/plugin-only installs.
        """
        candidates = _find_blueprints_dir()
        if candidates is None:
            return {
                "blueprints": [],
                "note": "Blueprints directory not available in this install.",
            }
        results = []
        for path in sorted(candidates.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            results.append(
                {
                    "id": path.stem,
                    "name": data.get("name", path.stem),
                    "description": data.get("description", ""),
                    "category": data.get("category", "misc"),
                }
            )
        return {"blueprints": results, "count": len(results)}

    async def get_blueprint(self, blueprint_id: str) -> dict:
        """Return the full JSON for a blueprint so the agent can graft it."""
        candidates = _find_blueprints_dir()
        if candidates is None:
            return {"error": "blueprints not available"}
        path = candidates / f"{blueprint_id}.json"
        if not path.exists():
            return {"error": f"blueprint '{blueprint_id}' not found"}
        try:
            return {"id": blueprint_id, "blueprint": json.loads(path.read_text())}
        except Exception as e:
            return {"error": f"failed to load blueprint: {e}"}

    async def list_node_types(self) -> dict:
        """Return the UI-node catalog so the agent can compose graphs.

        Sourced from ``frontend/src/data/nodes/manifest.json``. New node types
        are registered there without touching agent code.
        """
        manifest = _find_node_manifest()
        if manifest is None:
            return {"node_types": [], "note": "node manifest not available"}
        try:
            data = json.loads(manifest.read_text())
        except Exception as e:
            return {"node_types": [], "error": f"failed to read manifest: {e}"}
        return data

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    async def get_pipeline_status(self) -> dict:
        resp = await self._c().get("/api/v1/pipeline/status")
        resp.raise_for_status()
        return resp.json()

    async def get_current_graph(self) -> dict:
        """Return a best-effort snapshot of the currently running graph.

        The backend tracks the active session graph inside SessionContext; we
        expose it through /api/v1/session/metrics which includes sink ids.
        When no graph is running, returns None.
        """
        resp = await self._c().get("/api/v1/session/metrics")
        if resp.status_code != 200:
            return {"graph": None, "graph_hash": None, "running": False}
        data = resp.json()
        graph = data.get("graph")
        if not graph:
            return {"graph": None, "graph_hash": None, "running": False}
        return {
            "graph": graph,
            "graph_hash": _canonical_graph_hash(graph),
            "running": True,
        }

    async def get_session_metrics(self) -> dict:
        resp = await self._c().get("/api/v1/session/metrics")
        if resp.status_code != 200:
            return {"error": "no session", "status_code": resp.status_code}
        return resp.json()

    async def get_hardware_info(self) -> dict:
        resp = await self._c().get("/api/v1/hardware/info")
        resp.raise_for_status()
        return resp.json()

    async def get_logs(self, lines: int = 100, level: str | None = None) -> dict:
        params: dict[str, Any] = {"lines": max(1, min(lines, 500))}
        resp = await self._c().get("/api/v1/logs/tail", params=params)
        resp.raise_for_status()
        data = resp.json()
        log_lines = data.get("lines", [])
        if level:
            level_u = level.upper()
            log_lines = [ln for ln in log_lines if level_u in ln]
        return {"lines": log_lines[-lines:]}

    # ------------------------------------------------------------------
    # Vision
    # ------------------------------------------------------------------

    async def capture_frame(
        self, sink_node_id: str | None = None, quality: int = 80
    ) -> dict:
        """Capture current frame as base64 JPEG so it can be fed back to a
        multimodal model as tool_result image content."""
        params: dict[str, Any] = {"quality": max(1, min(quality, 95))}
        if sink_node_id:
            params["sink_node_id"] = sink_node_id
        resp = await self._c().get("/api/v1/session/frame", params=params)
        if resp.status_code != 200:
            return {
                "error": "frame capture failed",
                "status_code": resp.status_code,
                "detail": resp.text[:200] if resp.text else None,
            }
        data = resp.content
        b64 = base64.b64encode(data).decode("ascii")
        return {
            "media_type": "image/jpeg",
            "size_bytes": len(data),
            "base64": b64,
            "sink_node_id": sink_node_id,
        }

    # ------------------------------------------------------------------
    # Runtime control (auto-applied)
    # ------------------------------------------------------------------

    async def update_parameters(self, parameters: dict) -> dict:
        """Apply runtime parameter updates (prompts, noise, LoRA scales, etc)."""
        resp = await self._c().post("/api/v1/session/parameters", json=parameters or {})
        if resp.status_code != 200:
            return {
                "ok": False,
                "status_code": resp.status_code,
                "detail": resp.text[:400] if resp.text else None,
            }
        return {"ok": True, "applied": list((parameters or {}).keys())}

    async def load_pipeline(
        self, pipeline_ids: list[str], load_params: dict | None = None
    ) -> dict:
        body: dict[str, Any] = {"pipeline_ids": list(pipeline_ids)}
        if load_params:
            body["load_params"] = load_params
        resp = await self._c().post("/api/v1/pipeline/load", json=body)
        if resp.status_code != 200:
            return {
                "ok": False,
                "status_code": resp.status_code,
                "detail": resp.text[:400] if resp.text else None,
            }
        return {"ok": True, **(resp.json() if resp.text else {})}

    async def start_recording(self, node_id: str | None = None) -> dict:
        params = {"node_id": node_id} if node_id else None
        resp = await self._c().post("/api/v1/recordings/headless/start", params=params)
        if resp.status_code != 200:
            return {"ok": False, "detail": resp.text[:400]}
        return {"ok": True, **(resp.json() if resp.text else {})}

    async def stop_recording(self, node_id: str | None = None) -> dict:
        params = {"node_id": node_id} if node_id else None
        resp = await self._c().post("/api/v1/recordings/headless/stop", params=params)
        if resp.status_code != 200:
            return {"ok": False, "detail": resp.text[:400]}
        return {"ok": True, **(resp.json() if resp.text else {})}

    async def list_recordings(self) -> dict:
        resp = await self._c().get("/api/v1/recordings/headless")
        if resp.status_code != 200:
            return {"recordings": [], "note": "no active recording"}
        return {
            "download_url": "/api/v1/recordings/headless",
            "size_bytes": int(resp.headers.get("content-length", 0)),
        }

    # ------------------------------------------------------------------
    # Workflow proposal handshake
    # ------------------------------------------------------------------

    async def propose_workflow(
        self,
        graph: dict,
        rationale: str,
        pipeline_load_params: dict | None = None,
        input_mode: str = "video",
    ) -> dict:
        """Validate a graph and stage it as a pending proposal.

        The proposal is emitted via SSE to the frontend, which renders a card
        with Approve/Reject. On Approve, the loop will call apply_workflow.
        On Reject, the agent gets a next-turn user message with the reason.
        """
        try:
            cfg = GraphConfig(**graph)
        except Exception as e:
            return {
                "ok": False,
                "error": f"graph failed pydantic validation: {e}",
            }
        errors = cfg.validate_structure()
        if errors:
            return {"ok": False, "error": "invalid graph", "issues": errors}

        # Hash the graph-at-propose-time so later apply can detect user edits.
        graph_hash = _canonical_graph_hash(graph)
        proposal_id = f"prop_{uuid.uuid4().hex[:10]}"

        # Best-effort diff against the currently running graph.
        diff = _diff_graphs(await self._running_graph_or_none(), graph)

        proposal = WorkflowProposal(
            id=proposal_id,
            graph=graph,
            graph_hash_at_propose=graph_hash,
            rationale=rationale,
            pipeline_load_params=pipeline_load_params or {},
            input_mode=input_mode,
            diff=diff,
        )
        self._session.pending_proposal = proposal

        # Distinct pipelines that need to be loaded for this graph.
        pipelines = sorted(
            {
                n.get("pipeline_id")
                for n in graph.get("nodes", [])
                if n.get("type") == "pipeline" and n.get("pipeline_id")
            }
        )

        return {
            "ok": True,
            "proposal_id": proposal_id,
            "graph_hash": graph_hash,
            "rationale": rationale,
            "pipelines_to_load": pipelines,
            "diff": diff,
            "note": (
                "Proposal registered. The frontend will show Approve/Reject "
                "to the user. End your turn after proposing; on approval a "
                "new turn will be started with user feedback."
            ),
        }

    async def apply_workflow(
        self,
        proposal_id: str,
        expected_graph_hash: str,
    ) -> dict:
        """Confirm that a previously-proposed workflow was applied.

        The frontend writes the proposed graph into the React Flow canvas at
        approval time (before this tool runs). This tool just validates the
        hash, clears the pending proposal, and returns so the agent can end
        its turn with a short confirmation message.

        It intentionally does NOT start a session or load pipelines — the
        user presses Play to start, which runs the regular flow (including
        cloud routing when cloud mode is active). That keeps the agent out
        of environment-specific concerns and matches the "confirm workflows,
        user controls Play" product intent.
        """
        proposal = self._session.pending_proposal
        if proposal is None or proposal.id != proposal_id:
            return {"ok": False, "error": "no matching pending proposal"}
        if not proposal.approved:
            return {"ok": False, "error": "proposal has not been approved by user"}

        # Detect user edits during review: if the frontend recomputed the hash
        # at approve-time and it changed, bail and invite re-proposal.
        if expected_graph_hash != proposal.graph_hash_at_propose:
            return {
                "ok": False,
                "error": "graph changed since proposal; re-propose",
                "expected": expected_graph_hash,
                "actual": proposal.graph_hash_at_propose,
            }

        # Distinct pipelines that will be loaded when the user presses Play —
        # surface them so the agent can mention what's about to warm up.
        pipelines = sorted(
            {
                n.get("pipeline_id")
                for n in proposal.graph.get("nodes", [])
                if n.get("type") == "pipeline" and n.get("pipeline_id")
            }
        )

        # Clear proposal bookkeeping.
        self._session.pending_proposal = None
        return {
            "ok": True,
            "applied_to_canvas": True,
            "pipelines_in_graph": pipelines,
            "note": (
                "Graph has been written to the canvas. The user will press "
                "Play to start the session; do not try to start it yourself."
            ),
        }

    async def stop_session(self) -> dict:
        resp = await self._c().post("/api/v1/session/stop")
        return {"ok": resp.status_code == 200, "status_code": resp.status_code}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _running_graph_or_none(self) -> dict | None:
        try:
            resp = await self._c().get("/api/v1/session/metrics")
            if resp.status_code != 200:
                return None
            return resp.json().get("graph")
        except Exception:
            return None


# ----------------------------------------------------------------------
# Lookup helpers for filesystem-backed resources
# ----------------------------------------------------------------------


def _find_blueprints_dir() -> Path | None:
    """Locate frontend/src/data/blueprints relative to the running package."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "frontend" / "src" / "data" / "blueprints"
        if candidate.is_dir():
            return candidate
    return None


def _find_node_manifest() -> Path | None:
    """Locate frontend/src/data/nodes/manifest.json relative to the package."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "frontend" / "src" / "data" / "nodes" / "manifest.json"
        if candidate.is_file():
            return candidate
    return None


def _diff_graphs(current: dict | None, proposed: dict) -> dict:
    """Return a human-readable summary of how proposed differs from current."""
    if current is None:
        return {
            "summary": "no active graph; proposal creates a new session",
            "added_nodes": [n.get("id") for n in proposed.get("nodes", [])],
            "removed_nodes": [],
        }

    cur_ids = {n.get("id") for n in current.get("nodes", [])}
    new_ids = {n.get("id") for n in proposed.get("nodes", [])}
    added = sorted(new_ids - cur_ids)
    removed = sorted(cur_ids - new_ids)
    return {
        "summary": (
            f"+{len(added)} nodes, -{len(removed)} nodes "
            f"(was {len(cur_ids)}, now {len(new_ids)})"
        ),
        "added_nodes": added,
        "removed_nodes": removed,
    }


# ----------------------------------------------------------------------
# Tool specs (Anthropic-shaped; the OpenAI provider translates as needed)
# ----------------------------------------------------------------------


def build_tool_specs() -> list[dict]:
    """Return the tool specs the provider layer advertises to the LLM.

    Shape follows Anthropic's tool-use spec. The OpenAI-compatible provider
    translates each entry into the OpenAI function-calling format.
    """
    return [
        {
            "name": "list_pipelines",
            "description": (
                "List all registered pipelines. Always call this early in a "
                "turn so you know what's available; new pipelines (e.g. LTX2) "
                "show up here without any code changes."
            ),
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_pipeline_schema",
            "description": (
                "Return the full Pydantic config schema for a pipeline, "
                "including field types, ranges, UI hints, and supports_* "
                "capability flags. Call this before proposing a workflow or "
                "calling update_parameters so you use the real parameter names."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"pipeline_id": {"type": "string"}},
                "required": ["pipeline_id"],
            },
        },
        {
            "name": "list_loras",
            "description": "List installed LoRA adapter files with paths.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "list_assets",
            "description": "List images/videos in the assets directory.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "list_plugins",
            "description": "List installed Scope plugins and their pipelines.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "list_blueprints",
            "description": (
                "List pre-composed UI graph fragments (prompt switcher, LFO, "
                "timed cycler, etc.). Prefer grafting a blueprint over "
                "rebuilding composite behavior from raw nodes."
            ),
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_blueprint",
            "description": (
                "Return the full JSON of a blueprint. Its nodes/edges are "
                "UI-node types (trigger, subgraph, primitive, slider, etc.) "
                "— when grafting into propose_workflow, place them under "
                "ui_state.nodes / ui_state.edges, NOT top-level nodes/edges "
                "(top-level accepts only source|pipeline|sink|record)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"blueprint_id": {"type": "string"}},
                "required": ["blueprint_id"],
            },
        },
        {
            "name": "list_node_types",
            "description": (
                "Return the UI node-type catalog (slider, knobs, prompt_list, "
                "trigger, etc.) with their port signatures. Use when you need "
                "to compose a UI graph beyond the source/pipeline/sink/record "
                "primitives."
            ),
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_pipeline_status",
            "description": "Which pipelines are loaded / loading.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_current_graph",
            "description": (
                "Return the currently running graph and a stable hash. Use this "
                "before propose_workflow so you can diff against what's live."
            ),
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_session_metrics",
            "description": "fps/VRAM/frames_in/out for the active session.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_hardware_info",
            "description": "GPU VRAM and output-sink availability (Spout/NDI/Syphon).",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_logs",
            "description": "Recent server log lines, optionally filtered by level.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "lines": {"type": "integer", "default": 100},
                    "level": {
                        "type": "string",
                        "enum": ["DEBUG", "INFO", "WARNING", "ERROR"],
                    },
                },
                "required": [],
            },
        },
        {
            "name": "capture_frame",
            "description": (
                "Capture the current pipeline output as a JPEG. The response "
                "will be delivered back to you as an image so you can reason "
                "visually — use this when the user asks about what they're "
                "seeing or when tuning based on output quality."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "sink_node_id": {"type": "string"},
                    "quality": {"type": "integer", "default": 80},
                },
                "required": [],
            },
        },
        {
            "name": "update_parameters",
            "description": (
                "Apply runtime parameters (prompts, transition, noise_scale, "
                "denoising_step_list, kv_cache_attention_bias, lora_scales, "
                "vace_ref_images, vace_context_scale, input_source, "
                "output_sinks, paused, recording, pipeline-specific fields). "
                "Auto-applied immediately; use only for runtime changes, not "
                "graph structure."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"parameters": {"type": "object"}},
                "required": ["parameters"],
            },
        },
        {
            "name": "load_pipeline",
            "description": (
                "Load one or more pipelines. Usually invoked indirectly via "
                "apply_workflow; call directly only when the user explicitly "
                "asks to preload."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "pipeline_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "load_params": {"type": "object"},
                },
                "required": ["pipeline_ids"],
            },
        },
        {
            "name": "start_recording",
            "description": "Start recording the active session (optionally per record node).",
            "input_schema": {
                "type": "object",
                "properties": {"node_id": {"type": "string"}},
                "required": [],
            },
        },
        {
            "name": "stop_recording",
            "description": "Stop recording.",
            "input_schema": {
                "type": "object",
                "properties": {"node_id": {"type": "string"}},
                "required": [],
            },
        },
        {
            "name": "list_recordings",
            "description": "Return the download URL for the latest recording if any.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "propose_workflow",
            "description": (
                "Propose a new or replacement graph for the user to approve. "
                "Always use this for structural changes — never apply a graph "
                "without explicit user approval. End your turn with a short "
                "text summary after calling this tool; the frontend will "
                "render an Approve/Reject card."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "graph": {
                        "type": "object",
                        "description": (
                            "GraphConfig with two parts: top-level nodes/edges "
                            "(backend — ONLY node types source|pipeline|sink|"
                            "record; edge kind stream|parameter) AND an "
                            "optional ui_state: {nodes, edges} for all UI "
                            "nodes (trigger, subgraph, primitive, slider, "
                            "knobs, math, LFO, MIDI, prompt_list, etc.). UI "
                            "nodes placed in top-level nodes will fail "
                            "pydantic validation. Blueprint nodes always go "
                            "under ui_state. Call get_current_graph on a "
                            "loaded workflow to see the split in practice."
                        ),
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Short explanation shown to the user.",
                    },
                    "pipeline_load_params": {
                        "type": "object",
                        "description": "Optional load_params passed to pipeline/load.",
                    },
                    "input_mode": {
                        "type": "string",
                        "enum": ["text", "video"],
                        "default": "video",
                    },
                },
                "required": ["graph", "rationale"],
            },
        },
        {
            "name": "apply_workflow",
            "description": (
                "Confirm an approved proposal. The frontend already wrote the "
                "graph to the canvas at approval time; this tool just "
                "validates the hash and clears the pending proposal. It does "
                "NOT start a session or load pipelines — the user presses "
                "Play to start. Only callable after user approval (the loop "
                "will tell you with a [System] message)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "proposal_id": {"type": "string"},
                    "expected_graph_hash": {"type": "string"},
                },
                "required": ["proposal_id", "expected_graph_hash"],
            },
        },
        {
            "name": "stop_session",
            "description": "Stop the active headless session.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
    ]


# Dispatch table for agent_loop to resolve tool_name -> coroutine.
TOOL_METHODS = {
    "list_pipelines": "list_pipelines",
    "get_pipeline_schema": "get_pipeline_schema",
    "list_loras": "list_loras",
    "list_assets": "list_assets",
    "list_plugins": "list_plugins",
    "list_blueprints": "list_blueprints",
    "get_blueprint": "get_blueprint",
    "list_node_types": "list_node_types",
    "get_pipeline_status": "get_pipeline_status",
    "get_current_graph": "get_current_graph",
    "get_session_metrics": "get_session_metrics",
    "get_hardware_info": "get_hardware_info",
    "get_logs": "get_logs",
    "capture_frame": "capture_frame",
    "update_parameters": "update_parameters",
    "load_pipeline": "load_pipeline",
    "start_recording": "start_recording",
    "stop_recording": "stop_recording",
    "list_recordings": "list_recordings",
    "propose_workflow": "propose_workflow",
    "apply_workflow": "apply_workflow",
    "stop_session": "stop_session",
}
