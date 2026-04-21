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
import re
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI

from .agent_state import AgentSession, WorkflowProposal
from .graph_schema import GraphConfig

# Valid ui_state edge handle IDs look like 'param:noise_scale', 'stream:video',
# 'param:__vace', 'param:trigger_a'. The literal prefix 'parameter:' is invalid.
_HANDLE_RE = re.compile(r"^(param|stream):[A-Za-z0-9_.\-]+$")

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

    async def get_pipeline_handles(self, pipeline_id: str) -> dict:
        """Return the exact React Flow handle IDs available on a pipeline node.

        The agent MUST call this before writing any ui_state edge whose target
        is a pipeline: the answer tells it which ``param:<name>`` and
        ``stream:<name>`` handles actually exist on the node, including
        aggregate handles (``param:__prompt`` / ``param:__vace`` /
        ``param:__loras``) that only appear when the pipeline declares the
        matching capability.

        Derived from the pipeline's config_schema + supports_* flags. Mirrors
        the frontend's ``extractParameterPorts`` in ``graphUtils.ts``.
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
        return _derive_pipeline_handles(pipeline_id, schema)

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
                "hint": (
                    "Top-level nodes only accept type source|pipeline|sink|"
                    "record. UI nodes (slider, subgraph, primitive, ...) go "
                    "under ui_state.nodes."
                ),
            }
        errors = cfg.validate_structure()
        if errors:
            return {"ok": False, "error": "invalid graph", "issues": errors}

        # Pre-flight validate ui_state + pipeline handles. This is where the
        # agent gets actionable feedback about bad edge handles, missing
        # targets, subgraph inconsistencies, and likely-missing wires.
        pipeline_handles: dict[str, dict] = {}
        for node in graph.get("nodes", []) or []:
            if node.get("type") == "pipeline" and node.get("pipeline_id"):
                pid = node["pipeline_id"]
                if pid in pipeline_handles:
                    continue
                try:
                    pipeline_handles[pid] = await self.get_pipeline_handles(pid)
                except Exception as e:
                    logger.warning(f"handle lookup failed for {pid}: {e}")
                    pipeline_handles[pid] = {"error": str(e)}
        issues = _validate_proposal(graph, pipeline_handles)
        errors_only = [i for i in issues if i.get("severity") == "error"]
        warnings_only = [i for i in issues if i.get("severity") == "warning"]
        if errors_only:
            return {
                "ok": False,
                "error": "graph failed structural validation",
                "issues": errors_only,
                "warnings": warnings_only,
                "hint": (
                    "Fix each error (handle format is 'param:<name>' or "
                    "'stream:<name>'; call get_pipeline_handles to see valid "
                    "pipeline inputs) and call propose_workflow again."
                ),
            }

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
            "warnings": warnings_only,
            "note": (
                "Proposal registered. The frontend will show Approve/Reject "
                "to the user. End your turn after proposing; on approval a "
                "new turn will be started with user feedback."
                + (
                    " NOTE: the validator flagged warnings you may want to "
                    "revisit before approval."
                    if warnings_only
                    else ""
                )
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


def _validate_proposal(
    graph: dict,
    pipeline_handles: dict[str, dict],
) -> list[dict]:
    """Validate a proposed workflow graph and return a list of issues.

    Pure function — takes the graph dict plus a lookup of pipeline handles
    already fetched by the caller. Makes it easy to unit test without a
    running FastAPI app.

    Issue shape: ``{"severity": "error"|"warning", "message": str,
    "edge_id": str | None, "node_id": str | None}``.
    """
    issues: list[dict] = []

    backend_nodes = graph.get("nodes", []) or []
    backend_node_ids: set[str] = {n.get("id") for n in backend_nodes if n.get("id")}
    backend_pipeline_by_id: dict[str, dict] = {
        n["id"]: n for n in backend_nodes if n.get("type") == "pipeline" and n.get("id")
    }

    ui_state = graph.get("ui_state") or {}
    if not isinstance(ui_state, dict):
        ui_state = {}
    ui_nodes = ui_state.get("nodes", []) or []
    ui_edges = ui_state.get("edges", []) or []

    ui_node_by_id: dict[str, dict] = {n["id"]: n for n in ui_nodes if n.get("id")}
    top_level_node_ids = backend_node_ids | set(ui_node_by_id.keys())

    # Track VACE wiring so we can emit a warning if none reaches a pipeline.
    vace_nodes: list[str] = []
    vace_to_pipeline_edges: int = 0
    image_nodes: set[str] = set()

    for n in ui_nodes:
        t = n.get("type")
        nid = n.get("id")
        if not nid:
            issues.append(
                {
                    "severity": "error",
                    "message": "ui_state.nodes entry is missing 'id'",
                    "node_id": None,
                    "edge_id": None,
                }
            )
            continue
        if t == "vace":
            vace_nodes.append(nid)
        elif t == "image":
            image_nodes.add(nid)

    # Validate subgraph internal consistency.
    for n in ui_nodes:
        if n.get("type") != "subgraph":
            continue
        nid = n["id"]
        data = n.get("data") or {}
        sg_nodes = data.get("subgraphNodes") or []
        sg_edges = data.get("subgraphEdges") or []
        sg_inputs = data.get("subgraphInputs") or []
        sg_outputs = data.get("subgraphOutputs") or []
        inner_ids = {sn.get("id") for sn in sg_nodes if sn.get("id")}

        for e in sg_edges:
            eid = e.get("id")
            if e.get("source") not in inner_ids:
                issues.append(
                    {
                        "severity": "error",
                        "node_id": nid,
                        "edge_id": eid,
                        "message": (
                            f"subgraph '{nid}' subgraphEdge references missing "
                            f"source '{e.get('source')}' (not in subgraphNodes)"
                        ),
                    }
                )
            if e.get("target") not in inner_ids:
                issues.append(
                    {
                        "severity": "error",
                        "node_id": nid,
                        "edge_id": eid,
                        "message": (
                            f"subgraph '{nid}' subgraphEdge references missing "
                            f"target '{e.get('target')}' (not in subgraphNodes)"
                        ),
                    }
                )
            for side in ("sourceHandle", "targetHandle"):
                h = e.get(side)
                if h is not None and not _HANDLE_RE.match(str(h)):
                    issues.append(
                        {
                            "severity": "error",
                            "node_id": nid,
                            "edge_id": eid,
                            "message": (
                                f"subgraph '{nid}' subgraphEdge {side}='{h}' "
                                "is not a valid handle; use "
                                "'param:<name>' or 'stream:<name>'"
                            ),
                        }
                    )

        for port in sg_inputs + sg_outputs:
            inner = port.get("innerNodeId")
            if inner and inner not in inner_ids:
                issues.append(
                    {
                        "severity": "error",
                        "node_id": nid,
                        "edge_id": None,
                        "message": (
                            f"subgraph '{nid}' exposes port "
                            f"'{port.get('name')}' whose innerNodeId "
                            f"'{inner}' is not in subgraphNodes"
                        ),
                    }
                )

    # Build a per-subgraph set of exposed port names for external-edge checks.
    def _subgraph_port_names(node: dict, kind: str) -> set[str]:
        """kind in {"inputs", "outputs"}"""
        data = node.get("data") or {}
        key = "subgraphInputs" if kind == "inputs" else "subgraphOutputs"
        return {p.get("name") for p in (data.get(key) or []) if p.get("name")}

    # Validate ui_state.edges.
    for e in ui_edges:
        eid = e.get("id")
        src = e.get("source")
        tgt = e.get("target")
        src_h = e.get("sourceHandle")
        tgt_h = e.get("targetHandle")

        for side, handle in (("sourceHandle", src_h), ("targetHandle", tgt_h)):
            if handle is None:
                continue
            if not _HANDLE_RE.match(str(handle)):
                issues.append(
                    {
                        "severity": "error",
                        "edge_id": eid,
                        "node_id": None,
                        "message": (
                            f"ui_state.edge {side}='{handle}' is not a valid "
                            "handle; use 'param:<name>' or 'stream:<name>' "
                            "(the prefix 'parameter:' is invalid)"
                        ),
                    }
                )

        if src not in top_level_node_ids:
            issues.append(
                {
                    "severity": "error",
                    "edge_id": eid,
                    "node_id": src,
                    "message": (
                        f"ui_state.edge source '{src}' does not exist at "
                        "top level nor in ui_state.nodes (edges between "
                        "inner subgraph nodes live in that subgraph's "
                        "data.subgraphEdges, not ui_state.edges)"
                    ),
                }
            )
        if tgt not in top_level_node_ids:
            issues.append(
                {
                    "severity": "error",
                    "edge_id": eid,
                    "node_id": tgt,
                    "message": (
                        f"ui_state.edge target '{tgt}' does not exist at "
                        "top level nor in ui_state.nodes"
                    ),
                }
            )

        # Pipeline-target checks: verify handle exists on pipeline.
        if tgt in backend_pipeline_by_id:
            pipe_node = backend_pipeline_by_id[tgt]
            pid = pipe_node.get("pipeline_id")
            ph = pipeline_handles.get(pid) if pid else None
            if ph and "error" not in ph:
                valid = set(ph.get("stream_inputs") or []) | {
                    p.get("handle") for p in (ph.get("param_inputs") or [])
                }
                if tgt_h and tgt_h not in valid:
                    issues.append(
                        {
                            "severity": "error",
                            "edge_id": eid,
                            "node_id": tgt,
                            "message": (
                                f"pipeline '{pid}' has no input handle "
                                f"'{tgt_h}'. Valid handles: "
                                f"{sorted(valid)}"
                            ),
                        }
                    )

        # Track VACE → pipeline wires.
        src_node = ui_node_by_id.get(src)
        if (
            src_node
            and src_node.get("type") == "vace"
            and tgt in backend_pipeline_by_id
            and tgt_h == "param:__vace"
        ):
            vace_to_pipeline_edges += 1

        # Validate subgraph external ports (external edge handles must
        # match the subgraph's declared inputs/outputs).
        src_sg = ui_node_by_id.get(src) if src in ui_node_by_id else None
        tgt_sg = ui_node_by_id.get(tgt) if tgt in ui_node_by_id else None
        if src_sg and src_sg.get("type") == "subgraph" and src_h:
            port_name = str(src_h).split(":", 1)[1] if ":" in src_h else src_h
            outs = _subgraph_port_names(src_sg, "outputs")
            if port_name not in outs:
                issues.append(
                    {
                        "severity": "error",
                        "edge_id": eid,
                        "node_id": src,
                        "message": (
                            f"subgraph '{src}' has no declared output "
                            f"'{port_name}'. Expected one of: {sorted(outs)}"
                        ),
                    }
                )
        if tgt_sg and tgt_sg.get("type") == "subgraph" and tgt_h:
            port_name = str(tgt_h).split(":", 1)[1] if ":" in tgt_h else tgt_h
            ins = _subgraph_port_names(tgt_sg, "inputs")
            if port_name not in ins:
                issues.append(
                    {
                        "severity": "error",
                        "edge_id": eid,
                        "node_id": tgt,
                        "message": (
                            f"subgraph '{tgt}' has no declared input "
                            f"'{port_name}'. Expected one of: {sorted(ins)}"
                        ),
                    }
                )

    # Soft warnings — likely-missing wires.
    if vace_nodes and vace_to_pipeline_edges == 0:
        issues.append(
            {
                "severity": "warning",
                "node_id": vace_nodes[0],
                "edge_id": None,
                "message": (
                    f"vace node(s) {vace_nodes} present but none is wired to a "
                    "pipeline's 'param:__vace'. Add an edge "
                    "{source: <vace_id>, sourceHandle: 'param:__vace', "
                    "target: <pipeline_id>, targetHandle: 'param:__vace'}."
                ),
            }
        )

    for vid in vace_nodes:
        has_ref = any(
            e.get("target") == vid
            and str(e.get("targetHandle") or "")
            in ("param:ref_image", "param:first_frame", "param:last_frame")
            for e in ui_edges
        )
        if not has_ref:
            issues.append(
                {
                    "severity": "warning",
                    "node_id": vid,
                    "edge_id": None,
                    "message": (
                        f"vace node '{vid}' has no image input wired. Connect "
                        "an 'image' node's 'param:value' into 'param:ref_image' "
                        "(or param:first_frame / param:last_frame)."
                    ),
                }
            )

    # Warn if any pipeline supports prompts but nothing reaches its __prompt.
    for pid_node, pipe_node in backend_pipeline_by_id.items():
        pid = pipe_node.get("pipeline_id")
        ph = pipeline_handles.get(pid) if pid else None
        if not ph or not ph.get("supports_prompts"):
            continue
        reaches_prompt = any(
            e.get("target") == pid_node and e.get("targetHandle") == "param:__prompt"
            for e in ui_edges
        )
        if not reaches_prompt:
            issues.append(
                {
                    "severity": "warning",
                    "node_id": pid_node,
                    "edge_id": None,
                    "message": (
                        f"pipeline '{pid}' supports prompts but nothing is "
                        "wired to its 'param:__prompt'. Add a primitive/"
                        "subgraph/prompt_blend whose output feeds "
                        "'param:__prompt'."
                    ),
                }
            )

    return issues


def _derive_pipeline_handles(pipeline_id: str, schema: dict) -> dict:
    """Produce the list of handle IDs targetable on a pipeline node.

    Mirrors frontend/src/lib/graphUtils.ts::extractParameterPorts + the
    aggregate handles rendered by PipelineNode.tsx. Kept in Python so the
    agent doesn't need to read frontend code.
    """
    supports_prompts = bool(schema.get("supports_prompts"))
    supports_vace = bool(schema.get("supports_vace"))
    supports_lora = bool(schema.get("supports_lora"))
    produces_video = schema.get("produces_video", True)
    produces_audio = bool(schema.get("produces_audio"))

    props = ((schema.get("config_schema") or {}).get("properties") or {}) or {}

    param_inputs: list[dict] = []
    seen_names: set[str] = set()

    for name, prop in props.items():
        if not isinstance(prop, dict):
            continue
        ui = prop.get("ui")
        if not isinstance(ui, dict):
            # Fields without ui metadata don't get rendered as param handles.
            continue
        component = ui.get("component")
        if component in ("cache", "vace", "lora"):
            # These collapse into aggregate handles or are reset buttons; skip.
            continue

        type_hint = _infer_param_type(prop)
        if type_hint is None:
            continue

        entry = {
            "handle": f"param:{name}",
            "field": name,
            "type": type_hint,
            "modulatable": bool(ui.get("modulatable")),
            "is_load_param": bool(ui.get("is_load_param")),
        }
        if "modulatable_min" in ui:
            entry["modulatable_min"] = ui["modulatable_min"]
        if "modulatable_max" in ui:
            entry["modulatable_max"] = ui["modulatable_max"]
        if ui.get("modes"):
            entry["modes"] = list(ui["modes"])
        param_inputs.append(entry)
        seen_names.add(name)

    # Aggregate handles — only present on pipelines with the matching flag.
    if supports_prompts:
        param_inputs.append(
            {
                "handle": "param:__prompt",
                "aggregate": True,
                "type": "string",
                "note": (
                    "Aggregate prompt input. Connect any string-valued output "
                    "here (primitive, subgraph output, prompt_blend.prompts) "
                    "to replace the built-in prompt text."
                ),
            }
        )
    if supports_vace:
        param_inputs.append(
            {
                "handle": "param:__vace",
                "aggregate": True,
                "type": "vace",
                "note": (
                    "Aggregate VACE input. Connect a 'vace' node's "
                    "'param:__vace' output here."
                ),
            }
        )
    if supports_lora:
        param_inputs.append(
            {
                "handle": "param:__loras",
                "aggregate": True,
                "type": "lora",
                "note": (
                    "Aggregate LoRA input. Connect a 'lora' node's "
                    "'param:lora' output here."
                ),
            }
        )

    # Stream inputs.
    stream_inputs = ["stream:video"]
    if supports_vace:
        stream_inputs.extend(["stream:vace_input_frames", "stream:vace_input_masks"])

    # Stream outputs.
    stream_outputs: list[str] = []
    if produces_video:
        stream_outputs.append("stream:video")
    if produces_audio:
        stream_outputs.append("stream:audio")

    return {
        "pipeline_id": pipeline_id,
        "supports_prompts": supports_prompts,
        "supports_vace": supports_vace,
        "supports_lora": supports_lora,
        "stream_inputs": stream_inputs,
        "stream_outputs": stream_outputs,
        "param_inputs": param_inputs,
        "note": (
            "Use exactly these handle IDs in ui_state.edges. Format is "
            "'param:<name>' or 'stream:<name>' — never 'parameter:<name>'."
        ),
    }


def _infer_param_type(prop: dict) -> str | None:
    """Classify a JSON Schema property into a coarse handle type."""
    any_of = prop.get("anyOf")
    t = prop.get("type")

    # Direct array type.
    if t == "array":
        items = prop.get("items") or {}
        if isinstance(items, dict) and items.get("type") in ("integer", "number"):
            return "list_number"
        return "array"

    # anyOf with array variant (e.g. list[int] | None).
    if isinstance(any_of, list):
        for v in any_of:
            if not isinstance(v, dict):
                continue
            if v.get("type") == "array":
                items = v.get("items") or {}
                if isinstance(items, dict) and items.get("type") in (
                    "integer",
                    "number",
                ):
                    return "list_number"

    # Enum or $ref (treated as string).
    if prop.get("enum") or prop.get("$ref"):
        return "string"
    if isinstance(any_of, list):
        for v in any_of:
            if isinstance(v, dict) and v.get("$ref"):
                return "string"

    if t in ("integer", "number"):
        return "number"
    if isinstance(any_of, list):
        for v in any_of:
            if isinstance(v, dict) and v.get("type") in ("integer", "number"):
                return "number"

    if t == "boolean":
        return "boolean"
    if isinstance(any_of, list):
        for v in any_of:
            if isinstance(v, dict) and v.get("type") == "boolean":
                return "boolean"

    if t == "string":
        return "string"
    if isinstance(any_of, list):
        for v in any_of:
            if isinstance(v, dict) and v.get("type") == "string":
                return "string"

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
            "name": "get_pipeline_handles",
            "description": (
                "Return the exact React Flow handle IDs available on a "
                "pipeline node (stream_inputs, stream_outputs, param_inputs). "
                "Call this BEFORE writing any ui_state edge whose target is a "
                "pipeline, so you wire to handles that actually exist. "
                "Includes aggregate handles 'param:__prompt' (only if "
                "supports_prompts), 'param:__vace' (only if supports_vace), "
                "'param:__loras' (only if supports_lora), and VACE stream "
                "inputs ('stream:vace_input_frames', 'stream:vace_input_masks') "
                "for VACE-capable pipelines."
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
                "render an Approve/Reject card. The tool validates the "
                "proposal (handle format, node existence, pipeline handle "
                "presence, subgraph consistency) — if it returns errors, fix "
                "them and re-call propose_workflow. Warnings are informational "
                "and may indicate likely-missing wires (e.g. a VACE node with "
                "no edge to the pipeline)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "graph": {
                        "type": "object",
                        "description": (
                            "GraphConfig with two parts:\n"
                            "1) Top-level 'nodes'/'edges' (backend runtime "
                            "flow) — ONLY node types source|pipeline|sink|"
                            "record. Backend edges use {from, from_port, "
                            "to_node, to_port, kind: 'stream'|'parameter'}.\n"
                            "2) 'ui_state': {nodes, edges} — everything else. "
                            "UI node types include trigger, subgraph, "
                            "primitive, slider, knobs, math, midi, "
                            "prompt_list, prompt_blend, vace, lora, image, "
                            "etc. UI edges use React Flow shape: "
                            "{id, source, sourceHandle, target, targetHandle} "
                            "where handle IDs have the shape 'param:<name>' "
                            "(value port) or 'stream:<name>' (frame/audio "
                            "port). The literal prefix 'parameter:' is "
                            "INVALID — always use 'param:'. Before writing "
                            "any ui_state edge that targets a pipeline node, "
                            "call get_pipeline_handles(pipeline_id) to get "
                            "the exact handle IDs. UI nodes placed in "
                            "top-level 'nodes' will fail pydantic validation. "
                            "Blueprint nodes always go under ui_state. Call "
                            "get_current_graph on a loaded workflow to see "
                            "the split in practice."
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
    "get_pipeline_handles": "get_pipeline_handles",
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
