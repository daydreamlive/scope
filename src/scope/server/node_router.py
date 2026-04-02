"""REST and WebSocket endpoints for backend node management.

Provides:
- ``GET    /types``                 – list registered node types with schemas
- ``GET    /instances``             – list active node instances
- ``POST   /instances/{id}/input``  – update a node input value
- ``POST   /instances/{id}/config`` – update node configuration
- ``DELETE /instances/{id}``        – destroy a node instance
- ``GET    /instances/{id}/state``  – get current node state
- ``GET    /instances/{id}/ports``  – get current ports (for dynamic-port nodes)
- ``WS     /ws``                    – real-time output state observation
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue as stdlib_queue
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/nodes", tags=["nodes"])


# ---------------------------------------------------------------------------
# Deferred import to avoid circular dependency with app.py
# ---------------------------------------------------------------------------


def _get_node_manager():
    from .app import node_manager

    return node_manager


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class NodeInputRequest(BaseModel):
    name: str
    value: Any


class NodeConfigRequest(BaseModel):
    config: dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/types")
async def list_node_types():
    """Return schemas for all registered backend node types."""
    from scope.core.nodes import NodeRegistry

    return NodeRegistry.get_all_schemas()


@router.get("/instances")
async def list_instances():
    """Return metadata for all active node instances."""
    mgr = _get_node_manager()
    if mgr is None:
        return []
    return mgr.list_instances()


@router.post("/instances/{instance_id}/input")
async def update_input(instance_id: str, body: NodeInputRequest):
    """Update an input value on a node instance."""
    mgr = _get_node_manager()
    if mgr is None:
        return {"ok": False, "error": "Node manager not initialised"}
    ok = mgr.update_input(instance_id, body.name, body.value)
    if not ok:
        return {"ok": False, "error": f"Instance '{instance_id}' not found"}
    return {"ok": True}


@router.post("/instances/{instance_id}/config")
async def update_config(instance_id: str, body: NodeConfigRequest):
    """Update runtime configuration on a node instance."""
    mgr = _get_node_manager()
    if mgr is None:
        return {"ok": False, "error": "Node manager not initialised"}
    ok = mgr.update_config(instance_id, body.config)
    if not ok:
        return {"ok": False, "error": f"Instance '{instance_id}' not found"}
    return {"ok": True}


@router.delete("/instances/{instance_id}")
async def delete_instance(instance_id: str):
    """Destroy a node instance and clean up its resources."""
    mgr = _get_node_manager()
    if mgr is None:
        return {"ok": False, "error": "Node manager not initialised"}
    ok = mgr.remove_instance(instance_id)
    return {"ok": ok}


@router.get("/instances/{instance_id}/state")
async def get_state(instance_id: str):
    """Return the current observable state of a node instance."""
    mgr = _get_node_manager()
    if mgr is None:
        return {"error": "Node manager not initialised"}
    state = mgr.get_instance_state(instance_id)
    if state is None:
        return {"error": f"Instance '{instance_id}' not found"}
    return state


@router.get("/instances/{instance_id}/ports")
async def get_ports(instance_id: str):
    """Return the current ports for a dynamic-port node instance."""
    mgr = _get_node_manager()
    if mgr is None:
        return {"error": "Node manager not initialised"}
    ports = mgr.get_instance_ports(instance_id)
    if ports is None:
        return {"error": f"Instance '{instance_id}' not found or no dynamic ports"}
    return ports


@router.websocket("/ws")
async def node_ws(websocket: WebSocket):
    """WebSocket for real-time node output observation.

    Messages are JSON objects with:
    ``{"type": "node_output", "instance_id": "...", "port": "...", "value": ...}``

    On connect the server sends a ``full_state`` snapshot so the client can
    initialise its display immediately.

    Uses a thread-safe ``queue.Queue`` so that node threads (e.g. scheduler
    playback) can safely enqueue messages without asyncio thread-safety issues.
    """
    mgr = _get_node_manager()
    await websocket.accept()

    q: stdlib_queue.Queue = stdlib_queue.Queue(maxsize=512)
    if mgr is not None:
        mgr.register_ws_client(q)

        try:
            initial = mgr.get_all_states()
            await websocket.send_json({"type": "full_state", "states": initial})
        except Exception:
            logger.debug("Failed to send initial state on node WS")

    try:
        loop = asyncio.get_event_loop()
        last_msg_time = time.monotonic()
        while True:
            try:
                msg = await loop.run_in_executor(None, lambda: q.get(timeout=0.5))
                await websocket.send_text(json.dumps(msg))
                last_msg_time = time.monotonic()
                while not q.empty():
                    try:
                        msg = q.get_nowait()
                        await websocket.send_text(json.dumps(msg))
                    except stdlib_queue.Empty:
                        break
            except stdlib_queue.Empty:
                if time.monotonic() - last_msg_time > 30.0:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                    last_msg_time = time.monotonic()
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.debug("Node WebSocket closed unexpectedly")
    finally:
        if mgr is not None:
            mgr.unregister_ws_client(q)
