"""Development-only cloud relay for testing the cloud flow locally.

This module implements a lightweight WebSocket handler that speaks the same
protocol as ``scope.cloud.fal_app.ScopeApp.websocket_handler``, allowing a
second local Scope instance to act as the "cloud" side of the relay.

Enable by setting ``SCOPE_CLOUD_WS=1`` — the ``/ws`` route is then registered
in ``app.py``. See CLAUDE.md "Local Cloud Testing" for usage instructions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid

import httpx
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

logger = logging.getLogger(__name__)


def _local_url() -> str:
    """Return the base URL for the local Scope HTTP server.

    Read at request time so that the ``SCOPE_PORT`` env var set by
    ``run_server()`` is picked up correctly.
    """
    port = os.environ.get("SCOPE_PORT", "8000")
    return f"http://127.0.0.1:{port}"


async def cloud_ws_handler(ws: WebSocket) -> None:
    """WebSocket endpoint that mimics the fal.ai cloud protocol.

    Accepts a connection, sends a ``ready`` message, then proxies signaling
    and API messages to the local Scope HTTP server using ``httpx``.
    """
    await ws.accept()
    scope_url = _local_url()

    connection_id = str(uuid.uuid4())[:8]
    session_id: str | None = None

    logger.info("[%s] Cloud WS: connection accepted", connection_id)

    # Signal readiness — the client (CloudConnectionManager) expects this
    await ws.send_json({"type": "ready", "connection_id": connection_id})

    async def safe_send_json(payload: dict) -> None:
        try:
            if (
                ws.client_state != WebSocketState.CONNECTED
                or ws.application_state != WebSocketState.CONNECTED
            ):
                return
            await ws.send_json(payload)
        except (RuntimeError, WebSocketDisconnect):
            pass

    # --- message handlers ------------------------------------------------

    async def handle_get_ice_servers(payload: dict) -> dict:
        request_id = payload.get("request_id")
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{scope_url}/api/v1/webrtc/ice-servers",
                timeout=10.0,
            )
            return {
                "type": "ice_servers",
                "request_id": request_id,
                "data": response.json(),
                "status": response.status_code,
            }

    async def handle_offer(payload: dict) -> dict:
        nonlocal session_id
        request_id = payload.get("request_id")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{scope_url}/api/v1/webrtc/offer",
                    json={
                        "sdp": payload.get("sdp"),
                        "type": payload.get("sdp_type", "offer"),
                        "initialParameters": payload.get("initialParameters"),
                        "user_id": payload.get("user_id"),
                        "connection_id": connection_id,
                    },
                    timeout=30.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    session_id = data.get("sessionId")
                    return {
                        "type": "answer",
                        "request_id": request_id,
                        "sdp": data.get("sdp"),
                        "sdp_type": data.get("type"),
                        "sessionId": session_id,
                    }
                else:
                    return {
                        "type": "error",
                        "request_id": request_id,
                        "error": f"Offer failed: {response.status_code}",
                        "detail": response.text,
                    }
        except (httpx.TimeoutException, TimeoutError):
            return {
                "type": "error",
                "request_id": request_id,
                "error": "WebRTC offer timeout",
            }

    async def handle_icecandidate(payload: dict) -> dict:
        nonlocal session_id
        request_id = payload.get("request_id")
        candidate = payload.get("candidate")
        target_session = payload.get("sessionId") or session_id

        if not target_session:
            return {
                "type": "error",
                "request_id": request_id,
                "error": "No session ID available for ICE candidate",
            }

        if candidate is None:
            return {
                "type": "icecandidate_ack",
                "request_id": request_id,
                "status": "end_of_candidates",
            }

        async with httpx.AsyncClient() as client:
            response = await client.patch(
                f"{scope_url}/api/v1/webrtc/offer/{target_session}",
                json={
                    "candidates": [
                        {
                            "candidate": candidate.get("candidate"),
                            "sdpMid": candidate.get("sdpMid"),
                            "sdpMLineIndex": candidate.get("sdpMLineIndex"),
                        }
                    ]
                },
                timeout=10.0,
            )
            if response.status_code == 204:
                return {
                    "type": "icecandidate_ack",
                    "request_id": request_id,
                    "status": "ok",
                }
            else:
                return {
                    "type": "error",
                    "request_id": request_id,
                    "error": f"ICE candidate failed: {response.status_code}",
                    "detail": response.text,
                }

    async def handle_api_request(payload: dict) -> dict:
        method = payload.get("method", "GET").upper()
        path = payload.get("path", "")
        body = payload.get("body")
        request_id = payload.get("request_id")

        async with httpx.AsyncClient() as client:
            try:
                if method == "GET":
                    response = await client.get(f"{scope_url}{path}", timeout=30.0)
                elif method == "POST":
                    response = await client.post(
                        f"{scope_url}{path}", json=body, timeout=30.0
                    )
                elif method == "PATCH":
                    response = await client.patch(
                        f"{scope_url}{path}", json=body, timeout=30.0
                    )
                elif method == "DELETE":
                    response = await client.delete(f"{scope_url}{path}", timeout=30.0)
                else:
                    return {
                        "type": "api_response",
                        "request_id": request_id,
                        "status": 400,
                        "error": f"Unsupported method: {method}",
                    }

                try:
                    data = response.json()
                except Exception:
                    data = response.text

                return {
                    "type": "api_response",
                    "request_id": request_id,
                    "status": response.status_code,
                    "data": data,
                }
            except httpx.TimeoutException:
                return {
                    "type": "api_response",
                    "request_id": request_id,
                    "status": 504,
                    "error": "Request timeout",
                }
            except Exception as e:
                return {
                    "type": "api_response",
                    "request_id": request_id,
                    "status": 500,
                    "error": str(e),
                }

    async def handle_message(payload: dict) -> dict | None:
        msg_type = payload.get("type")
        request_id = payload.get("request_id")

        if msg_type == "ping":
            return {"type": "pong", "request_id": request_id}
        elif msg_type == "set_user_id":
            user_id = payload.get("user_id")
            logger.info("[%s] Cloud WS: user_id set to %s", connection_id, user_id)
            return {"type": "user_id_set", "user_id": user_id}
        elif msg_type == "get_ice_servers":
            return await handle_get_ice_servers(payload)
        elif msg_type == "offer":
            return await handle_offer(payload)
        elif msg_type == "icecandidate":
            return await handle_icecandidate(payload)
        elif msg_type == "api":
            return await handle_api_request(payload)
        else:
            return {
                "type": "error",
                "request_id": request_id,
                "error": f"Unknown message type: {msg_type}",
            }

    # --- main message loop -----------------------------------------------

    try:
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=60.0)
            except TimeoutError:
                continue
            except RuntimeError:
                break

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as e:
                await safe_send_json({"type": "error", "error": f"Invalid JSON: {e}"})
                continue

            response = await handle_message(payload)
            if response:
                await safe_send_json(response)

    except WebSocketDisconnect:
        logger.info("[%s] Cloud WS: disconnected", connection_id)
    except Exception as e:
        logger.error(
            "[%s] Cloud WS: error (%s): %s", connection_id, type(e).__name__, e
        )
    finally:
        # Tear down the WebRTC session so video streaming stops
        if session_id:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.delete(
                        f"{scope_url}/api/v1/webrtc/offer/{session_id}",
                        timeout=10.0,
                    )
                    if resp.status_code == 204:
                        logger.info(
                            "[%s] Cloud WS: closed WebRTC session %s",
                            connection_id,
                            session_id,
                        )
                    else:
                        logger.warning(
                            "[%s] Cloud WS: failed to close WebRTC session %s: %s",
                            connection_id,
                            session_id,
                            resp.status_code,
                        )
            except Exception as e:
                logger.warning(
                    "[%s] Cloud WS: failed to close WebRTC session %s: %s",
                    connection_id,
                    session_id,
                    e,
                )
