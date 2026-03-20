"""Livepeer runner WebSocket app.

Runs a single-process runner that:
- uses a WebSocket endpoint for signaling/lifecycle,
- consumes control/events trickle channels,
- dispatches API calls in-process to the Scope FastAPI app via ASGI transport,
- processes media directly using trickle publish/subscribe channels.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import click
import httpx
import uvicorn
from av import VideoFrame
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from livepeer_gateway.channel_reader import JSONLReader
from livepeer_gateway.channel_writer import JSONLWriter
from livepeer_gateway.media_output import MediaOutput
from livepeer_gateway.media_publish import MediaPublish, MediaPublishConfig
from pydantic import BaseModel

import scope.server.app as scope_app_module
from scope.server.app import app as scope_app
from scope.server.app import lifespan as scope_lifespan
from scope.server.frame_processor import FrameProcessor

logger = logging.getLogger(__name__)
scope_client: httpx.AsyncClient | None = None

STREAM_TASK_SHUTDOWN_GRACE_S = 1.0
STREAM_TASK_CANCEL_TIMEOUT_S = 1.0


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize embedded Scope app lifespan and ASGI client."""
    global scope_client
    async with scope_lifespan(scope_app):
        scope_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=scope_app),
            base_url="http://runner",
        )
        try:
            yield
        finally:
            await scope_client.aclose()
            scope_client = None


app = FastAPI(
    lifespan=lifespan,
    title="Livepeer Runner App",
    description="Receives LV2V job info over WebSocket and subscribes to control/media channels",
)


class Lv2vJobInfo(BaseModel):
    """Shape of the LV2V orchestrator HTTP response forwarded by the client."""

    manifest_id: str | None = None
    control_url: str | None = None
    events_url: str | None = None
    publish_url: str | None = None
    subscribe_url: str | None = None
    daydream_user_id: str | None = None


@dataclass
class LivepeerSession:
    """Per-connection runner session state."""

    ws: WebSocket | None = None
    publish_url: str | None = None
    subscribe_url: str | None = None
    active_channels: list[dict[str, str]] = field(default_factory=list)
    ws_pending_responses: dict[str, asyncio.Future[dict[str, Any]]] = field(
        default_factory=dict
    )
    frame_processor: FrameProcessor | None = None
    media_input_task: asyncio.Task | None = None
    media_output_task: asyncio.Task | None = None
    media_stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    stream_stop_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    user_id: str | None = None
    connection_id: str | None = None


async def _shutdown_task(
    task: asyncio.Task | None,
    *,
    task_name: str,
    grace_timeout: float = STREAM_TASK_SHUTDOWN_GRACE_S,
    cancel_timeout: float = STREAM_TASK_CANCEL_TIMEOUT_S,
) -> None:
    """Prefer graceful task exit, then fall back to cancellation."""
    if task is None:
        return

    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=grace_timeout)
        return
    except TimeoutError:
        logger.info(
            "Task %s did not stop within %.1fs; cancelling",
            task_name,
            grace_timeout,
        )
    except asyncio.CancelledError:
        if task.done():
            return
        raise
    except Exception as exc:
        logger.warning("Task %s exited during shutdown: %s", task_name, exc)
        return

    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=cancel_timeout)
    except TimeoutError:
        logger.warning(
            "Task %s did not finish within %.1fs after cancellation",
            task_name,
            cancel_timeout,
        )
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.warning("Task %s failed after cancellation: %s", task_name, exc)


async def _stop_stream(session: LivepeerSession) -> None:
    """Stop frame processor and media tasks."""
    async with session.stream_stop_lock:
        session.media_stop_event.set()

        media_input_task = session.media_input_task
        media_output_task = session.media_output_task
        session.media_input_task = None
        session.media_output_task = None

        await _shutdown_task(media_input_task, task_name="media_input")
        await _shutdown_task(media_output_task, task_name="media_output")

        if session.frame_processor is not None:
            session.frame_processor.stop()
            session.frame_processor = None

        if session.active_channels:
            channel_urls = [ch["url"] for ch in session.active_channels]
            try:
                await session.ws.send_json(
                    {"type": "close_channels", "channels": channel_urls}
                )
            except Exception as exc:
                logger.warning("Failed to send close_channels over websocket: %s", exc)
        session.active_channels = []
        session.media_stop_event = asyncio.Event()


async def _media_input_loop(
    publish_url: str,
    frame_processor: FrameProcessor,
    stop_event: asyncio.Event,
) -> None:
    """Receive decoded trickle frames and push into FrameProcessor."""
    media_output = MediaOutput(publish_url)
    try:
        async for decoded in media_output.frames():
            if stop_event.is_set():
                break
            if getattr(decoded, "kind", None) != "video":
                continue
            frame = getattr(decoded, "frame", None)
            if frame is None:
                continue
            frame_processor.put(frame)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("Media input loop failed: %s", exc)
    finally:
        try:
            await media_output.close()
        except Exception as exc:
            logger.warning("Media output close failed: %s", exc)


async def _media_output_loop(
    subscribe_url: str,
    frame_processor: FrameProcessor,
    stop_event: asyncio.Event,
    fps: float = 30.0,
) -> None:
    """Read processed frames from FrameProcessor and publish over trickle."""
    publisher = MediaPublish(subscribe_url, config=MediaPublishConfig(fps=fps))
    try:
        while not stop_event.is_set():
            frame_tensor = frame_processor.get()
            if frame_tensor is None:
                await asyncio.sleep(0.01)  # no frame yet, wait a bit
                continue
            video_frame = VideoFrame.from_ndarray(frame_tensor.numpy(), format="rgb24")
            await publisher.write_frame(video_frame)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("Media output loop failed: %s", exc)
    finally:
        try:
            await publisher.close()
        except Exception as exc:
            logger.warning("Media publisher close failed: %s", exc)


async def _handle_api_request(
    payload: dict[str, Any], session: LivepeerSession
) -> dict[str, Any]:
    """Proxy arbitrary API requests to embedded Scope FastAPI app."""
    method = str(payload.get("method", "GET")).upper()
    path = str(payload.get("path", ""))
    body = payload.get("body")
    request_id = payload.get("request_id")
    from urllib.parse import unquote, urlparse

    normalized_path = unquote(urlparse(path).path).rstrip("/")
    logger.info(
        "Processing API request id=%s method=%s path=%s", request_id, method, path
    )

    # Check plugin against backend allow list.
    if method == "POST" and normalized_path == "/api/v1/plugins":
        requested_package = body.get("package", "") if isinstance(body, dict) else ""
        allowed = await _is_plugin_allowed(requested_package)
        if allowed is None:
            return {
                "type": "api_response",
                "request_id": request_id,
                "status": 503,
                "error": "Unable to verify plugin allowlist — the Daydream API is currently unavailable. Please try again later.",
            }
        if not allowed:
            return {
                "type": "api_response",
                "request_id": request_id,
                "status": 403,
                "error": f"Plugin '{requested_package}' is not in the allowed list for cloud mode",
            }

    # Pass through validated user_id for pipeline load requests.
    if (
        method == "POST"
        and normalized_path == "/api/v1/pipeline/load"
        and isinstance(body, dict)
        and session.user_id
    ):
        body["user_id"] = session.user_id
        body["connection_id"] = session.connection_id

    client = scope_client
    if client is None:
        return {
            "type": "api_response",
            "request_id": request_id,
            "status": 503,
            "error": "Runner is not initialized",
        }
    try:
        is_binary_upload = body and isinstance(body, dict) and "_base64_content" in body
        is_cdn_upload = body and isinstance(body, dict) and "_cdn_url" in body

        if method == "GET":
            timeout = 120.0 if "/recordings/" in path else 30.0
            response = await client.get(path, timeout=timeout)
        elif method == "POST":
            if is_cdn_upload:
                cdn_url = body["_cdn_url"]
                content_type = body.get("_content_type", "application/octet-stream")
                cdn_result = await _download_content(
                    cdn_url,
                    request_id,
                )
                if cdn_result.error_response is not None:
                    return cdn_result.error_response
                response = await client.post(
                    path,
                    content=cdn_result.content,
                    headers={"Content-Type": content_type},
                    timeout=60.0,
                )
            elif is_binary_upload:
                binary_content = base64.b64decode(body["_base64_content"])
                content_type = body.get("_content_type", "application/octet-stream")
                response = await client.post(
                    path,
                    content=binary_content,
                    headers={"Content-Type": content_type},
                    timeout=60.0,
                )
            else:
                post_timeout = 300.0 if normalized_path == "/api/v1/loras" else 30.0
                response = await client.post(
                    path,
                    json=body,
                    timeout=post_timeout,
                )
        elif method == "PATCH":
            response = await client.patch(
                path,
                json=body,
                timeout=30.0,
            )
        elif method == "DELETE":
            response = await client.delete(path, timeout=30.0)
        else:
            return {
                "type": "api_response",
                "request_id": request_id,
                "status": 400,
                "error": f"Unsupported method: {method}",
            }

        content_type = response.headers.get("content-type", "")
        is_binary_response = any(
            media_type in content_type
            for media_type in [
                "video/",
                "audio/",
                "application/octet-stream",
                "image/",
            ]
        )

        if is_binary_response and response.status_code == 200:
            binary_content = response.content
            logger.info(
                "Completed API request id=%s status=%s binary=true bytes=%s",
                request_id,
                response.status_code,
                len(binary_content),
            )
            return {
                "type": "api_response",
                "request_id": request_id,
                "status": response.status_code,
                "_base64_content": base64.b64encode(binary_content).decode("utf-8"),
                "_content_type": content_type,
                "_content_length": len(binary_content),
            }

        try:
            data = response.json()
        except Exception:
            data = response.text

        logger.info(
            "Completed API request id=%s status=%s binary=false",
            request_id,
            response.status_code,
        )
        return {
            "type": "api_response",
            "request_id": request_id,
            "status": response.status_code,
            "data": data,
        }
    except httpx.TimeoutException:
        logger.warning(
            "API request timed out id=%s method=%s path=%s",
            request_id,
            method,
            path,
        )
        return {
            "type": "api_response",
            "request_id": request_id,
            "status": 504,
            "error": "Request timeout",
        }
    except Exception as exc:
        logger.exception(
            "API request failed id=%s method=%s path=%s", request_id, method, path
        )
        return {
            "type": "api_response",
            "request_id": request_id,
            "status": 500,
            "error": str(exc),
        }


async def _handle_control_message(
    payload: dict[str, Any],
    session: LivepeerSession,
) -> dict[str, Any] | None:
    """Handle one control message and optionally return an events response payload."""
    msg_type = payload.get("type")
    request_id = payload.get("request_id")

    if msg_type == "ping":
        return {
            "type": "pong",
            "request_id": request_id,
            "timestamp": payload.get("timestamp"),
        }
    if msg_type == "api":
        logger.info("Received API control message id=%s", request_id)
        return await _handle_api_request(payload, session)
    if msg_type == "start_stream":
        params = payload.get("params") or {}
        if not isinstance(params, dict):
            return {
                "type": "error",
                "request_id": request_id,
                "error": "start_stream params must be an object",
            }
        # Remove transport-only user marker if present so it never reaches pipelines.
        params.pop("daydream_user_id", None)

        if session.frame_processor is not None:
            logger.info("start_stream ignored: stream already running")
            return {
                "type": "stream_started",
                "request_id": request_id,
                "status": "already_running",
            }
        pipeline_manager = scope_app_module.pipeline_manager
        if pipeline_manager is None:
            return {
                "type": "error",
                "request_id": request_id,
                "error": "Pipeline manager is not initialized",
            }

        status_info = await pipeline_manager.get_status_info_async()
        pipeline_ids = params.get("pipeline_ids")
        if not pipeline_ids:
            pipeline_ids = status_info.get("pipeline_ids") or []
        if not pipeline_ids:
            return {
                "type": "error",
                "request_id": request_id,
                "error": "No pipeline loaded. Load a pipeline before start_stream.",
            }

        # Each create_channels handshake gets a unique request_id so we can match
        # the corresponding response from the websocket listener.
        ws_request_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        new_channels_future: asyncio.Future[dict[str, Any]] = loop.create_future()
        session.ws_pending_responses[ws_request_id] = new_channels_future

        # Ask the orchestrator to create stream channels. request_id is echoed back
        # on a generic response so we can resolve the correct pending future.
        await session.ws.send_json(
            {
                "type": "create_channels",
                "request_id": ws_request_id,
                "mime_type": "video/MP2T",
                "direction": "bidirectional",
            }
        )

        try:
            # Wait for the websocket listener to resolve this request_id. Keep the
            # timeout tight so control calls fail fast when the orchestrator stalls.
            ws_response = await asyncio.wait_for(
                new_channels_future,
                timeout=5.0,
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            return {
                "type": "error",
                "request_id": request_id,
                "error": "Timed out waiting for websocket response from orchestrator",
            }
        finally:
            # Always remove this request from the pending map. If it was never
            # fulfilled, cancel the future so nothing can resolve it later.
            pending = session.ws_pending_responses.pop(ws_request_id, None)
            if pending is not None and not pending.done():
                pending.cancel()

        # Validate and normalize channel metadata from the orchestrator response.
        # We need both directions to wire media input and output endpoints.
        channels = ws_response.get("channels")
        if not isinstance(channels, list):
            return {
                "type": "error",
                "request_id": request_id,
                "error": "Invalid new_channels payload: channels must be a list",
            }

        outbound_url: str | None = None
        inbound_url: str | None = None
        normalized_channels: list[dict[str, str]] = []
        for channel in channels:
            if not isinstance(channel, dict):
                continue
            url = channel.get("url")
            direction = channel.get("direction")
            mime_type = channel.get("mime_type")
            if not isinstance(url, str) or not isinstance(direction, str):
                continue
            normalized_channels.append(
                {
                    "url": url,
                    "direction": direction,
                    "mime_type": mime_type if isinstance(mime_type, str) else "",
                }
            )
            if direction == "out":
                outbound_url = url
            elif direction == "in":
                inbound_url = url

        # Bidirectional streaming requires both in and out channels.
        if not outbound_url or not inbound_url:
            return {
                "type": "error",
                "request_id": request_id,
                "error": "response did not include in and out URLs",
            }

        # Persist URLs so _stop_stream can send full URLs back in stop_stream and
        # so media loops know where to read/write frames for this stream session.
        session.active_channels = normalized_channels
        session.subscribe_url = inbound_url
        session.publish_url = outbound_url

        session.frame_processor = FrameProcessor(
            pipeline_manager=pipeline_manager,
            initial_parameters={**params, "pipeline_ids": pipeline_ids},
        )
        session.frame_processor.start()
        session.media_stop_event.clear()
        session.media_input_task = asyncio.create_task(
            _media_input_loop(
                session.subscribe_url, session.frame_processor, session.media_stop_event
            )
        )
        fps = float(params.get("fps", 30.0))
        session.media_output_task = asyncio.create_task(
            _media_output_loop(
                session.publish_url,
                session.frame_processor,
                session.media_stop_event,
                fps=fps,
            )
        )
        logger.info("Started stream with pipeline_ids=%s", pipeline_ids)
        return {
            "type": "stream_started",
            "request_id": request_id,
            "channels": normalized_channels,
        }

    if msg_type == "stop_stream":
        await _stop_stream(session)
        logger.info("Stopped stream")
        return {"type": "stream_stopped", "request_id": request_id}

    if msg_type == "parameters":
        params = payload.get("params") or {}
        if not isinstance(params, dict):
            return {
                "type": "error",
                "request_id": request_id,
                "error": "parameters params must be an object",
            }
        if session.frame_processor is None:
            return {
                "type": "error",
                "request_id": request_id,
                "error": "No active stream",
            }
        session.frame_processor.update_parameters(params)
        logger.info("Updated stream parameters")
        return {"type": "parameters_ack", "request_id": request_id, "status": "ok"}

    logger.warning("Unknown control message type: %s payload=%s", msg_type, payload)
    return {
        "type": "error",
        "request_id": request_id,
        "error": f"Unknown message type: {msg_type}",
    }


async def _subscribe_control(
    control_url: str,
    events_url: str,
    session: LivepeerSession,
    stop_event: asyncio.Event,
) -> None:
    """Subscribe to control channel and publish responses to events channel."""
    logger.info("Subscribing to control channel: %s", control_url)
    events_writer = JSONLWriter(events_url)

    try:
        async for message in JSONLReader(control_url)():
            if stop_event.is_set():
                break
            if not isinstance(message, dict):
                logger.warning("Ignoring non-dict control message: %r", message)
                continue

            response = await _handle_control_message(message, session)
            if response is not None:
                await events_writer.write(response)
    except asyncio.CancelledError:
        logger.info("Control channel subscription cancelled")
    except Exception as exc:
        logger.error("Control channel subscription error: %s", exc)
    finally:
        await _stop_stream(session)
        try:
            await events_writer.close()
        except Exception as exc:
            logger.warning("Events writer close failed: %s", exc)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Accept a WebSocket connection, read LV2V job info, then subscribe to the control channel."""
    await ws.accept()
    logger.info("WebSocket client connected")

    stop_event = asyncio.Event()
    control_task: asyncio.Task | None = None

    # Generate a unique connection ID for this WebSocket session
    connection_id = str(uuid.uuid4())[:8]  # Short ID for readability in logs

    session = LivepeerSession(ws=ws, connection_id=connection_id)

    # Send ready message with connection_id
    await ws.send_json({"type": "ready", "connection_id": connection_id})

    try:
        raw = await ws.receive_text()
        job_info = Lv2vJobInfo.model_validate_json(raw)
        logger.info("Received LV2V job info: manifest_id=%s", job_info.manifest_id)
        requested_user_id = job_info.daydream_user_id
        if not await validate_user_access(requested_user_id):
            await ws.send_json(
                {
                    "type": "error",
                    "error": "Access denied",
                    "code": "ACCESS_DENIED",
                }
            )
            await ws.close(code=4003, reason="Access denied")
            return
        session.user_id = requested_user_id

        if not job_info.control_url:
            await ws.send_text(
                json.dumps({"error": "control_url is required but was not provided"})
            )
            return
        if not job_info.events_url:
            await ws.send_text(
                json.dumps({"error": "events_url is required but was not provided"})
            )
            return

        control_task = asyncio.create_task(
            _subscribe_control(
                job_info.control_url,
                job_info.events_url,
                session,
                stop_event,
            )
        )

        # Keep the WebSocket open and route orchestrator responses used by control handlers.
        while True:
            raw_message = await ws.receive_text()
            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON on websocket: %r", raw_message[:200])
                continue

            if not isinstance(message, dict):
                logger.debug("Ignoring non-dict websocket payload: %r", message)
                continue

            msg_type = message.get("type")
            if msg_type == "response":
                ws_request_id = message.get("request_id")
                if not isinstance(ws_request_id, str) or not ws_request_id:
                    logger.warning("Received response without a valid request_id")
                    continue
                pending = session.ws_pending_responses.pop(ws_request_id, None)
                if pending is None:
                    logger.warning(
                        "Received unmatched websocket response request_id=%s",
                        ws_request_id,
                    )
                    continue
                if not pending.done():
                    pending.set_result(message)
            else:
                logger.debug("Ignoring websocket message type: %s", msg_type)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
    finally:
        for pending in session.ws_pending_responses.values():
            if not pending.done():
                pending.set_exception(RuntimeError("WebSocket closed"))
        session.ws_pending_responses.clear()
        stop_event.set()
        await _stop_stream(session)
        if control_task is not None:
            await _shutdown_task(control_task, task_name="control_channel")


def get_daydream_api_base() -> str:
    return os.getenv("DAYDREAM_API_BASE", "https://api.daydream.live")


async def validate_user_access(user_id: str | None) -> bool:
    """Validate that a user has access to cloud mode."""
    import urllib.error
    import urllib.request

    if not user_id:
        logger.warning("Access denied: no user ID provided")
        return False

    url = f"{get_daydream_api_base()}/v1/users/{user_id}"
    logger.info("Validating user access for %s via %s", user_id, url)

    def fetch_user():
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())

    try:
        await asyncio.get_event_loop().run_in_executor(None, fetch_user)
        return True
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            logger.warning("Access denied for user %s: user not found", user_id)
            return False
        logger.warning(
            "Access denied for user %s: failed to fetch user (%s)", user_id, exc.code
        )
        return False
    except Exception as exc:
        logger.warning("Access denied for user %s: validation error: %s", user_id, exc)
        return False


async def _is_plugin_allowed(package: str) -> bool | None:
    """Check whether a plugin package is allowed for cloud installation."""
    import re

    def normalize_plugin_url(url: str) -> str:
        normalized = url.lower().strip()
        normalized = re.sub(r"^git\+https?://", "", normalized)
        normalized = re.sub(r"^https?://", "", normalized)
        if normalized.endswith(".git"):
            normalized = normalized[:-4]
        return normalized.rstrip("/")

    normalized_package = normalize_plugin_url(package)
    base_url = f"{get_daydream_api_base()}/v1/plugins"
    limit = 100
    offset = 0

    try:
        async with httpx.AsyncClient() as client:
            while True:
                resp = await client.get(
                    base_url,
                    params={
                        "remoteOnly": "true",
                        "limit": limit,
                        "offset": offset,
                    },
                    timeout=10.0,
                )
                resp.raise_for_status()
                data = resp.json()
                for plugin in data.get("plugins", []):
                    plugin_url = plugin.get("repositoryUrl", "")
                    if plugin_url and normalized_package == normalize_plugin_url(
                        plugin_url
                    ):
                        return True
                if not data.get("hasMore", False):
                    break
                offset += limit
    except Exception as exc:
        logger.warning("Failed to fetch allowed plugins from %s: %s", base_url, exc)
        return None

    return False


@dataclass(frozen=True, slots=True)
class DownloadContentResult:
    content: bytes | None = None
    error_response: dict[str, Any] | None = None


async def _download_content(
    url: str,
    request_id: str | None,
) -> DownloadContentResult:
    logger.info("Downloading content from: %s", url)
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=120.0, follow_redirects=True)
    except Exception as exc:
        logger.warning("Download failed for request id=%s: %s", request_id, exc)
        return DownloadContentResult(
            error_response={
                "type": "api_response",
                "request_id": request_id,
                "status": 502,
                "error": f"Download error: {exc}",
            }
        )
    if resp.status_code != 200:
        logger.warning(
            "Download returned status=%s for request id=%s",
            resp.status_code,
            request_id,
        )
        return DownloadContentResult(
            error_response={
                "type": "api_response",
                "request_id": request_id,
                "status": 502,
                "error": f"Download failed: {resp.status_code}",
            }
        )
    if not resp.content:
        logger.warning("Download returned empty content for request id=%s", request_id)
        return DownloadContentResult(
            error_response={
                "type": "api_response",
                "request_id": request_id,
                "status": 502,
                "error": "Download failed: empty response body",
            }
        )
    return DownloadContentResult(content=resp.content)


@click.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Host to bind to")
@click.option("--port", default=8001, show_default=True, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def main(host: str, port: int, reload: bool) -> None:
    """Run the Livepeer runner WebSocket server."""
    log_level = logging.DEBUG if os.getenv("LIVEPEER_DEBUG") else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    if os.getenv("LIVEPEER_DEBUG"):
        logging.getLogger("livepeer_gateway").setLevel(logging.DEBUG)
        logging.getLogger(__name__).setLevel(logging.DEBUG)

    uvicorn.run(
        "scope.cloud.livepeer_app:app",
        host=host,
        port=port,
        reload=reload,
        log_config=None,
    )


if __name__ == "__main__":
    main()
