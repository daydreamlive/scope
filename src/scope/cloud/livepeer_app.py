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
import fractions
import json
import logging
import os
import queue
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
import httpx
import uvicorn
from av import AudioFrame, VideoFrame
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from livepeer_gateway.channel_reader import JSONLReader
from livepeer_gateway.channel_writer import JSONLWriter
from livepeer_gateway.media_output import MediaOutput
from livepeer_gateway.media_publish import (
    AudioOutputConfig,
    MediaPublish,
    MediaPublishConfig,
    VideoOutputConfig,
)
from pydantic import BaseModel

import scope.server.app as scope_app_module
from scope.server.app import app as scope_app
from scope.server.app import lifespan as scope_lifespan
from scope.server.frame_processor import FrameProcessor
from scope.server.kafka_publisher import publish_event
from scope.server.media_packets import ensure_video_packet

logger = logging.getLogger(__name__)
scope_client: httpx.AsyncClient | None = None
_connection_active = False
_connection_count = 0

STREAM_TASK_SHUTDOWN_GRACE_S = 1.0
STREAM_TASK_CANCEL_TIMEOUT_S = 1.0
MEDIA_STATS_INTERVAL_S = 10.0
REMOTE_VIDEO_CLOCK_RATE = 90_000
REMOTE_VIDEO_TIME_BASE = fractions.Fraction(1, REMOTE_VIDEO_CLOCK_RATE)
ASSETS_DIR_PATH = os.getenv("DAYDREAM_SCOPE_ASSETS_DIR", "/tmp/.daydream-scope/assets")


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
    description="Receives job info over WebSocket and subscribes to control/media channels",
)


class ScopeJobInfo(BaseModel):
    """Shape of the orchestrator HTTP response forwarded by the client."""

    manifest_id: str | None = None
    control_url: str | None = None
    events_url: str | None = None
    params: dict[str, Any] | None = None


@dataclass
class LivepeerSession:
    """Per-connection runner session state."""

    ws: WebSocket | None = None
    input_subscribe_urls: list[str | None] = field(default_factory=list)
    output_publish_urls: list[str | None] = field(default_factory=list)
    input_source_node_ids: list[str | None] = field(default_factory=list)
    output_sink_node_ids: list[str | None] = field(default_factory=list)
    output_record_node_ids: list[str | None] = field(default_factory=list)
    active_channels: list[dict[str, Any]] = field(default_factory=list)
    ws_pending_responses: dict[str, asyncio.Future[dict[str, Any]]] = field(
        default_factory=dict
    )
    frame_processor: FrameProcessor | None = None
    media_input_tasks: list[asyncio.Task] = field(default_factory=list)
    media_output_tasks: list[asyncio.Task] = field(default_factory=list)
    media_stats_task: asyncio.Task | None = None
    media_stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    stream_stop_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    media_outputs: list[MediaOutput | None] = field(default_factory=list)
    media_publishes: list[MediaPublish | None] = field(default_factory=list)
    user_id: str | None = None
    connection_id: str | None = None
    manifest_id: str | None = None
    session_id: str | None = None
    connection_info: dict[str, Any] | None = None


def _build_connection_info() -> dict[str, Any]:
    """Mirror the fal-wrapper's connection_info shape from env vars.

    Keeps the runner's events (stream_*, pipeline_*) carrying the same
    fal-region / runner-id / log-labels the wrapper's websocket_connected
    uses, so downstream consumers can correlate them.
    """
    fal_log_labels_raw = os.getenv("FAL_LOG_LABELS", "unknown")
    try:
        fal_log_labels = json.loads(fal_log_labels_raw)
    except (json.JSONDecodeError, TypeError):
        fal_log_labels = fal_log_labels_raw
    return {
        "gpu_type": os.getenv("FAL_MACHINE_TYPE", "GPU-H100"),
        "fal_region": os.getenv("NOMAD_DC", "unknown"),
        "fal_runner_id": os.getenv("FAL_JOB_ID", os.getenv("FAL_RUNNER_ID", "unknown")),
        "fal_log_labels": fal_log_labels,
    }


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


def _parse_browser_graph_routes(
    params: dict[str, Any],
) -> tuple[list[str | None], list[str], list[str]]:
    """Return browser source IDs, sink IDs, and record IDs."""
    source_ids: list[str | None] = []
    sink_ids: list[str] = []
    record_ids: list[str] = []
    graph = params.get("graph")
    if isinstance(graph, dict):
        for node in graph.get("nodes", []):
            if not isinstance(node, dict):
                continue
            node_id = node.get("id")
            if not isinstance(node_id, str):
                continue
            node_type = node.get("type")
            if node_type == "source":
                source_ids.append(node_id)
            elif node_type == "sink":
                sink_mode = node.get("sink_mode")
                if sink_mode not in ("spout", "ndi", "syphon"):
                    sink_ids.append(node_id)
            elif node_type == "record":
                record_ids.append(node_id)
    return source_ids, sink_ids, record_ids


def _resolve_output_route_ids(
    output_idx: int,
    sink_node_ids: list[str | None],
    record_node_ids: list[str],
) -> tuple[str | None, str | None]:
    """Return sink/record node ids for a mixed output slot index."""
    sink_count = len(sink_node_ids)
    if output_idx < sink_count:
        return sink_node_ids[output_idx], None

    record_idx = output_idx - sink_count
    if 0 <= record_idx < len(record_node_ids):
        return None, record_node_ids[record_idx]
    return None, None


def _resolve_produces_audio(
    params: dict[str, Any], status_info: dict[str, Any]
) -> bool:
    """Resolve audio capability from explicit params first, then pipeline status."""
    if "produces_audio" in params:
        return bool(params.get("produces_audio", False))
    return bool(status_info.get("produces_audio", False))


def _resolve_produces_video(
    params: dict[str, Any], status_info: dict[str, Any]
) -> bool:
    """Resolve video capability from explicit params first, then pipeline status."""
    if "produces_video" in params:
        return bool(params.get("produces_video", True))
    return bool(status_info.get("produces_video", True))


async def _request_stream_channels(
    session: LivepeerSession,
    *,
    direction: str,
    mime_type: str = "video/MP2T",
) -> list[dict[str, Any]]:
    """Request media channels from orchestrator over websocket."""
    ws_request_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    future: asyncio.Future[dict[str, Any]] = loop.create_future()
    session.ws_pending_responses[ws_request_id] = future
    await session.ws.send_json(
        {
            "type": "create_channels",
            "request_id": ws_request_id,
            "mime_type": mime_type,
            "direction": direction,
        }
    )
    try:
        ws_response = await asyncio.wait_for(future, timeout=5.0)
    finally:
        pending = session.ws_pending_responses.pop(ws_request_id, None)
        if pending is not None and not pending.done():
            pending.cancel()
    channels = ws_response.get("channels")
    if not isinstance(channels, list):
        raise RuntimeError("Invalid new_channels payload: channels must be a list")
    normalized: list[dict[str, Any]] = []
    for channel in channels:
        if not isinstance(channel, dict):
            continue
        url = channel.get("url")
        ch_direction = channel.get("direction")
        mime_type = channel.get("mime_type")
        if not isinstance(url, str) or not isinstance(ch_direction, str):
            continue
        normalized.append(
            {
                "url": url,
                "direction": ch_direction,
                "mime_type": mime_type if isinstance(mime_type, str) else "",
            }
        )
    return normalized


async def _request_restart(session: LivepeerSession) -> None:
    """
    Request restart acknowledgment over websocket.

    Allows the orchestrator to do cleanup, eg stopping keepalives that
    might otherwise behave unexpectedly during a runner restart.
    """
    ws_request_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    future: asyncio.Future[dict[str, Any]] = loop.create_future()
    session.ws_pending_responses[ws_request_id] = future
    # This should stop keepalives on the server.
    await session.ws.send_json(
        {
            "type": "restarting",
            "request_id": ws_request_id,
        }
    )
    try:
        ws_response = await asyncio.wait_for(future, timeout=5.0)
    finally:
        pending = session.ws_pending_responses.pop(ws_request_id, None)
        if pending is not None and not pending.done():
            pending.cancel()

    if ws_response.get("type") != "response":
        raise RuntimeError(
            "Invalid restarting response payload: expected response type"
        )


async def _stop_stream(session: LivepeerSession) -> None:
    """Stop frame processor and media tasks."""
    async with session.stream_stop_lock:
        session.media_stop_event.set()

        media_input_tasks = list(session.media_input_tasks)
        media_output_tasks = list(session.media_output_tasks)
        media_stats_task = session.media_stats_task
        session.media_input_tasks = []
        session.media_output_tasks = []
        session.media_stats_task = None
        session.media_outputs = []
        session.media_publishes = []
        session.input_subscribe_urls = []
        session.output_publish_urls = []
        session.input_source_node_ids = []
        session.output_sink_node_ids = []
        session.output_record_node_ids = []

        for i, task in enumerate(media_input_tasks):
            await _shutdown_task(task, task_name=f"media_input[{i}]")
        for i, task in enumerate(media_output_tasks):
            await _shutdown_task(task, task_name=f"media_output[{i}]")
        await _shutdown_task(media_stats_task, task_name="media_stats")

        if session.frame_processor is not None:
            session.frame_processor.stop()
            session.frame_processor = None
            # Pair with the session_created emitted at stream start.
            if session.session_id is not None:
                publish_event(
                    event_type="session_closed",
                    session_id=session.session_id,
                    connection_id=session.manifest_id,
                    user_id=session.user_id,
                    connection_info=session.connection_info,
                )
                session.session_id = None

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
    session: LivepeerSession,
    *,
    subscribe_url: str,
    source_node_id: str | None,
    input_track_index: int,
) -> None:
    """Receive decoded trickle frames and push into FrameProcessor."""
    frame_processor = session.frame_processor
    stop_event = session.media_stop_event
    if frame_processor is None:
        logger.error("Media input loop started without complete session state")
        return

    media_output = MediaOutput(subscribe_url)
    if input_track_index >= len(session.media_outputs):
        session.media_outputs.extend(
            [None] * (input_track_index + 1 - len(session.media_outputs))
        )
    session.media_outputs[input_track_index] = media_output
    try:
        async for decoded in media_output.frames():
            if stop_event.is_set():
                break
            if getattr(decoded, "kind", None) != "video":
                continue
            frame = getattr(decoded, "frame", None)
            if frame is None:
                continue
            if source_node_id is None:
                frame_processor.put(frame)
            else:
                frame_processor.put_to_source(frame, source_node_id)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("Media input loop failed: %s", exc)
    finally:
        try:
            await media_output.close()
        except Exception as exc:
            logger.warning("Media output close failed: %s", exc)
        if (
            input_track_index < len(session.media_outputs)
            and session.media_outputs[input_track_index] is media_output
        ):
            session.media_outputs[input_track_index] = None


async def _media_output_loop(
    session: LivepeerSession,
    *,
    publish_url: str,
    output_track_index: int,
    sink_node_id: str | None = None,
    record_node_id: str | None = None,
    fps: float = 30.0,
) -> None:
    """Read processed frames from FrameProcessor and publish over trickle."""
    frame_processor = session.frame_processor
    stop_event = session.media_stop_event
    if frame_processor is None:
        logger.error("Media output loop started without complete session state")
        return

    # Queue size should be large enough to absorb bursts. Encoder will drop
    # frames if it's draining slower than realtime, so large queues are OK
    publisher_queue_size = 30
    publisher = MediaPublish(
        publish_url,
        config=MediaPublishConfig(
            tracks=[VideoOutputConfig(fps=fps, queue_size=publisher_queue_size)]
        ),
    )
    video_tracks = publisher.get_tracks("video")
    publisher_track = video_tracks[0] if video_tracks else None
    if output_track_index >= len(session.media_publishes):
        session.media_publishes.extend(
            [None] * (output_track_index + 1 - len(session.media_publishes))
        )
    session.media_publishes[output_track_index] = publisher
    next_pts = 0
    try:
        while not stop_event.is_set():
            # TODO make this blocking; we busy-wait a LOT
            frame_item = None
            if record_node_id is not None:
                frame_item = frame_processor.sink_manager.recording.get_packet(
                    record_node_id
                )
                if frame_item is None:
                    await asyncio.sleep(0.01)  # no frame yet, wait a bit
                    continue
            elif sink_node_id is not None:
                frame_item = frame_processor.get_packet_from_sink(sink_node_id)
            else:
                frame_item = frame_processor.get_packet()
            if frame_item is None:
                await asyncio.sleep(0.01)  # no frame yet, wait a bit
                continue
            frame_packet = ensure_video_packet(frame_item)

            target_queue_size: int | None = None
            # TODO: Unify sink/record queue-size lookup into a single output-track path.
            if sink_node_id is not None:
                target_queue_size = frame_processor.sink_manager.get_sink_queue_maxsize(
                    sink_node_id
                )
            elif record_node_id is not None:
                target_queue_size = (
                    frame_processor.sink_manager.get_record_queue_maxsize(
                        record_node_id
                    )
                )

            # TODO: Queue sizing policy currently exists in both graph_executor.py
            # and pipeline_processor.py; centralize this in one place later.
            if (
                publisher_track is not None
                and target_queue_size is not None
                and target_queue_size > publisher_queue_size
            ):
                try:
                    publisher_track.resize(target_queue_size)
                    logger.info(
                        "Resized Livepeer output track queue %d -> %d "
                        "(output_track_index=%d sink_node_id=%s record_node_id=%s)",
                        publisher_queue_size,
                        target_queue_size,
                        output_track_index,
                        sink_node_id,
                        record_node_id,
                    )
                    publisher_queue_size = target_queue_size
                except Exception as exc:
                    logger.warning(
                        "Failed to resize Livepeer output track queue to %d "
                        "(output_track_index=%d): %s",
                        target_queue_size,
                        output_track_index,
                        exc,
                    )

            if sink_node_id is not None:
                sink_fps = frame_processor.get_fps_for_sink(sink_node_id)
                frame_ptime = 1.0 / sink_fps if sink_fps > 0 else 1.0 / fps
            else:
                stream_fps = frame_processor.get_fps()
                frame_ptime = 1.0 / stream_fps if stream_fps > 0 else 1.0 / fps

            video_frame = VideoFrame.from_ndarray(
                frame_packet.tensor.numpy(), format="rgb24"
            )
            if frame_packet.timestamp.is_valid:
                video_frame.pts = frame_packet.timestamp.pts
                video_frame.time_base = frame_packet.timestamp.time_base
            else:
                video_frame.pts = next_pts
                video_frame.time_base = REMOTE_VIDEO_TIME_BASE
                next_pts += int(frame_ptime * REMOTE_VIDEO_CLOCK_RATE)
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
        if (
            output_track_index < len(session.media_publishes)
            and session.media_publishes[output_track_index] is publisher
        ):
            session.media_publishes[output_track_index] = None


async def _media_audio_output_loop(
    session: LivepeerSession,
    *,
    publish_url: str,
    publish_slot_index: int,
) -> None:
    """Read processed audio chunks from FrameProcessor and publish over trickle."""
    frame_processor = session.frame_processor
    stop_event = session.media_stop_event
    if frame_processor is None:
        logger.error("Media audio output loop started without complete session state")
        return

    # Audio uses a larger queue than video to absorb jitter from async resampling.
    publisher = MediaPublish(
        publish_url,
        config=MediaPublishConfig(tracks=[AudioOutputConfig()]),
    )
    if 0 <= publish_slot_index < len(session.media_publishes):
        session.media_publishes[publish_slot_index] = publisher

    import numpy as np

    next_audio_pts = 0
    pts_sample_rate: int | None = None
    last_audio_media_ts: float | None = None

    try:
        while not stop_event.is_set():
            audio_packet = frame_processor.get_audio_packet()
            if audio_packet is None:
                await asyncio.sleep(0.01)
                continue
            audio_tensor = audio_packet.audio
            sample_rate = audio_packet.sample_rate
            if sample_rate is None or sample_rate <= 0:
                continue

            audio_np = audio_tensor.numpy()
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(1, -1)
            if audio_np.shape[0] > 2:
                audio_np = audio_np[:2]
            audio_np = np.asarray(audio_np, dtype=np.float32)

            sample_rate_int = int(sample_rate)
            layout = "mono" if audio_np.shape[0] == 1 else "stereo"
            frame = AudioFrame.from_ndarray(audio_np, format="fltp", layout=layout)
            frame.sample_rate = sample_rate_int
            frame_samples = int(getattr(frame, "samples", 0) or 0)
            if frame_samples <= 0:
                continue

            should_use_preserved_ts = False
            if audio_packet.timestamp.is_valid:
                media_ts = audio_packet.timestamp.pts * float(
                    audio_packet.timestamp.time_base
                )
                frame_duration_s = frame_samples / sample_rate_int
                if last_audio_media_ts is None or media_ts >= last_audio_media_ts:
                    should_use_preserved_ts = True
                    frame.pts = int(audio_packet.timestamp.pts)
                    frame.time_base = audio_packet.timestamp.time_base
                    last_audio_media_ts = media_ts + frame_duration_s
                    # Keep synthetic fallback aligned with the preserved timeline.
                    pts_sample_rate = sample_rate_int
                    next_audio_pts = (
                        int(round(media_ts * sample_rate_int)) + frame_samples
                    )
                else:
                    logger.warning(
                        "Ignoring non-monotonic preserved audio timestamp "
                        "(pts=%s, time_base=%s, start=%.6f, previous_end=%.6f)",
                        audio_packet.timestamp.pts,
                        audio_packet.timestamp.time_base,
                        media_ts,
                        last_audio_media_ts,
                    )

            if should_use_preserved_ts:
                await publisher.write_frame(frame)
                continue

            # Fallback: stamp explicit monotonic PTS in sample-time so mux timing
            # does not fall back to wall-clock heuristics.
            if pts_sample_rate is None:
                pts_sample_rate = sample_rate_int
            elif sample_rate_int != pts_sample_rate:
                # `next_audio_pts` is the accumulated sample-count timeline in
                # `pts_sample_rate` units (the previous frame rate basis).
                # Convert it into `sample_rate_int` units (the new frame rate basis)
                # so PTS stays continuous when sample rate changes mid-stream.
                next_audio_pts = int(
                    round(next_audio_pts * sample_rate_int / pts_sample_rate)
                )
                pts_sample_rate = sample_rate_int
            frame.pts = next_audio_pts
            frame.time_base = fractions.Fraction(1, sample_rate_int)
            next_audio_pts += frame_samples
            last_audio_media_ts = (frame.pts * float(frame.time_base)) + (
                frame_samples / sample_rate_int
            )
            await publisher.write_frame(frame)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("Media audio output loop failed: %s", exc)
    finally:
        try:
            await publisher.close()
        except Exception as exc:
            logger.warning("Media audio publisher close failed: %s", exc)
        if (
            0 <= publish_slot_index < len(session.media_publishes)
            and session.media_publishes[publish_slot_index] is publisher
        ):
            session.media_publishes[publish_slot_index] = None


async def _media_stats_loop(session: LivepeerSession) -> None:
    """Periodically log MediaPublish / MediaOutput statistics."""
    try:
        while not session.media_stop_event.is_set():
            await asyncio.sleep(MEDIA_STATS_INTERVAL_S)
            if session.media_stop_event.is_set():
                break
            for publisher in session.media_publishes:
                if publisher is not None:
                    logger.info(publisher.get_stats())
            for output in session.media_outputs:
                if output is not None:
                    logger.info(output.get_stats())
    except asyncio.CancelledError:
        pass


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
    logger.debug(
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

    if method == "POST" and normalized_path == "/api/v1/restart":
        # Notify orchestrator of the restart so it can tear down some stuff
        try:
            await _request_restart(session)
        except TimeoutError:
            return {
                "type": "api_response",
                "request_id": request_id,
                "status": 504,
                "error": "Timed out waiting for websocket restarting response",
            }
        except Exception as exc:
            return {
                "type": "api_response",
                "request_id": request_id,
                "status": 502,
                "error": f"Failed restart websocket handshake: {exc}",
            }

    # Pass through validated user_id and the orchestrator-provided
    # manifest_id for pipeline load requests. Using manifest_id (not the
    # runner's internal connection_id) so pipeline_loaded events correlate
    # with the fal wrapper's websocket_connected.connection_id.
    if (
        method == "POST"
        and normalized_path == "/api/v1/pipeline/load"
        and isinstance(body, dict)
        and session.user_id
    ):
        body["user_id"] = session.user_id
        body["connection_id"] = session.manifest_id or session.connection_id
        if session.connection_info is not None:
            body["connection_info"] = session.connection_info

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
            logger.debug(
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

        logger.debug(
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
        logger.debug("Received API control message id=%s", request_id)
        return await _handle_api_request(payload, session)
    if msg_type == "start_stream":
        params = payload.get("params") or {}
        if not isinstance(params, dict):
            return {
                "type": "error",
                "request_id": request_id,
                "error": "start_stream params must be an object",
            }

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

        produces_video = _resolve_produces_video(params, status_info)
        produces_audio = _resolve_produces_audio(params, status_info)
        input_mode = params.get("input_mode")
        source_node_ids, sink_node_ids, record_node_ids = _parse_browser_graph_routes(
            params
        )

        output_sink_node_ids: list[str | None]
        if not produces_video:
            output_sink_node_ids = []
            record_node_ids = []
        elif sink_node_ids:
            output_sink_node_ids = [sink_node_ids[0], *sink_node_ids[1:]]
        elif produces_audio:
            # Audio-only pipelines should not synthesize a placeholder video output.
            output_sink_node_ids = []
        else:
            output_sink_node_ids = [None]
        output_record_node_ids = [None] * len(output_sink_node_ids) + list(
            record_node_ids
        )

        active_channels: list[dict[str, Any]] = []
        input_subscribe_urls: list[str | None] = [None] * len(source_node_ids)
        output_publish_urls: list[str | None] = [None] * (
            len(output_sink_node_ids) + len(record_node_ids)
        )
        audio_publish_url: str | None = None

        try:
            if input_mode != "text":
                for input_idx, source_node_id in enumerate(source_node_ids):
                    channels = await _request_stream_channels(session, direction="in")
                    inbound_url: str | None = None
                    for channel in channels:
                        ch = {
                            **channel,
                            "role": "input",
                            "input_track_index": input_idx,
                            "source_node_id": source_node_id,
                        }
                        active_channels.append(ch)
                        if channel["direction"] == "in":
                            inbound_url = channel["url"]
                    if inbound_url is None:
                        raise RuntimeError("response did not include input track URL")
                    input_subscribe_urls[input_idx] = inbound_url

            for output_idx in range(len(output_publish_urls)):
                channels = await _request_stream_channels(session, direction="out")
                outbound_url: str | None = None
                sink_node_id, record_node_id = _resolve_output_route_ids(
                    output_idx=output_idx,
                    sink_node_ids=output_sink_node_ids,
                    record_node_ids=record_node_ids,
                )
                for channel in channels:
                    ch = {
                        **channel,
                        "role": "output",
                        "output_track_index": output_idx,
                        "sink_node_id": sink_node_id,
                        "record_node_id": record_node_id,
                    }
                    active_channels.append(ch)
                    if channel["direction"] == "out":
                        outbound_url = channel["url"]
                if outbound_url is None:
                    raise RuntimeError("response did not include output track URL")
                output_publish_urls[output_idx] = outbound_url
            if produces_audio:
                channels = await _request_stream_channels(
                    session,
                    direction="out",
                    mime_type="audio/MP2T",
                )
                for channel in channels:
                    ch = {
                        **channel,
                        "role": "output_audio",
                        "output_media_kind": "audio",
                    }
                    active_channels.append(ch)
                    if channel["direction"] == "out":
                        audio_publish_url = channel["url"]
                if audio_publish_url is None:
                    raise RuntimeError(
                        "response did not include audio output track URL"
                    )
        except TimeoutError:
            return {
                "type": "error",
                "request_id": request_id,
                "error": "Timed out waiting for websocket response from orchestrator",
            }
        except RuntimeError as exc:
            return {"type": "error", "request_id": request_id, "error": str(exc)}

        session.frame_processor = FrameProcessor(
            pipeline_manager=pipeline_manager,
            initial_parameters={
                **params,
                "pipeline_ids": pipeline_ids,
                "produces_video": produces_video,
                "produces_audio": produces_audio,
            },
            session_id=session.session_id,
            user_id=session.user_id,
            connection_id=session.manifest_id,
            connection_info=session.connection_info,
        )
        session.frame_processor.start()
        # Emit session_created so livepeer-mode event streams match the
        # cloud-relay shape (webrtc.py:731 fires this in relay mode; we
        # mirror it here because livepeer doesn't go through the WebRTC
        # offer handler).
        publish_event(
            event_type="session_created",
            session_id=session.session_id,
            connection_id=session.manifest_id,
            pipeline_ids=pipeline_ids if pipeline_ids else None,
            user_id=session.user_id,
            metadata={"mode": "livepeer"},
            connection_info=session.connection_info,
        )
        session.media_stop_event.clear()
        session.active_channels = active_channels
        session.input_subscribe_urls = input_subscribe_urls
        session.output_publish_urls = output_publish_urls
        session.input_source_node_ids = source_node_ids
        session.output_sink_node_ids = output_sink_node_ids
        session.output_record_node_ids = output_record_node_ids
        session.media_input_tasks = []
        session.media_output_tasks = []
        session.media_outputs = [None] * len(input_subscribe_urls)
        session.media_publishes = [None] * len(output_publish_urls)
        if input_mode != "text":
            for input_idx, subscribe_url in enumerate(input_subscribe_urls):
                if subscribe_url is None:
                    continue
                source_node_id = source_node_ids[input_idx]
                session.media_input_tasks.append(
                    asyncio.create_task(
                        _media_input_loop(
                            session,
                            subscribe_url=subscribe_url,
                            source_node_id=source_node_id,
                            input_track_index=input_idx,
                        )
                    )
                )
        fps = float(params.get("fps", 30.0))
        for output_idx, publish_url in enumerate(output_publish_urls):
            if publish_url is None:
                continue
            sink_node_id, record_node_id = _resolve_output_route_ids(
                output_idx=output_idx,
                sink_node_ids=output_sink_node_ids,
                record_node_ids=record_node_ids,
            )
            session.media_output_tasks.append(
                asyncio.create_task(
                    _media_output_loop(
                        session,
                        publish_url=publish_url,
                        output_track_index=output_idx,
                        sink_node_id=sink_node_id,
                        record_node_id=record_node_id,
                        fps=fps,
                    )
                )
            )
        if audio_publish_url is not None:
            audio_publish_slot = len(session.media_publishes)
            session.media_publishes.append(None)
            session.media_output_tasks.append(
                asyncio.create_task(
                    _media_audio_output_loop(
                        session,
                        publish_url=audio_publish_url,
                        publish_slot_index=audio_publish_slot,
                    )
                )
            )
        session.media_stats_task = asyncio.create_task(_media_stats_loop(session))
        logger.info(
            "Started stream with pipeline_ids=%s inputs=%s outputs=%s audio=%s",
            pipeline_ids,
            len(input_subscribe_urls),
            len(output_publish_urls),
            produces_audio,
        )
        return {
            "type": "stream_started",
            "request_id": request_id,
            "channels": active_channels,
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
    logging_id = session.connection_id or f"logging_{uuid.uuid4()!s:.8}"

    async def _forward_logs_to_events(log_queue: queue.Queue[str]) -> None:
        log_batch_limit = 50
        poll_interval = 0.5
        try:
            while not stop_event.is_set():
                batch: list[str] = []
                while len(batch) < log_batch_limit:
                    try:
                        batch.append(log_queue.get_nowait())
                    except queue.Empty:
                        break

                if batch:
                    await events_writer.write({"type": "logs", "lines": batch})
                else:
                    await asyncio.sleep(poll_interval)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("Failed to forward logs to events channel: %s", exc)

    log_queue = log_broadcaster.subscribe(logging_id)
    logs_task = asyncio.create_task(_forward_logs_to_events(log_queue))

    try:
        await events_writer.write(
            {
                "type": "runner_ready",
                "runner_job_id": os.getenv("FAL_JOB_ID") or os.getenv("FAL_RUNNER_ID"),
            }
        )
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
        logs_task.cancel()
        try:
            await logs_task
        except asyncio.CancelledError:
            pass
        log_broadcaster.unsubscribe(logging_id)
        await _stop_stream(session)
        try:
            await events_writer.close()
        except Exception as exc:
            logger.warning("Events writer close failed: %s", exc)


async def _cleanup_plugins_via_scope_client() -> dict[str, Any]:
    """Uninstall all installed plugins via the embedded Scope API."""
    client = scope_client
    if client is None:
        raise RuntimeError("Runner is not initialized")

    response = await client.get("/api/v1/plugins", timeout=10.0)
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to list plugins for cleanup: HTTP {response.status_code}"
        )

    payload = response.json()
    plugins = payload.get("plugins", []) if isinstance(payload, dict) else []
    removed: list[str] = []
    skipped: list[str] = []
    failed: list[dict[str, Any]] = []

    for plugin in plugins:
        if not isinstance(plugin, dict):
            continue

        name = plugin.get("name")
        if not name:
            continue
        if plugin.get("bundled"):
            skipped.append(name)
            continue
        try:
            uninstall = await client.delete(f"/api/v1/plugins/{name}", timeout=60.0)
            if uninstall.status_code == 200:
                removed.append(name)
            else:
                failed.append(
                    {
                        "name": name,
                        "status": uninstall.status_code,
                        "error": uninstall.text[:200],
                    }
                )
        except Exception as exc:
            failed.append({"name": name, "error": str(exc)})

    return {
        "removed": removed,
        "skipped": skipped,
        "failed": failed,
        "total": len(plugins),
    }


def _cleanup_assets_dir() -> dict[str, Any]:
    """Delete all files and directories inside the configured assets directory."""
    assets_dir = Path(ASSETS_DIR_PATH).expanduser()
    deleted = 0
    errors: list[dict[str, str]] = []

    if not assets_dir.exists():
        return {"path": str(assets_dir), "deleted": deleted, "errors": errors}

    for item in assets_dir.iterdir():
        try:
            if item.is_file():
                item.unlink()
                deleted += 1
            elif item.is_dir():
                shutil.rmtree(item)
                deleted += 1
        except Exception as exc:
            errors.append({"path": str(item), "error": str(exc)})

    return {"path": str(assets_dir), "deleted": deleted, "errors": errors}


@app.post("/internal/cleanup-session")
async def cleanup_session() -> dict[str, Any]:
    """Cleanup plugins and assets after the outer fal websocket disconnects."""
    try:
        plugins = await _cleanup_plugins_via_scope_client()
        assets = _cleanup_assets_dir()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "ok": not plugins["failed"] and not assets["errors"],
        "plugins": plugins,
        "assets": assets,
    }


@app.get("/internal/status")
async def internal_status() -> dict[str, Any]:
    return {
        "listener_ready": scope_client is not None,
        "connection_active": _connection_active,
        "connection_count": _connection_count,
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Accept a WebSocket connection, read job info, then subscribe to the control channel."""
    global _connection_active
    global _connection_count

    await ws.accept()
    _connection_active = True
    _connection_count += 1
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
        job_info = ScopeJobInfo.model_validate_json(raw)
        logger.info("Received job info: manifest_id=%s", job_info.manifest_id)
        params = job_info.params or {}
        user_id = params.get("daydream_user_id")
        if not await validate_user_access(user_id):
            await ws.send_json(
                {
                    "type": "error",
                    "error": "Access denied",
                    "code": "ACCESS_DENIED",
                }
            )
            await ws.close(code=4003, reason="Access denied")
            return
        # Remove transport-only user marker if present so it never reaches pipelines.
        # TODO move this into the top level request
        params.pop("daydream_user_id", None)
        session.user_id = user_id
        # Persist fields needed downstream so Kafka events emitted by the
        # inner pipeline_manager / FrameProcessor correlate with the fal
        # wrapper's websocket_connected (which uses manifest_id as
        # connection_id).
        session.manifest_id = job_info.manifest_id
        session.session_id = str(uuid.uuid4())
        session.connection_info = _build_connection_info()

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

        # Complete the handshake with the orchestrator
        await ws.send_json({"type": "started"})

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
            elif msg_type == "ping":
                await ws.send_json(
                    {
                        "type": "pong",
                        "request_id": message.get("request_id"),
                        "timestamp": message.get("timestamp"),
                    }
                )
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
        _connection_active = False


def get_daydream_api_base() -> str:
    return os.getenv("DAYDREAM_API_BASE", "https://api.daydream.live")


def _is_dev_mode() -> bool:
    value = os.getenv("LIVEPEER_DEV_MODE")
    if value is None:
        return False
    return True


async def validate_user_access(user_id: str | None) -> bool:
    """Validate that a user has access to cloud mode."""
    import urllib.error
    import urllib.request

    if not user_id:
        if _is_dev_mode():
            logger.info("LIVEPEER_DEV_MODE enabled; skipping user access validation")
            return True
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


# ---------------------------------------------------------------------------
# Trickle log forwarding — sends in-process log records to remote clients
# ---------------------------------------------------------------------------

_CLOUD_LOG_SKIP_LOGGERS_DEFAULT = {
    "scope.server.kafka_publisher",
    "livepeer_gateway.channel_writer",
}
_cloud_log_skip_loggers: set[str] = set()
_trickle_log_handler: logging.Handler | None = None


class LogBroadcaster:
    """Thread-safe broadcaster that fans out log lines to subscribers."""

    def __init__(self, max_queue_size: int = 200):
        self._subscribers: dict[str, queue.Queue[str]] = {}
        self._lock = threading.Lock()
        self._max_queue_size = max_queue_size

    def publish(self, line: str) -> None:
        with self._lock:
            for subscriber_queue in self._subscribers.values():
                try:
                    subscriber_queue.put_nowait(line)
                except queue.Full:
                    # Slow subscribers drop lines to avoid backpressure.
                    pass

    def subscribe(self, connection_id: str) -> queue.Queue[str]:
        subscriber_queue: queue.Queue[str] = queue.Queue(maxsize=self._max_queue_size)
        with self._lock:
            self._subscribers[connection_id] = subscriber_queue
        return subscriber_queue

    def unsubscribe(self, connection_id: str) -> None:
        with self._lock:
            self._subscribers.pop(connection_id, None)


class TrickleLogHandler(logging.Handler):
    """Log handler that forwards selected records into LogBroadcaster."""

    def emit(self, record: logging.LogRecord) -> None:
        if not _should_forward_log_record(record):
            return
        try:
            line = self.format(record)
            if line:
                log_broadcaster.publish(line)
        except Exception:
            self.handleError(record)


log_broadcaster = LogBroadcaster()


def _init_cloud_log_skip_loggers() -> set[str]:
    skip = set(_CLOUD_LOG_SKIP_LOGGERS_DEFAULT)
    extra = os.environ.get("CLOUD_LOG_SKIP_LOGGERS", "")
    for name in extra.split(","):
        name = name.strip()
        if name:
            skip.add(name)
    return skip


def _should_forward_log_record(record: logging.LogRecord) -> bool:
    global _cloud_log_skip_loggers
    if not _cloud_log_skip_loggers:
        _cloud_log_skip_loggers = _init_cloud_log_skip_loggers()

    if record.levelno >= logging.WARNING:
        return True

    return record.name not in _cloud_log_skip_loggers


def _configure_trickle_log_handler(
    level: int, formatter: logging.Formatter
) -> logging.Handler:
    """Attach a singleton trickle log handler to the root logger."""
    global _trickle_log_handler
    if _trickle_log_handler is not None:
        _trickle_log_handler.setLevel(level)
        _trickle_log_handler.setFormatter(formatter)
        return _trickle_log_handler

    handler = TrickleLogHandler(level=level)
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    _trickle_log_handler = handler
    return handler


@click.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Host to bind to")
@click.option("--port", default=8001, show_default=True, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def main(host: str, port: int, reload: bool) -> None:
    """Run the Livepeer runner WebSocket server."""
    log_level = logging.DEBUG if os.getenv("LIVEPEER_DEBUG") else logging.INFO
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=log_level, format=log_format)
    _configure_trickle_log_handler(
        level=log_level,
        formatter=logging.Formatter(log_format),
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
