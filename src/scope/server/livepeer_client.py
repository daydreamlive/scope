"""LivepeerClient - Trickle HTTP client for Livepeer inference.

This client is transport-only. It manages the Livepeer job lifecycle and exposes
simple frame/parameter methods used by the relay manager.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import numpy as np
from av import VideoFrame
from livepeer_gateway.channel_reader import JSONLReader
from livepeer_gateway.channel_writer import JSONLWriter
from livepeer_gateway.errors import SkipPaymentCycle
from livepeer_gateway.media_output import MediaOutput
from livepeer_gateway.media_publish import (
    MediaPublish,
    MediaPublishConfig,
    VideoOutputConfig,
)
from livepeer_gateway.scope import StartJobRequest, start_scope

from scope.core.pacing import (
    MediaPacingDecision,
    MediaPacingState,
    compute_pacing_decision,
)

from .cloud_relay import AudioOutputHandler, FrameOutputHandler
from .logs_config import set_connection_id

logger = logging.getLogger(__name__)
LIVEPEER_ORCH_URL_ENV = "LIVEPEER_ORCH_URL"
LIVEPEER_WS_URL_ENV = "LIVEPEER_WS_URL"
LIVEPEER_SIGNER_ENV = "LIVEPEER_SIGNER"
LIVEPEER_SIGNER_FALSY_VALUES = {
    "",
    "0",
    "false",
    "off",
    "no",
    "none",
    "null",
    "disabled",
}
STOP_STREAM_SEND_TIMEOUT_S = 1.0
SHUTDOWN_TIMEOUT_S = 5.0
TASK_DRAIN_TIMEOUT_S = 0.25
RUNNER_RESTART_TIMEOUT_S = 30.0
PAYMENT_SEND_INTERVAL_S = 10.0


@dataclass(slots=True)
class _MediaHandles:
    subscriber_tasks: list[asyncio.Task]
    publishers: list[MediaPublish | None]
    outputs: list[MediaOutput | None]


@dataclass(slots=True)
class _BrowserGraphInfo:
    """Result of parsing `initial_parameters.graph` for Livepeer media layout."""

    source_node_to_track_index: dict[str, int]
    sink_node_ids: list[str]
    record_node_ids: list[str]
    """All record nodes (remote + sink-teed combined), in graph node order.
    Used to compute graph output indices that match cloud_track wiring expectations."""
    remote_record_node_ids: list[str]
    """Record nodes that need a dedicated runner output (pipeline-attached)."""
    sink_teed_records: list[tuple[str, int]]
    """Sink-attached records: (record_id, index in sink_node_ids). Mirrored locally."""

    @staticmethod
    def empty() -> _BrowserGraphInfo:
        return _BrowserGraphInfo(
            source_node_to_track_index={},
            sink_node_ids=[],
            record_node_ids=[],
            remote_record_node_ids=[],
            sink_teed_records=[],
        )


@dataclass(slots=True)
class _OutputMapping:
    """How runner tracks map to graph output slots (sinks + recorders)."""

    num_output_tracks: int
    """Number of remote outputs the runner will create."""
    num_local_handlers: int
    """Number of local output_handlers (includes sink-teed records)."""
    remote_to_local: list[int]
    """runner track index -> graph output index."""
    sink_tee_pairs: list[tuple[int, int]]
    """(sink_handler_index, record_handler_index) pairs for local mirroring."""


class LivepeerClient:
    """Livepeer transport client.

    This client opens a Livepeer job, publishes frames to the input channel,
    subscribes to output frames, and forwards output frames to callbacks.
    """

    def __init__(
        self,
        token: str,
        model_id: str,
        app_id: str | None = None,
        api_key: str | None = None,
        fps: float = 30.0,
    ):
        self._token = token
        self._model_id = model_id
        self._api_key = api_key
        self._fps = fps
        self._orchestrator_url = self._normalize_orchestrator_url(
            os.getenv(LIVEPEER_ORCH_URL_ENV)
        )
        env_ws_url = self._normalize_ws_url(os.getenv(LIVEPEER_WS_URL_ENV))
        ws_url = self._ws_url_from_app_id(app_id)
        # Keep explicit ws URL support; app id support is a convenient fallback.
        self._ws_url = env_ws_url or ws_url

        self._job = None
        self._media_publishers: list[MediaPublish | None] = []
        self._media_outputs: list[MediaOutput | None] = []
        self._control_writer: JSONLWriter | None = None
        self._media_subscriber_tasks: list[asyncio.Task] = []
        self._events_task: asyncio.Task | None = None
        self._ping_task: asyncio.Task | None = None
        self._payment_task: asyncio.Task | None = None
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._connected = False
        self._media_connected = False
        self._shutdown_started = False
        self._shutdown_lock = asyncio.Lock()
        self._runner_ready_event = asyncio.Event()
        self._connection_id: str | None = None

        # TODO: Relay interface is currently duck-typed by CloudTrack. Keep
        # these members aligned until a dedicated backend abstraction exists.
        self.input_tracks: list[Any] = []
        self.output_handlers: list[FrameOutputHandler] = [FrameOutputHandler()]
        self.audio_output_handler = AudioOutputHandler()
        self.source_node_to_track_index: dict[str, int] = {}

        self._stats = {
            "connected_at": None,
            "api_requests_sent": 0,
            "api_requests_successful": 0,
            "pacing": {
                "observations": 0,
                "valid_timestamp_frames": 0,
                "hard_resets": 0,
                "soft_reanchors": 0,
                "sleep_total_s": 0.0,
                "drift_samples": 0,
                "drift_abs_sum_s": 0.0,
                "drift_max_abs_s": 0.0,
                "stall_positive_samples": 0,
                "stall_positive_sum_s": 0.0,
                "stall_positive_max_s": 0.0,
            },
        }

    @property
    def is_connected(self) -> bool:
        return self._connected and self._job is not None

    @property
    def media_connected(self) -> bool:
        return (
            self.is_connected
            and self._media_connected
            and (bool(self._media_publishers) or bool(self._media_subscriber_tasks))
        )

    @property
    def connection_id(self) -> str | None:
        return self._connection_id

    async def connect(self, initial_parameters: dict | None = None) -> None:
        """Create a Livepeer job and start the events channel."""
        if self.is_connected:
            await self.disconnect()

        set_connection_id(None)
        self._loop = asyncio.get_running_loop()
        self._shutdown_started = False
        params: dict[str, Any] = dict(initial_parameters or {})
        logger.info(
            "Livepeer inference target: %s",
            self._ws_url or "default",
        )
        if self._ws_url and "ws_url" not in params:
            params["ws_url"] = self._ws_url

        # Configure signer if needed
        signer_env = os.environ.get(LIVEPEER_SIGNER_ENV)
        if signer_env is None:
            # Unset -> preserve default signer behavior.
            signer_url = "signer.daydream.live" if self._api_key else None
        elif signer_env.strip().lower() in LIVEPEER_SIGNER_FALSY_VALUES:
            # Explicit falsy value -> disable signer URL override.
            signer_url = None
        else:
            signer_url = signer_env
        signer_headers = (
            {"Authorization": f"Bearer {self._api_key}"}
            if self._api_key and signer_url
            else None
        )

        # Construct job parameters
        request = StartJobRequest(
            model_id=self._model_id,
            params=params or None,
        )

        # start_scope is synchronous and may block on network I/O, so run it in
        # a worker thread to avoid blocking the event loop.
        self._job = await asyncio.to_thread(
            start_scope,
            # If unset, orchestrator is discovered via token signer/discovery fields.
            self._orchestrator_url,
            request,
            token=self._token,
            signer_url=signer_url,
            signer_headers=signer_headers,
            timeout=300.0,
            use_tofu=bool(os.environ.get("LIVEPEER_DEV_MODE")),
        )
        self._connection_id = getattr(self._job, "manifest_id", None)
        set_connection_id(self._connection_id)

        # start_scope runs in a worker thread without an event loop, so
        # deferred async initialisers need to be kicked off now.
        if self._job.control_url:
            self._control_writer = JSONLWriter(self._job.control_url)

        if self._job.signer_url:
            self._payment_task = asyncio.create_task(
                self._payment_loop(self._job, self._job.payment_session)
            )
        else:
            logger.debug("Livepeer signer not configured; payment loop disabled")

        self._connected = True
        self._media_connected = False
        self._stats["connected_at"] = time.time()
        self._stats["api_requests_sent"] = 0
        self._stats["api_requests_successful"] = 0
        self._events_task = asyncio.create_task(self._events_loop())
        self._ping_task = asyncio.create_task(self._ping_loop())

        logger.info(
            "Connected to Scope on Livepeer (%s)",
            self._connection_id or "unknown",
        )

    @staticmethod
    def _normalize_orchestrator_url(value: str | None) -> str | None:
        if value is None:
            return None
        try:
            trimmed = value.strip()
            if not trimmed:
                raise ValueError
            # Intentionally very basic validation: parseability + non-empty only.
            # Accepts values like a plain hostname without scheme or port.
            _ = urlparse(trimmed)
            return trimmed
        except Exception:
            raise ValueError(
                "Invalid orchestrator URL. Expected host[:port] or http(s)://host[:port]."
            ) from None

    @staticmethod
    def _normalize_ws_url(value: str | None) -> str | None:
        if value is None:
            return None
        try:
            trimmed = value.strip()
            if not trimmed:
                raise ValueError
            parsed = urlparse(trimmed)
            # Accessing .port forces urllib to validate malformed/non-numeric ports.
            _ = parsed.port
        except Exception:
            raise ValueError(
                "Invalid LIVEPEER_WS_URL. Expected a valid ws:// or wss:// URL."
            ) from None
        if parsed.scheme not in {"ws", "wss"}:
            raise ValueError(
                "Invalid LIVEPEER_WS_URL. Expected a valid ws:// or wss:// URL."
            )
        if not parsed.hostname:
            raise ValueError(
                "Invalid LIVEPEER_WS_URL. Expected a valid ws:// or wss:// URL."
            )
        return trimmed

    @staticmethod
    def _ws_url_from_app_id(value: str | None) -> str | None:
        if not value:
            return None
        try:
            trimmed = value.strip()
            if not trimmed:
                raise ValueError
            app_id = trimmed.strip("/")
            if not app_id.endswith("/ws"):
                raise ValueError
            ws_url = f"wss://fal.run/{app_id}"
            parsed = urlparse(ws_url)
            # Accessing .port forces urllib to validate malformed/non-numeric ports.
            _ = parsed.port
        except Exception:
            raise ValueError(
                "Invalid cloud app id. Expected a non-empty app id ending in "
                "`/ws` (for example `daydream/scope-livepeer--prod/ws`)."
            ) from None
        if parsed.scheme not in {"ws", "wss"}:
            raise ValueError("Invalid ws_url. Expected a valid ws:// or wss:// URL.")
        if not parsed.hostname:
            raise ValueError("Invalid ws_url. Expected a valid ws:// or wss:// URL.")
        return ws_url

    @staticmethod
    def _parse_browser_graph(
        initial_parameters: dict[str, Any] | None,
    ) -> _BrowserGraphInfo:
        """Return source map, sinks, records, and sink-attached vs remote record split."""
        if not initial_parameters:
            return _BrowserGraphInfo.empty()
        graph_data = initial_parameters.get("graph")
        if not isinstance(graph_data, dict):
            return _BrowserGraphInfo.empty()

        source_node_to_track_index: dict[str, int] = {}
        sink_node_ids: list[str] = []
        record_node_ids: list[str] = []
        remote_record_node_ids: list[str] = []
        sink_teed_records: list[tuple[str, int]] = []

        node_by_id: dict[str, dict[str, Any]] = {}
        src_count = 0
        for node in graph_data.get("nodes", []):
            if not isinstance(node, dict):
                continue
            node_id = node.get("id")
            if not isinstance(node_id, str):
                continue
            node_by_id[node_id] = node
            node_type = node.get("type")
            if node_type == "source":
                source_node_to_track_index[node_id] = src_count
                src_count += 1
            elif node_type == "sink":
                sink_mode = node.get("sink_mode")
                if sink_mode not in ("spout", "ndi", "syphon"):
                    sink_node_ids.append(node_id)
            elif node_type == "record":
                record_node_ids.append(node_id)

        # Classify each record node by its input type: records fed from a
        # sink node are "teed" locally (no extra runner output needed), while
        # records fed from a pipeline node need their own remote output.
        # Teed records are dropped from the graph before sending to the runner.
        for rec_id in record_node_ids:
            inbound_from: str | None = None
            for edge in graph_data.get("edges", []):
                if not isinstance(edge, dict):
                    continue
                if edge.get("to_node") != rec_id or edge.get("kind") != "stream":
                    continue
                from_id = edge.get("from")
                if isinstance(from_id, str):
                    inbound_from = from_id
                    break

            if inbound_from is None:
                remote_record_node_ids.append(rec_id)
                continue

            src_node = node_by_id.get(inbound_from)
            if src_node is not None and src_node.get("type") == "sink":
                if inbound_from in sink_node_ids:
                    sink_teed_records.append(
                        (rec_id, sink_node_ids.index(inbound_from))
                    )
                else:
                    # Hardware sink excluded from WebRTC sink list — needs remote output.
                    remote_record_node_ids.append(rec_id)
            else:
                remote_record_node_ids.append(rec_id)

        return _BrowserGraphInfo(
            source_node_to_track_index=source_node_to_track_index,
            sink_node_ids=sink_node_ids,
            record_node_ids=record_node_ids,
            remote_record_node_ids=remote_record_node_ids,
            sink_teed_records=sink_teed_records,
        )

    @staticmethod
    def _filter_runner_params(
        initial_parameters: dict[str, Any] | None,
        parsed: _BrowserGraphInfo,
    ) -> dict[str, Any]:
        """Deep-copy params and drop sink-attached record nodes.

        The runner creates a dedicated output for every record node it sees, so
        removing sink-attached records avoids redundant remote outputs whose
        frames are identical to the parent sink.
        """
        if not initial_parameters:
            return {}
        if not parsed.sink_teed_records:
            return copy.deepcopy(initial_parameters)
        collapsed = {rid for rid, _ in parsed.sink_teed_records}
        params = copy.deepcopy(initial_parameters)
        graph = params.get("graph")
        if not isinstance(graph, dict):
            return params
        nodes = graph.get("nodes")
        if isinstance(nodes, list):
            graph["nodes"] = [
                n for n in nodes if isinstance(n, dict) and n.get("id") not in collapsed
            ]
        edges = graph.get("edges")
        if isinstance(edges, list):
            graph["edges"] = [
                e
                for e in edges
                if isinstance(e, dict)
                and e.get("to_node") not in collapsed
                and e.get("from") not in collapsed
            ]
        return params

    @staticmethod
    def _build_output_mapping(parsed: _BrowserGraphInfo) -> _OutputMapping:
        """Compute how runner tracks map to graph output slots."""
        num_sink_slots = max(len(parsed.sink_node_ids), 1)
        num_output_tracks = num_sink_slots + len(parsed.remote_record_node_ids)
        num_local_handlers = num_sink_slots + len(parsed.record_node_ids)

        remote_to_local: list[int] = list(range(num_sink_slots))
        for rid in parsed.remote_record_node_ids:
            rec_pos = parsed.record_node_ids.index(rid)
            remote_to_local.append(num_sink_slots + rec_pos)

        sink_tee_pairs: list[tuple[int, int]] = []
        for rec_id, sink_idx in parsed.sink_teed_records:
            rec_pos = parsed.record_node_ids.index(rec_id)
            sink_tee_pairs.append((sink_idx, num_sink_slots + rec_pos))

        return _OutputMapping(
            num_output_tracks=num_output_tracks,
            num_local_handlers=num_local_handlers,
            remote_to_local=remote_to_local,
            sink_tee_pairs=sink_tee_pairs,
        )

    def _make_input_track_handle(self, track_index: int):
        """Create a minimal sync put_frame handle expected by CloudSourceInputHandler."""
        handle = type("LivepeerInputTrackHandle", (), {})()
        handle.put_frame = lambda frame, idx=track_index: self.send_frame_to_track(
            frame, idx
        )
        return handle

    @staticmethod
    def _build_video_publish_config(fps: float) -> MediaPublishConfig:
        return MediaPublishConfig(tracks=[VideoOutputConfig(fps=fps, queue_size=30)])

    async def start_media(self, initial_parameters: dict | None = None) -> None:
        """Start media I/O and notify runner about stream start parameters."""
        if not self.is_connected or self._job is None:
            raise RuntimeError("Livepeer job is not connected")
        if self._loop is None:
            raise RuntimeError("Livepeer event loop not initialized")

        if self.media_connected:
            logger.info("Media already started")
            return

        parsed = self._parse_browser_graph(initial_parameters)
        runner_params = self._filter_runner_params(initial_parameters, parsed)

        request_id = str(uuid.uuid4())
        future: asyncio.Future = self._loop.create_future()
        self._pending_requests[request_id] = future
        await self._send_control(
            {
                "type": "start_stream",
                "request_id": request_id,
                "params": runner_params or {},
            }
        )

        try:
            response = await asyncio.wait_for(future, timeout=10.0)
        except TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise RuntimeError("Livepeer start_stream timeout after 10s") from None

        if response.get("type") == "error":
            raise RuntimeError(
                response.get("message")
                or response.get("error")
                or "start_stream failed"
            )

        channels = response.get("channels")
        if not isinstance(channels, list):
            raise RuntimeError("stream_started response missing channels list")

        self.source_node_to_track_index = parsed.source_node_to_track_index
        num_input_tracks = max(len(parsed.source_node_to_track_index), 1)
        # Sink-attached records are not sent to the runner but cloud_track
        # still expects all original graph outputs (sinks and recorders),
        # so map runner track indices to account for the gaps.
        mapping = self._build_output_mapping(parsed)

        input_urls_by_track: dict[int, str] = {}
        output_urls_by_track: dict[int, str] = {}
        audio_output_url: str | None = None
        for channel in channels:
            if not isinstance(channel, dict):
                continue
            url = channel.get("url")
            direction = channel.get("direction")
            role = channel.get("role")
            media_kind = channel.get("output_media_kind")
            if not isinstance(url, str) or not isinstance(direction, str):
                continue
            if direction == "in":
                if role not in (None, "input"):
                    continue
                idx_val = channel.get("input_track_index")
                track_index = (
                    idx_val if isinstance(idx_val, int) and idx_val >= 0 else 0
                )
                input_urls_by_track[track_index] = url
            elif direction == "out":
                if role == "output_audio" or media_kind == "audio":
                    audio_output_url = url
                    continue
                if role not in (None, "output"):
                    continue
                idx_val = channel.get("output_track_index")
                track_index = (
                    idx_val if isinstance(idx_val, int) and idx_val >= 0 else 0
                )
                output_urls_by_track[track_index] = url

        if (
            not input_urls_by_track
            and not output_urls_by_track
            and audio_output_url is None
        ):
            raise RuntimeError("stream_started response missing usable channels")

        self._media_publishers = [None] * num_input_tracks
        self._media_outputs = [None] * mapping.num_output_tracks
        self._media_subscriber_tasks = []
        self.input_tracks = [
            self._make_input_track_handle(i) for i in range(num_input_tracks)
        ]
        self.output_handlers = [
            FrameOutputHandler() for _ in range(mapping.num_local_handlers)
        ]
        if not self.output_handlers:
            self.output_handlers = [FrameOutputHandler()]

        # Wire sink-teed records: when the sink receives a frame,
        # mirror it into the recorder so recording works without the
        # runner sending a separate track for the recorder.
        for sink_handler_idx, rec_handler_idx in mapping.sink_tee_pairs:
            if sink_handler_idx < len(self.output_handlers) and rec_handler_idx < len(
                self.output_handlers
            ):
                self.output_handlers[sink_handler_idx].add_callback(
                    self.output_handlers[rec_handler_idx].handle_frame
                )

        publish_config = self._build_video_publish_config(self._fps)
        for input_idx in range(num_input_tracks):
            input_url = input_urls_by_track.get(input_idx)
            if input_url is None:
                continue
            publisher = MediaPublish(input_url, config=publish_config)
            self._media_publishers[input_idx] = publisher

        for output_idx in range(mapping.num_output_tracks):
            output_url = output_urls_by_track.get(output_idx)
            if output_url is None:
                continue
            media_output = MediaOutput(output_url, start_seq=-1)
            self._media_outputs[output_idx] = media_output
            local_handler_index = mapping.remote_to_local[output_idx]
            subscriber = asyncio.create_task(
                self._receive_loop(
                    media_output,
                    output_track_index=output_idx,
                    local_handler_index=local_handler_index,
                    media_kind="video",
                )
            )
            self._media_subscriber_tasks.append(subscriber)
        if audio_output_url is not None:
            media_output = MediaOutput(audio_output_url, start_seq=-1)
            audio_output_index = len(self._media_outputs)
            self._media_outputs.append(media_output)
            subscriber = asyncio.create_task(
                self._receive_loop(
                    media_output,
                    output_track_index=audio_output_index,
                    local_handler_index=0,
                    media_kind="audio",
                )
            )
            self._media_subscriber_tasks.append(subscriber)

        self._media_connected = any(self._media_publishers) or any(self._media_outputs)
        logger.info(
            "Media channels started (inputs=%s outputs=%s audio=%s)",
            sum(1 for p in self._media_publishers if p is not None),
            sum(1 for o in self._media_outputs if o is not None),
            audio_output_url is not None,
        )

    async def stop_media(self, current_task: asyncio.Task | None = None) -> None:
        """Stop media I/O while keeping signaling channels alive."""
        if current_task is None:
            current_task = asyncio.current_task()

        async with self._shutdown_lock:
            if self._shutdown_started:
                return
            if not self.is_connected and not self.media_connected:
                return
            if self._media_is_stopped():
                return

            # Snapshot and clear state before any awaits so concurrent callers
            # see the media as stopped immediately.
            media_handles = self._take_media_handles()
            control_writer = self._control_writer

        if control_writer is not None:
            try:
                await asyncio.wait_for(
                    self._send_control_message(control_writer, {"type": "stop_stream"}),
                    timeout=STOP_STREAM_SEND_TIMEOUT_S,
                )
            except TimeoutError:
                logger.warning(
                    "Timed out sending stop_stream control message after %.1fs",
                    STOP_STREAM_SEND_TIMEOUT_S,
                )
            except Exception as e:
                logger.warning(f"Failed to send stop_stream control message: {e}")

        await self._teardown_media_handles(media_handles, current_task=current_task)

        logger.info("Media channels stopped")

    async def _receive_loop(
        self,
        output: MediaOutput,
        *,
        output_track_index: int,
        local_handler_index: int,
        media_kind: str,
    ) -> None:
        """Consume output frames from Livepeer and notify callbacks.

        ``output_track_index`` indexes ``_media_outputs`` (media tracks).
        ``local_handler_index`` maps video into ``output_handlers`` (may differ
        from ``output_track_index`` when sink-attached records are collapsed).
        """
        pacing = MediaPacingState()
        unexpected_reason: str | None = None
        try:
            async for decoded in output.frames():
                if not self._connected or not self._media_connected:
                    break
                frame = getattr(decoded, "frame", None)
                if frame is None:
                    continue

                decoded_kind = getattr(decoded, "kind", None)
                if decoded_kind == "audio":
                    time_base = getattr(frame, "time_base", None)
                    audio_ts = (
                        frame.pts * float(time_base)
                        if time_base is not None and frame.pts is not None
                        else None
                    )
                    decision = compute_pacing_decision(
                        pacing,
                        media_ts=audio_ts,
                        now_monotonic=time.monotonic(),
                    )
                    self._record_pacing_observation(decision)
                    if decision.sleep_s > 0:
                        await asyncio.sleep(decision.sleep_s)
                    pacing.prev_wall_monotonic = time.monotonic()
                    self.audio_output_handler.handle_frame(frame)
                    continue
                if decoded_kind != "video":
                    continue

                time_base = getattr(frame, "time_base", None)
                video_ts = (
                    frame.pts * float(time_base)
                    if time_base is not None and frame.pts is not None
                    else None
                )
                decision = compute_pacing_decision(
                    pacing,
                    media_ts=video_ts,
                    now_monotonic=time.monotonic(),
                )
                # Keep observability in lock-step with every pacing decision.
                self._record_pacing_observation(decision)
                if decision.sleep_s > 0:
                    await asyncio.sleep(decision.sleep_s)
                # Track actual dispatch time (after optional sleep) so the next
                # wall_delta reflects real scheduling delay.
                pacing.prev_wall_monotonic = time.monotonic()
                if 0 <= local_handler_index < len(self.output_handlers):
                    self.output_handlers[local_handler_index].handle_frame(frame)
                else:
                    logger.debug(
                        "Dropping frame for unknown output handler index %s",
                        local_handler_index,
                    )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            unexpected_reason = (
                f"Livepeer {media_kind} output loop {output_track_index} failed: {e}"
            )
            logger.error(f"Output loop failed: {e}")
        finally:
            try:
                await output.close()
            except Exception as e:
                logger.warning(f"Error while closing media output: {e}")
            # Avoid clearing newer media outputs when stop/start races.
            if (
                0 <= output_track_index < len(self._media_outputs)
                and self._media_outputs[output_track_index] is output
            ):
                self._media_outputs[output_track_index] = None
            if self._media_connected and not self._shutdown_started:
                if unexpected_reason is None:
                    unexpected_reason = (
                        "Livepeer "
                        f"{media_kind} output loop {output_track_index} stopped unexpectedly"
                    )
                logger.warning("%s; stopping media only", unexpected_reason)
                await self.stop_media(current_task=asyncio.current_task())
            logger.info(
                "Output loop stopped (track=%s kind=%s)",
                output_track_index,
                media_kind,
            )

    async def _events_loop(self) -> None:
        """Consume control/events channel and resolve pending API requests."""
        if self._job is None or not getattr(self._job, "events_url", None):
            return

        unexpected_reason: str | None = None
        try:
            async for event in JSONLReader(self._job.events_url)():
                if not self._connected:
                    break
                if not isinstance(event, dict):
                    continue

                msg_type = event.get("type")
                request_id = event.get("request_id")

                if msg_type in {"api_response", "stream_started", "error"}:
                    future = self._pending_requests.pop(request_id, None)
                    if future is None:
                        logger.warning(
                            "Received unmatched %s event request_id=%s",
                            msg_type,
                            request_id,
                        )
                        continue
                    if not future.done():
                        future.set_result(event)
                    continue

                if msg_type == "pong":
                    timestamp = event.get("timestamp")
                    if isinstance(timestamp, (int, float)):
                        latency_ms = (time.time() - timestamp) * 1000.0
                        logger.info("Pong latency: %.1fms", latency_ms)
                    continue

                if msg_type == "runner_ready":
                    logger.info("Runner ready")
                    self._runner_ready_event.set()
                    continue

                if msg_type == "logs":
                    _handle_cloud_logs(event)
                    continue

                logger.debug(f"Event: {event}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            unexpected_reason = f"Livepeer events loop failed: {e}"
            logger.error(f"Events loop failed: {e}")
        finally:
            if self._connected and not self._shutdown_started:
                if unexpected_reason is None:
                    unexpected_reason = "Livepeer events loop stopped unexpectedly"
                await self._shutdown(
                    unexpected_reason=unexpected_reason,
                    current_task=asyncio.current_task(),
                )
            logger.info("Events loop stopped")

    async def _ping_loop(self) -> None:
        """Send periodic keepalive pings over the control channel."""
        try:
            while self._connected:
                await asyncio.sleep(10.0)
                if not self._connected:
                    break

                # TODO: Add a signature to the ping payload for tamper resistance.
                ping_message = {"type": "ping", "timestamp": time.time()}
                try:
                    await self._send_control(ping_message)
                except Exception as e:
                    logger.debug(f"Failed to send keepalive ping: {e}")
                if self._media_connected:
                    for publisher in self._media_publishers:
                        if publisher is not None:
                            logger.info(publisher.get_stats())
                    for media_output in self._media_outputs:
                        if media_output is not None:
                            logger.info(media_output.get_stats())
        except asyncio.CancelledError:
            pass

    async def _payment_loop(self, job: Any, payment_session: Any) -> None:
        """Send periodic payments while this job remains active."""
        logger.info(
            "Livepeer payment loop started (interval=%.1fs)",
            PAYMENT_SEND_INTERVAL_S,
        )
        # NB: This sends a payment immediately. It may have been a while
        # since the upfront payment was made due to cold starts, so get
        # ahead of the orchestrator's own balance check.
        try:
            while not self._shutdown_started and self._job is job:
                if not self._connected or self._job is not job:
                    break
                try:
                    await asyncio.to_thread(payment_session.send_payment)
                except SkipPaymentCycle as e:
                    logger.debug("Livepeer payment loop skipped cycle: %s", e)
                except Exception as e:
                    logger.warning("Livepeer periodic payment failed: %s", e)
                if (
                    self._shutdown_started
                    or not self._connected
                    or self._job is not job
                ):
                    break
                await asyncio.sleep(PAYMENT_SEND_INTERVAL_S)
        except asyncio.CancelledError:
            pass
        finally:
            logger.debug("Livepeer payment loop stopped")

    async def _send_control(self, message: dict[str, Any]) -> None:
        """Send a typed control message to the runner."""
        if self._control_writer is None:
            raise RuntimeError("Livepeer control channel is not available")
        await self._control_writer.write(message)

    async def api_request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Proxy an API request through Livepeer control/events channels."""
        if not self.is_connected:
            raise RuntimeError("Livepeer job is not connected")
        if self._loop is None:
            raise RuntimeError("Livepeer event loop not initialized")

        method_upper = method.upper()
        normalized_path = urlparse(path).path.rstrip("/") or "/"
        is_restart_request = (
            method_upper == "POST" and normalized_path == "/api/v1/restart"
        )

        request_id = str(uuid.uuid4())
        message: dict[str, Any] = {
            "type": "api",
            "request_id": request_id,
            "method": method_upper,
            "path": path,
        }
        if body is not None:
            message["body"] = body

        self._stats["api_requests_sent"] += 1
        future: asyncio.Future = self._loop.create_future()
        self._pending_requests[request_id] = future
        if is_restart_request:
            self._runner_ready_event.clear()
        await self._send_control(message)

        try:
            response = await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise RuntimeError(
                f"Livepeer API request timeout after {timeout}s: {method} {path}"
            ) from None

        if response.get("type") == "error":
            raise RuntimeError(
                response.get("message")
                or response.get("error")
                or "Livepeer API request failed"
            )

        self._stats["api_requests_successful"] += 1
        status = response.get("status", 999)
        logger.info(f"API response: {status} for {method} {path}")
        if is_restart_request and status < 400:
            logger.info("Waiting for runner_ready after restart")
            try:
                await asyncio.wait_for(
                    self._runner_ready_event.wait(),
                    timeout=RUNNER_RESTART_TIMEOUT_S,
                )
            except TimeoutError:
                raise RuntimeError(
                    "Timed out waiting for Livepeer runner to become ready after restart"
                ) from None
            logger.info("Received runner_ready after restart")
        return response

    def send_frame(self, frame: VideoFrame | np.ndarray) -> bool:
        """Send an input frame to Livepeer.

        Returns False if no active job or publishing fails.
        """
        return self.send_frame_to_track(frame, 0)

    def send_frame_to_track(
        self, frame: VideoFrame | np.ndarray, track_index: int
    ) -> bool:
        """Send a frame to the requested Livepeer input track."""
        if not self.media_connected:
            return False
        if track_index < 0 or track_index >= len(self._media_publishers):
            return False
        publisher = self._media_publishers[track_index]
        if publisher is None:
            return False
        if isinstance(frame, np.ndarray):
            frame = VideoFrame.from_ndarray(frame, format="rgb24")
        try:
            tracks = publisher.get_tracks("video")
            if not tracks:
                return False
            result = tracks[0].write_frame(frame)
            if inspect.isawaitable(result):
                if self._loop is None:
                    return False
                asyncio.run_coroutine_threadsafe(result, self._loop)
            return True
        except Exception as e:
            logger.debug(f"Failed to send frame for track {track_index}: {e}")
            return False

    def send_parameters(self, params: dict[str, Any]) -> None:
        """Send parameter updates to the Livepeer control channel."""
        if not self.is_connected or self._control_writer is None:
            return
        if self._loop is None:
            return

        try:
            asyncio.run_coroutine_threadsafe(
                self._control_writer.write(
                    {
                        "type": "parameters",
                        "params": params,
                    }
                ),
                self._loop,
            )
        except Exception as e:  # pragma: no cover - defensive scheduling guard
            logger.error(f"Failed to send control parameters: {e}")

    async def disconnect(self) -> None:
        """Close Livepeer channels and background tasks."""
        await self._shutdown()

    def get_stats(self) -> dict[str, Any]:
        stats = copy.deepcopy(self._stats)
        if stats["connected_at"] is not None:
            stats["uptime_seconds"] = time.time() - stats["connected_at"]
        return stats

    def _record_pacing_observation(self, decision: MediaPacingDecision) -> None:
        pacing = self._stats.get("pacing")
        if not isinstance(pacing, dict):
            return
        pacing["observations"] += 1
        if decision.has_valid_ts:
            pacing["valid_timestamp_frames"] += 1
        if decision.hard_reset:
            pacing["hard_resets"] += 1
        if decision.soft_reanchor:
            pacing["soft_reanchors"] += 1
        pacing["sleep_total_s"] += decision.sleep_s

        if decision.drift_s is not None:
            abs_drift = abs(decision.drift_s)
            pacing["drift_samples"] += 1
            pacing["drift_abs_sum_s"] += abs_drift
            pacing["drift_max_abs_s"] = max(pacing["drift_max_abs_s"], abs_drift)

        # Positive stall_delta captures per-frame wall-clock stalls even when
        # cumulative drift stays bounded by later catch-up.
        if decision.stall_delta_s is not None and decision.stall_delta_s > 0:
            pacing["stall_positive_samples"] += 1
            pacing["stall_positive_sum_s"] += decision.stall_delta_s
            pacing["stall_positive_max_s"] = max(
                pacing["stall_positive_max_s"], decision.stall_delta_s
            )

    async def _shutdown(
        self,
        *,
        unexpected_reason: str | None = None,
        current_task: asyncio.Task | None = None,
    ) -> None:
        """Tear down media, control, and job resources."""
        if current_task is None:
            current_task = asyncio.current_task()

        async with self._shutdown_lock:
            if self._shutdown_started:
                return

            self._shutdown_started = True
            media_handles = self._take_media_handles()
            control_writer = self._control_writer
            events_task = self._events_task
            ping_task = self._ping_task
            payment_task = self._payment_task
            job = self._job

            self._events_task = None
            self._ping_task = None
            self._payment_task = None
            self._job = None
            self._control_writer = None
            self._connected = False
            self._connection_id = None
            set_connection_id(None)

        await self._teardown_media_handles(media_handles, current_task=current_task)
        await self._drain_or_cancel_task("events loop", events_task, current_task)
        await self._drain_or_cancel_task("ping loop", ping_task, current_task)
        await self._drain_or_cancel_task("payment loop", payment_task, current_task)

        self._fail_pending_requests(unexpected_reason or "Livepeer connection closed")

        if control_writer is not None:
            await self._close_resource("control writer", control_writer.close())

        if job is not None:
            await self._close_resource("job", job.close())

        logger.info("Disconnected")

    def _media_is_stopped(self) -> bool:
        return (
            not self._media_connected
            and not self._media_subscriber_tasks
            and not any(self._media_publishers)
            and not any(self._media_outputs)
        )

    def _take_media_handles(self) -> _MediaHandles:
        handles = _MediaHandles(
            subscriber_tasks=list(self._media_subscriber_tasks),
            publishers=list(self._media_publishers),
            outputs=list(self._media_outputs),
        )
        self._media_subscriber_tasks = []
        self._media_publishers = []
        self._media_outputs = []
        self.input_tracks = []
        self.output_handlers = [FrameOutputHandler()]
        self.audio_output_handler = AudioOutputHandler()
        self.source_node_to_track_index = {}
        self._media_connected = False
        return handles

    async def _teardown_media_handles(
        self,
        handles: _MediaHandles,
        *,
        current_task: asyncio.Task | None,
    ) -> None:
        """Close media resources and let subscribers exit before cancellation."""
        for output, subscriber_task in zip(
            handles.outputs, handles.subscriber_tasks, strict=False
        ):
            if output is not None and subscriber_task is not current_task:
                await self._close_resource("media output", output.close())
            await self._drain_or_cancel_task(
                "media subscriber", subscriber_task, current_task
            )
        for publisher in handles.publishers:
            if publisher is not None:
                await self._close_resource("media publisher", publisher.close())

    async def _drain_or_cancel_task(
        self,
        name: str,
        task: asyncio.Task | None,
        current_task: asyncio.Task | None,
    ) -> None:
        """Let a task finish briefly before falling back to cancellation."""
        if task is None or task is current_task:
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=TASK_DRAIN_TIMEOUT_S)
            return
        except asyncio.CancelledError:
            if task.done():
                return
            raise
        except TimeoutError:
            task.cancel()
        except Exception as e:
            logger.warning("Error while draining %s task: %s", name, e)
            return

        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=SHUTDOWN_TIMEOUT_S)
        except (asyncio.CancelledError, TimeoutError):
            pass
        except Exception as e:
            logger.warning("Error while cancelling %s task: %s", name, e)

    async def _close_resource(self, name: str, coro) -> None:
        """Await a close coroutine with a timeout."""
        try:
            await asyncio.wait_for(coro, timeout=SHUTDOWN_TIMEOUT_S)
        except TimeoutError:
            logger.warning(
                "Timed out closing %s after %.1fs, proceeding", name, SHUTDOWN_TIMEOUT_S
            )
        except Exception as e:
            logger.warning("Error while closing %s: %s", name, e)

    async def _send_control_message(
        self, control_writer: JSONLWriter | None, message: dict[str, Any]
    ) -> None:
        if control_writer is None:
            raise RuntimeError("Livepeer control channel is not available")
        await control_writer.write(message)

    def _fail_pending_requests(self, reason: str) -> None:
        for request_id, future in self._pending_requests.items():
            if not future.done():
                future.set_exception(
                    RuntimeError(f"{reason} (pending request {request_id})")
                )
        self._pending_requests.clear()


# ---------------------------------------------------------------------------
# Cloud log re-emission — forwards runner log lines into local logging
# ---------------------------------------------------------------------------


def _handle_cloud_logs(data: dict[str, Any]) -> None:
    """Re-emit cloud runner log lines into local Python logging."""
    cloud_logger = logging.getLogger("scope.cloud")
    lines = data.get("lines", [])
    if not isinstance(lines, list):
        return
    for line in lines:
        if not isinstance(line, str):
            continue
        level = logging.INFO
        if " - ERROR - " in line:
            level = logging.ERROR
        elif " - WARNING - " in line:
            level = logging.WARNING
        elif " - DEBUG - " in line:
            level = logging.DEBUG
        cloud_logger.log(level, "%s", line)
