"""LivepeerClient - Trickle HTTP client for Livepeer LV2V inference.

This client is transport-only. It manages the Livepeer job lifecycle and exposes
simple frame/parameter methods used by the relay manager.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import numpy as np
from av import AudioFrame, VideoFrame
from livepeer_gateway.channel_reader import JSONLReader
from livepeer_gateway.channel_writer import JSONLWriter
from livepeer_gateway.errors import SkipPaymentCycle
from livepeer_gateway.lv2v import StartJobRequest, start_lv2v
from livepeer_gateway.media_output import MediaOutput
from livepeer_gateway.media_publish import MediaPublish, MediaPublishConfig

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
    subscriber_task: asyncio.Task | None
    publisher: MediaPublish | None
    output: MediaOutput | None


class LivepeerClient:
    """Livepeer LV2V transport client.

    This client opens a Livepeer LV2V job, publishes frames to the input channel,
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
        self._media_publisher: MediaPublish | None = None
        self._media_output: MediaOutput | None = None
        self._control_writer: JSONLWriter | None = None
        self._media_subscriber_task: asyncio.Task | None = None
        self._events_task: asyncio.Task | None = None
        self._ping_task: asyncio.Task | None = None
        self._payment_task: asyncio.Task | None = None
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._callbacks: list[Callable[[VideoFrame], None]] = []
        self._audio_callbacks: list[Callable[[AudioFrame], None]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._connected = False
        self._media_connected = False
        self._shutdown_started = False
        self._shutdown_lock = asyncio.Lock()
        self._runner_ready_event = asyncio.Event()
        self._connection_id: str | None = None

        self._stats = {
            "connected_at": None,
            "api_requests_sent": 0,
            "api_requests_successful": 0,
        }

    @property
    def is_connected(self) -> bool:
        return self._connected and self._job is not None

    @property
    def media_connected(self) -> bool:
        return (
            self.is_connected
            and self._media_connected
            and (
                self._media_publisher is not None
                or self._media_subscriber_task is not None
            )
        )

    @property
    def connection_id(self) -> str | None:
        return self._connection_id

    async def connect(self, initial_parameters: dict | None = None) -> None:
        """Create a Livepeer job and start the events channel."""
        if self.is_connected:
            await self.disconnect()

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

        # start_lv2v is synchronous and may block on network I/O, so run it in
        # a worker thread to avoid blocking the event loop.
        self._job = await asyncio.to_thread(
            start_lv2v,
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

        # start_lv2v runs in a worker thread without an event loop, so
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

        logger.info("Connected to Livepeer LV2V")

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
        # HACK: Ignore the default app_id to use the orchestrator's own config
        if not value or value == "Daydream/scope-app--prod/ws":
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
                "`/ws` (for example `daydream/scope-app/ws`)."
            ) from None
        if parsed.scheme not in {"ws", "wss"}:
            raise ValueError("Invalid ws_url. Expected a valid ws:// or wss:// URL.")
        if not parsed.hostname:
            raise ValueError("Invalid ws_url. Expected a valid ws:// or wss:// URL.")
        return ws_url

    async def start_media(self, initial_parameters: dict | None = None) -> None:
        """Start media I/O and notify runner about stream start parameters."""
        if not self.is_connected or self._job is None:
            raise RuntimeError("Livepeer job is not connected")
        if self._loop is None:
            raise RuntimeError("Livepeer event loop not initialized")

        if self.media_connected:
            logger.info("Media already started")
            return

        request_id = str(uuid.uuid4())
        future: asyncio.Future = self._loop.create_future()
        self._pending_requests[request_id] = future
        await self._send_control(
            {
                "type": "start_stream",
                "request_id": request_id,
                "params": initial_parameters or {},
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

        input_url: str | None = None
        output_url: str | None = None
        for channel in channels:
            if not isinstance(channel, dict):
                continue
            url = channel.get("url")
            direction = channel.get("direction")
            if not isinstance(url, str) or not isinstance(direction, str):
                continue
            if direction == "in":
                input_url = url
            elif direction == "out":
                output_url = url

        if input_url is None and output_url is None:
            raise RuntimeError("stream_started response missing usable channels")

        publisher: MediaPublish | None = None
        if input_url is not None:
            publisher = MediaPublish(
                input_url, config=MediaPublishConfig(fps=self._fps)
            )

        media_output: MediaOutput | None = None
        subscriber: asyncio.Task | None = None
        if output_url is not None:
            media_output = MediaOutput(output_url, start_seq=-1)
            subscriber = asyncio.create_task(self._receive_loop(media_output))

        self._media_connected = True

        self._media_publisher = publisher
        self._media_output = media_output
        self._media_subscriber_task = subscriber
        logger.info(
            "Media channels started (input=%s output=%s)",
            bool(input_url),
            bool(output_url),
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

    async def _receive_loop(self, output: MediaOutput) -> None:
        """Consume output frames from Livepeer and notify callbacks."""
        prev_video_ts: float | None = None
        prev_dispatch_time: float | None = None
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
                    for callback in list(self._audio_callbacks):
                        try:
                            callback(frame)
                        except (
                            Exception
                        ) as e:  # pragma: no cover - defensive callback guard
                            logger.error(f"Audio callback failed: {e}")
                    continue
                if decoded_kind != "video":
                    continue

                # Hack: Sleep here to pace output (pts based)
                time_base = getattr(frame, "time_base", None)
                video_ts = (
                    frame.pts * float(time_base)
                    if time_base is not None and frame.pts is not None
                    else None
                )
                now_monotonic = time.monotonic()
                if prev_video_ts is not None and video_ts is not None:
                    seconds_delta = video_ts - prev_video_ts
                    if seconds_delta > 0 and prev_dispatch_time is not None:
                        elapsed = now_monotonic - prev_dispatch_time
                        wait = seconds_delta - elapsed
                        if wait > 0:
                            await asyncio.sleep(wait)
                prev_video_ts = video_ts
                prev_dispatch_time = time.monotonic()
                for callback in list(self._callbacks):
                    try:
                        callback(frame)
                    except (
                        Exception
                    ) as e:  # pragma: no cover - defensive callback guard
                        logger.error(f"Frame callback failed: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            unexpected_reason = f"Livepeer output loop failed: {e}"
            logger.error(f"Output loop failed: {e}")
        finally:
            try:
                await output.close()
            except Exception as e:
                logger.warning(f"Error while closing media output: {e}")
            # Avoid clearing a newer media output when stop/start races with this loop.
            if self._media_output is output:
                self._media_output = None
            if self._media_connected and not self._shutdown_started:
                if unexpected_reason is None:
                    unexpected_reason = "Livepeer output loop stopped unexpectedly"
                logger.warning("%s; stopping media only", unexpected_reason)
                await self.stop_media(current_task=asyncio.current_task())
            logger.info("Output loop stopped")

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
                    if self._media_publisher is not None:
                        logger.info(self._media_publisher.get_stats())
                    if self._media_output is not None:
                        logger.info(self._media_output.get_stats())
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
        if not self.media_connected or self._media_publisher is None:
            return False

        if isinstance(frame, np.ndarray):
            frame = VideoFrame.from_ndarray(frame, format="rgb24")

        try:
            result = self._media_publisher.write_frame(frame)
            if inspect.isawaitable(result):
                if self._loop is None:
                    return False
                asyncio.run_coroutine_threadsafe(result, self._loop)
            return True
        except Exception as e:
            logger.debug(f"Failed to send frame: {e}")
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

    def add_frame_callback(self, callback: Callable[[VideoFrame], None]) -> None:
        self._callbacks.append(callback)

    def remove_frame_callback(self, callback: Callable[[VideoFrame], None]) -> None:
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def add_audio_callback(self, callback: Callable[[AudioFrame], None]) -> None:
        self._audio_callbacks.append(callback)

    def remove_audio_callback(self, callback: Callable[[AudioFrame], None]) -> None:
        if callback in self._audio_callbacks:
            self._audio_callbacks.remove(callback)

    def get_stats(self) -> dict[str, Any]:
        stats = dict(self._stats)
        if stats["connected_at"] is not None:
            stats["uptime_seconds"] = time.time() - stats["connected_at"]
        return stats

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
            and self._media_subscriber_task is None
            and self._media_publisher is None
            and self._media_output is None
        )

    def _take_media_handles(self) -> _MediaHandles:
        handles = _MediaHandles(
            subscriber_task=self._media_subscriber_task,
            publisher=self._media_publisher,
            output=self._media_output,
        )
        self._media_subscriber_task = None
        self._media_publisher = None
        self._media_output = None
        self._media_connected = False
        return handles

    async def _teardown_media_handles(
        self,
        handles: _MediaHandles,
        *,
        current_task: asyncio.Task | None,
    ) -> None:
        """Close media resources and let the subscriber exit before cancelling it."""
        output = handles.output
        subscriber_task = handles.subscriber_task

        if output is not None and subscriber_task is not current_task:
            await self._close_resource("media output", output.close())

        await self._drain_or_cancel_task(
            "media subscriber", subscriber_task, current_task
        )

        if handles.publisher is not None:
            await self._close_resource("media publisher", handles.publisher.close())

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
