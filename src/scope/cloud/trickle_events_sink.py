"""Telemetry sink that forwards events over the trickle events channel.

On the cloud runner there is normally no reachable Kafka broker, so telemetry
emitted via ``scope.server.kafka_publisher.publish_event`` must instead be
forwarded to the local SDK client. This sink enqueues each (already-built) event
envelope and a drain task writes it onto the per-session events channel as a
``{"type": "telemetry", "event": {...}}`` message. The client receives it and
re-publishes verbatim to its own egress sink.

This mirrors the notification-forwarding pattern in ``livepeer_app.py``
(``_enqueue_notification`` / ``_forward_notifications_to_events``): a bounded
queue with drop-oldest semantics so telemetry is best-effort and never blocks
the media or control paths. It structurally satisfies the
``scope.server.kafka_publisher.TelemetrySink`` protocol.
"""

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Bounded so a stalled/slow events channel can never grow memory without limit.
DEFAULT_QUEUE_MAXSIZE = 256


class TrickleEventsSink:
    """Forward telemetry events onto the trickle events channel.

    ``emit`` is safe to call from worker threads (FrameProcessor). It marshals
    onto the runner event loop and drops the oldest queued event on overflow.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        maxsize: int = DEFAULT_QUEUE_MAXSIZE,
    ):
        self._loop = loop
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(
            maxsize=maxsize
        )
        self._drop_warned = False

    def emit(self, event: dict[str, Any]) -> None:
        """Thread-safe, non-blocking enqueue (drop-oldest on overflow)."""

        def _put_with_drop() -> None:
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop the oldest event to make room for the newest.
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    self._queue.put_nowait(event)
                except asyncio.QueueFull:
                    if not self._drop_warned:
                        logger.warning("Dropped telemetry event under queue pressure")
                        self._drop_warned = True

        try:
            self._loop.call_soon_threadsafe(_put_with_drop)
        except RuntimeError:
            # Loop is closing; drop silently.
            pass

    async def drain_to(self, events_writer: Any, stop_event: asyncio.Event) -> None:
        """Drain queued events onto the events channel until stopped.

        Exits on the ``stop_event``, on a ``None`` sentinel (see
        ``request_stop``), or on cancellation.
        """
        try:
            while not stop_event.is_set():
                try:
                    event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except TimeoutError:
                    continue
                if event is None:
                    break
                try:
                    await events_writer.write({"type": "telemetry", "event": event})
                except Exception:
                    logger.debug(
                        "Failed to forward telemetry to events channel",
                        exc_info=True,
                    )
        except asyncio.CancelledError:
            pass

    def request_stop(self) -> bool:
        """Enqueue the sentinel to wake the drain loop so it exits.

        Must be called from the runner event loop thread. Returns ``False`` if
        the queue is saturated and the caller should cancel the drain task.
        """
        try:
            self._queue.put_nowait(None)
            return True
        except asyncio.QueueFull:
            return False
