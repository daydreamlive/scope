"""In-process event bus for server lifecycle events.

Provides a lightweight pub/sub mechanism that replaces direct Kafka publishing.
Events emitted here can be consumed via the SSE endpoint (GET /api/v1/events/stream)
by external services like fal_app, which then publish them to Kafka.

The event bus is thread-safe: emit_event() can be called from sync worker threads
(e.g., FrameProcessor, PipelineProcessor) and events are delivered to async subscribers.
"""

import asyncio
import json
import logging
import threading
import time
from collections.abc import AsyncGenerator
from typing import Any

logger = logging.getLogger(__name__)


class EventBus:
    """Thread-safe in-process event bus with async subscriber support.

    Allows sync worker threads to emit events that are delivered to async
    subscribers (e.g., SSE endpoint consumers) via bounded asyncio.Queues.
    """

    def __init__(self):
        self._subscribers: list[asyncio.Queue] = []
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the event loop for thread-safe event delivery.

        Must be called from the main async context during startup.
        """
        self._loop = loop

    def _build_event(
        self,
        event_type: str,
        session_id: str | None = None,
        pipeline_ids: list[str] | None = None,
        user_id: str | None = None,
        error: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a structured event dict from the given parameters."""
        event: dict[str, Any] = {
            "event_type": event_type,
            "timestamp": str(int(time.time() * 1000)),
        }
        if session_id:
            event["session_id"] = session_id
        if pipeline_ids:
            event["pipeline_ids"] = pipeline_ids
        if user_id:
            event["user_id"] = user_id
        if error:
            event["error"] = error
        if metadata:
            event.update(metadata)
        return event

    def _deliver(self, event: dict[str, Any]):
        """Deliver an event to all subscribers (must be called from the event loop thread)."""
        with self._lock:
            subscribers = list(self._subscribers)

        for q in subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(
                    f"Event bus subscriber queue full, dropping event: "
                    f"{event.get('event_type')}"
                )

    def emit(
        self,
        event_type: str,
        session_id: str | None = None,
        pipeline_ids: list[str] | None = None,
        user_id: str | None = None,
        error: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit an event from any context (thread-safe).

        Safe to call from sync worker threads (e.g., FrameProcessor).
        Events are delivered to subscribers on the main event loop.

        Args:
            event_type: Type of event (e.g., "stream_started", "stream_stopped")
            session_id: Optional session ID associated with the event
            pipeline_ids: Optional list of pipeline IDs associated with the event
            user_id: Optional user ID associated with the event
            error: Optional error details for error events
            metadata: Optional additional metadata
        """
        if not self._loop or not self._loop.is_running():
            return

        event = self._build_event(
            event_type=event_type,
            session_id=session_id,
            pipeline_ids=pipeline_ids,
            user_id=user_id,
            error=error,
            metadata=metadata,
        )

        self._loop.call_soon_threadsafe(self._deliver, event)
        logger.debug(f"Emitted event: {event_type}")

    async def emit_async(
        self,
        event_type: str,
        session_id: str | None = None,
        pipeline_ids: list[str] | None = None,
        user_id: str | None = None,
        error: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit an event from an async context.

        Args:
            event_type: Type of event (e.g., "session_created", "session_closed")
            session_id: Optional session ID associated with the event
            pipeline_ids: Optional list of pipeline IDs associated with the event
            user_id: Optional user ID associated with the event
            error: Optional error details for error events
            metadata: Optional additional metadata
        """
        event = self._build_event(
            event_type=event_type,
            session_id=session_id,
            pipeline_ids=pipeline_ids,
            user_id=user_id,
            error=error,
            metadata=metadata,
        )
        self._deliver(event)
        logger.debug(f"Emitted event (async): {event_type}")

    async def subscribe(self) -> AsyncGenerator[dict[str, Any], None]:
        """Subscribe to the event stream.

        Yields events as they arrive. The subscription is automatically cleaned
        up when the generator is closed.

        Yields:
            Event dicts with at least an "event_type" key.
        """
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        with self._lock:
            self._subscribers.append(q)
        try:
            while True:
                event = await q.get()
                yield event
        finally:
            with self._lock:
                self._subscribers.remove(q)

    async def subscribe_sse(self) -> AsyncGenerator[str, None]:
        """Subscribe and yield events formatted as SSE data lines.

        Yields:
            SSE-formatted strings: "data: {json}\\n\\n"
        """
        async for event in self.subscribe():
            yield f"data: {json.dumps(event)}\n\n"


# Global event bus instance
_event_bus: EventBus | None = None


def get_event_bus() -> EventBus | None:
    """Get the global EventBus instance."""
    return _event_bus


def set_event_bus(bus: EventBus | None):
    """Set the global EventBus instance."""
    global _event_bus
    _event_bus = bus


def emit_event(
    event_type: str,
    session_id: str | None = None,
    pipeline_ids: list[str] | None = None,
    user_id: str | None = None,
    error: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Convenience function to emit an event using the global event bus.

    Safe to call even if the event bus is not initialized - it will no-op.
    Use this from sync contexts (threads).

    Args:
        event_type: Type of event (e.g., "stream_started", "stream_stopped")
        session_id: Optional session ID associated with the event
        pipeline_ids: Optional list of pipeline IDs associated with the event
        user_id: Optional user ID associated with the event
        error: Optional error details for error events
        metadata: Optional additional metadata
    """
    bus = get_event_bus()
    if bus:
        bus.emit(
            event_type=event_type,
            session_id=session_id,
            pipeline_ids=pipeline_ids,
            user_id=user_id,
            error=error,
            metadata=metadata,
        )


async def emit_event_async(
    event_type: str,
    session_id: str | None = None,
    pipeline_ids: list[str] | None = None,
    user_id: str | None = None,
    error: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Async convenience function to emit an event using the global event bus.

    Safe to call even if the event bus is not initialized - it will no-op.
    Use this from async contexts.

    Args:
        event_type: Type of event (e.g., "session_created", "session_closed")
        session_id: Optional session ID associated with the event
        pipeline_ids: Optional list of pipeline IDs associated with the event
        user_id: Optional user ID associated with the event
        error: Optional error details for error events
        metadata: Optional additional metadata
    """
    bus = get_event_bus()
    if bus:
        await bus.emit_async(
            event_type=event_type,
            session_id=session_id,
            pipeline_ids=pipeline_ids,
            user_id=user_id,
            error=error,
            metadata=metadata,
        )
