"""Kafka event publisher for stream lifecycle events.

This module provides optional Kafka event publishing for stream lifecycle events.
Events are only emitted when Kafka credentials are configured via environment variables.

Environment Variables:
    KAFKA_BOOTSTRAP_SERVERS: Comma-separated list of Kafka broker addresses (required to enable)
    KAFKA_TOPIC: Topic name for events (default: "scope-events")
    KAFKA_SASL_USERNAME: SASL username for authentication (optional)
    KAFKA_SASL_PASSWORD: SASL password for authentication (optional)
"""

import asyncio
import json
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Kafka configuration from environment variables
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "scope-events")
KAFKA_SASL_USERNAME = os.getenv("KAFKA_SASL_USERNAME")
KAFKA_SASL_PASSWORD = os.getenv("KAFKA_SASL_PASSWORD")


def is_kafka_enabled() -> bool:
    """Check if Kafka is configured via environment variables."""
    return KAFKA_BOOTSTRAP_SERVERS is not None


class KafkaPublisher:
    """Async Kafka event publisher with thread-safe wrapper for sync contexts.

    This class provides:
    - Async event publishing using aiokafka
    - Thread-safe wrapper to allow calling from sync contexts (e.g., FrameProcessor threads)
    - Automatic JSON serialization of events
    - Graceful handling when Kafka is unavailable
    """

    def __init__(self):
        self._producer = None
        self._started = False
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    async def start(self) -> bool:
        """Start the Kafka producer.

        Returns:
            True if started successfully, False if Kafka is not configured or failed.
        """
        if not is_kafka_enabled():
            logger.info("Kafka not configured, event publishing disabled")
            return False

        try:
            from aiokafka import AIOKafkaProducer

            # Build producer configuration
            config = {
                "bootstrap_servers": KAFKA_BOOTSTRAP_SERVERS,
                "value_serializer": lambda v: json.dumps(v).encode("utf-8"),
                "key_serializer": lambda k: k.encode("utf-8") if k else None,
            }

            # Add SASL authentication if configured
            if KAFKA_SASL_USERNAME and KAFKA_SASL_PASSWORD:
                import ssl

                # Create SSL context for SASL_SSL (required for Confluent Cloud)
                ssl_context = ssl.create_default_context()

                config.update(
                    {
                        "security_protocol": "SASL_SSL",
                        "sasl_mechanism": "PLAIN",
                        "sasl_plain_username": KAFKA_SASL_USERNAME,
                        "sasl_plain_password": KAFKA_SASL_PASSWORD,
                        "ssl_context": ssl_context,
                    }
                )

            self._producer = AIOKafkaProducer(**config)
            await self._producer.start()
            self._started = True
            self._event_loop = asyncio.get_running_loop()

            logger.info(
                f"Kafka publisher started, publishing to topic '{KAFKA_TOPIC}' "
                f"on {KAFKA_BOOTSTRAP_SERVERS}"
            )
            return True

        except ImportError:
            logger.warning(
                "aiokafka not installed, Kafka event publishing disabled. "
                "Install with: pip install aiokafka"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to start Kafka producer: {e}")
            return False

    async def stop(self):
        """Stop the Kafka producer and flush pending messages."""
        if self._producer and self._started:
            try:
                await self._producer.stop()
                logger.info("Kafka publisher stopped")
            except Exception as e:
                logger.error(f"Error stopping Kafka producer: {e}")
            finally:
                self._started = False
                self._producer = None
                self._event_loop = None

    async def publish_async(
        self,
        event_type: str,
        session_id: str | None = None,
        connection_id: str | None = None,
        pipeline_ids: list[str] | None = None,
        user_id: str | None = None,
        error: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        connection_info: dict[str, Any] | None = None,
    ) -> bool:
        """Publish an event to Kafka asynchronously.

        Args:
            event_type: Type of event (e.g., "stream_started", "stream_stopped")
            session_id: Optional session ID associated with the event
            connection_id: Optional connection ID from fal.ai WebSocket for event correlation
            pipeline_ids: Optional list of pipeline IDs associated with the event
            user_id: Optional user ID associated with the event
            error: Optional error details for error events
            metadata: Optional additional metadata
            connection_info: Optional connection metadata (e.g., gpu_type, region)

        Returns:
            True if published successfully, False otherwise.
        """
        if not self._started or not self._producer:
            return False

        import time
        import uuid

        # Generate a unique ID for this event (used as Kafka key)
        event_id = str(uuid.uuid4())
        # Timestamp as milliseconds string (matching Go format)
        timestamp_ms = str(int(time.time() * 1000))

        # Build data payload
        data: dict[str, Any] = {
            "type": event_type,
            "client_source": "scope",
            "timestamp": timestamp_ms,
        }
        if session_id:
            data["session_id"] = session_id
        if connection_id:
            data["connection_id"] = connection_id
        if pipeline_ids:
            data["pipeline_ids"] = pipeline_ids
        if user_id:
            data["user_id"] = user_id
        if error:
            data["error"] = error
        if connection_info:
            data["connection_info"] = connection_info
        if metadata:
            data.update(metadata)

        # Event structure matching Go kafka.go format
        event = {
            "id": event_id,
            "type": "stream_trace",
            "timestamp": timestamp_ms,
            "data": data,
        }

        try:
            # Use event ID as key (matching Go format)
            key = event_id
            await self._producer.send_and_wait(KAFKA_TOPIC, value=event, key=key)
            logger.info(
                f"Published Kafka event: {event_type} (id={event_id}, session={session_id})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to publish Kafka event {event_type}: {e}")
            return False

    def publish(
        self,
        event_type: str,
        session_id: str | None = None,
        connection_id: str | None = None,
        pipeline_ids: list[str] | None = None,
        user_id: str | None = None,
        error: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        connection_info: dict[str, Any] | None = None,
    ) -> None:
        """Publish an event to Kafka from a sync context (thread-safe).

        This method is safe to call from worker threads (e.g., FrameProcessor).
        It schedules the async publish in the main event loop.

        Args:
            event_type: Type of event (e.g., "stream_started", "stream_stopped")
            session_id: Optional session ID associated with the event
            connection_id: Optional connection ID from fal.ai WebSocket for event correlation
            pipeline_ids: Optional list of pipeline IDs associated with the event
            user_id: Optional user ID associated with the event
            error: Optional error details for error events
            metadata: Optional additional metadata
            connection_info: Optional connection metadata (e.g., gpu_type, region)
        """
        if not self._started or not self._event_loop:
            return

        with self._lock:
            if not self._event_loop or not self._event_loop.is_running():
                return

            # Schedule the async publish in the main event loop
            try:
                asyncio.run_coroutine_threadsafe(
                    self.publish_async(
                        event_type=event_type,
                        session_id=session_id,
                        connection_id=connection_id,
                        pipeline_ids=pipeline_ids,
                        user_id=user_id,
                        error=error,
                        metadata=metadata,
                        connection_info=connection_info,
                    ),
                    self._event_loop,
                )
                logger.debug(f"Scheduled Kafka event: {event_type}")
            except Exception as e:
                logger.error(f"Failed to schedule Kafka event publish: {e}")

    @property
    def is_running(self) -> bool:
        """Check if the publisher is running."""
        return self._started


# Global publisher instance (initialized in app.py if Kafka is configured)
_publisher: KafkaPublisher | None = None


def get_kafka_publisher() -> KafkaPublisher | None:
    """Get the global Kafka publisher instance.

    Returns:
        The KafkaPublisher instance if initialized, None otherwise.
    """
    return _publisher


def set_kafka_publisher(publisher: KafkaPublisher | None):
    """Set the global Kafka publisher instance.

    Args:
        publisher: The KafkaPublisher instance to set as global.
    """
    global _publisher
    _publisher = publisher


def publish_event(
    event_type: str,
    session_id: str | None = None,
    connection_id: str | None = None,
    pipeline_ids: list[str] | None = None,
    user_id: str | None = None,
    error: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    connection_info: dict[str, Any] | None = None,
) -> None:
    """Convenience function to publish an event using the global publisher.

    This is safe to call even if Kafka is not configured - it will simply no-op.
    Use this from sync contexts (threads).

    Args:
        event_type: Type of event (e.g., "stream_started", "stream_stopped")
        session_id: Optional session ID associated with the event
        connection_id: Optional connection ID from fal.ai WebSocket for event correlation
        pipeline_ids: Optional list of pipeline IDs associated with the event
        user_id: Optional user ID associated with the event
        error: Optional error details for error events
        metadata: Optional additional metadata
        connection_info: Optional connection metadata (e.g., gpu_type, region)
    """
    publisher = get_kafka_publisher()
    if publisher and publisher.is_running:
        publisher.publish(
            event_type=event_type,
            session_id=session_id,
            connection_id=connection_id,
            pipeline_ids=pipeline_ids,
            user_id=user_id,
            error=error,
            metadata=metadata,
            connection_info=connection_info,
        )


async def publish_event_async(
    event_type: str,
    session_id: str | None = None,
    connection_id: str | None = None,
    pipeline_ids: list[str] | None = None,
    user_id: str | None = None,
    error: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    connection_info: dict[str, Any] | None = None,
) -> bool:
    """Async convenience function to publish an event using the global publisher.

    This is safe to call even if Kafka is not configured - it will simply no-op.
    Use this from async contexts.

    Args:
        event_type: Type of event (e.g., "stream_started", "stream_stopped")
        session_id: Optional session ID associated with the event
        connection_id: Optional connection ID from fal.ai WebSocket for event correlation
        pipeline_ids: Optional list of pipeline IDs associated with the event
        user_id: Optional user ID associated with the event
        error: Optional error details for error events
        metadata: Optional additional metadata
        connection_info: Optional connection metadata (e.g., gpu_type, region)

    Returns:
        True if published successfully, False otherwise.
    """
    publisher = get_kafka_publisher()
    if publisher and publisher.is_running:
        return await publisher.publish_async(
            event_type=event_type,
            session_id=session_id,
            connection_id=connection_id,
            pipeline_ids=pipeline_ids,
            user_id=user_id,
            error=error,
            metadata=metadata,
            connection_info=connection_info,
        )
    return False
