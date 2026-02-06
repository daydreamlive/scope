"""Kafka event publisher for cloud deployments.

This module provides async Kafka event publishing for use in fal_app and other
cloud deployment contexts. Events are only emitted when Kafka credentials are
configured via environment variables.

Environment Variables:
    KAFKA_BOOTSTRAP_SERVERS: Comma-separated list of Kafka broker addresses (required to enable)
    KAFKA_TOPIC: Topic name for events (default: "scope-events")
    KAFKA_SASL_USERNAME: SASL username for authentication (optional)
    KAFKA_SASL_PASSWORD: SASL password for authentication (optional)
"""

import json
import logging
import os
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class KafkaPublisher:
    """Async Kafka event publisher.

    Reads configuration from environment variables at runtime (not module load time),
    which is required for fal.ai where secrets may arrive late.
    """

    def __init__(self):
        self._producer = None
        self._started = False
        self._topic = None

    async def start(self) -> bool:
        """Start the Kafka producer.

        Returns:
            True if started successfully, False if not configured or failed.
        """
        bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
        self._topic = os.getenv("KAFKA_TOPIC", "scope-events")
        sasl_username = os.getenv("KAFKA_SASL_USERNAME")
        sasl_password = os.getenv("KAFKA_SASL_PASSWORD")

        if not bootstrap_servers:
            logger.info("Kafka not configured, event publishing disabled")
            return False

        try:
            from aiokafka import AIOKafkaProducer

            config = {
                "bootstrap_servers": bootstrap_servers,
                "value_serializer": lambda v: json.dumps(v).encode("utf-8"),
                "key_serializer": lambda k: k.encode("utf-8") if k else None,
            }

            if sasl_username and sasl_password:
                import ssl

                ssl_context = ssl.create_default_context()
                config.update(
                    {
                        "security_protocol": "SASL_SSL",
                        "sasl_mechanism": "PLAIN",
                        "sasl_plain_username": sasl_username,
                        "sasl_plain_password": sasl_password,
                        "ssl_context": ssl_context,
                    }
                )

            self._producer = AIOKafkaProducer(**config)
            await self._producer.start()
            self._started = True
            logger.info(
                f"Kafka publisher started, topic: {self._topic}, "
                f"servers: {bootstrap_servers}"
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

    async def publish(self, event_type: str, data: dict[str, Any]) -> bool:
        """Publish an event to Kafka.

        Args:
            event_type: Type of event (e.g., "stream_started", "pipeline_loaded")
            data: Event-specific data payload

        Returns:
            True if published successfully, False otherwise.
        """
        if not self._started or not self._producer:
            return False

        event_id = str(uuid.uuid4())
        timestamp_ms = str(int(time.time() * 1000))

        event = {
            "id": event_id,
            "type": "stream_trace",
            "timestamp": timestamp_ms,
            "data": {"event_type": event_type, "client_source": "scope", **data},
        }

        try:
            await self._producer.send_and_wait(self._topic, value=event, key=event_id)
            logger.info(f"Published Kafka event: {event_type} (id={event_id})")
            return True
        except Exception as e:
            logger.error(f"Failed to publish Kafka event {event_type}: {e}")
            return False

    @property
    def is_running(self) -> bool:
        """Check if the publisher is running."""
        return self._started
