"""Reliable metrics reporter that forwards network_events to Daydream /v1/metrics.

Implements the fail-loud / no-loss client contract defined in the Daydream
metrics API: events are buffered locally and flushed in batches via HTTP POST.
Reliability is achieved entirely through client-side buffering with exponential
backoff, disk-backed resume, and response-aware retry logic.

Environment Variables:
    DAYDREAM_METRICS_URL: Endpoint URL (default: https://api.daydream.monster/v1/metrics)
    SCOPE_CLOUD_API_KEY: Bearer token for authentication (required to enable)
    SCOPE_METRICS_ENABLED: Explicit enable/disable ("true"/"false"; default: on when key present)
    SCOPE_METRICS_FLUSH_INTERVAL_MS: Flush interval in milliseconds (default: 2000)
    SCOPE_METRICS_MAX_BATCH: Max events per batch (default: 500, API limit)
    SCOPE_METRICS_MAX_BUFFER: Max buffered events before dropping oldest (default: 50000)
    SCOPE_METRICS_BUFFER_PATH: Path for disk-backed resume buffer (default: ~/.daydream-scope/metrics_buffer.jsonl)
"""

import asyncio
import json
import logging
import os
import platform
import threading
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DAYDREAM_METRICS_URL = os.getenv(
    "DAYDREAM_METRICS_URL", "https://api.daydream.monster/v1/metrics"
)
MAX_BATCH_SIZE = int(os.getenv("SCOPE_METRICS_MAX_BATCH", "500"))
MAX_BUFFER_SIZE = int(os.getenv("SCOPE_METRICS_MAX_BUFFER", "50000"))
FLUSH_INTERVAL_S = int(os.getenv("SCOPE_METRICS_FLUSH_INTERVAL_MS", "2000")) / 1000.0
MAX_BODY_BYTES = 1_000_000  # 1 MiB API limit
INITIAL_BACKOFF_S = 1.0
MAX_BACKOFF_S = 60.0
RATE_LIMIT_WINDOW_S = 60.0

_HOSTNAME = platform.node()


def _default_buffer_path() -> Path | None:
    """Return the default disk buffer path, or None if explicitly disabled."""
    env = os.getenv("SCOPE_METRICS_BUFFER_PATH")
    if env == "":
        return None
    if env:
        return Path(env)
    base = os.getenv("DAYDREAM_SCOPE_MODELS_DIR")
    if base:
        return Path(base).parent / "metrics_buffer.jsonl"
    return Path.home() / ".daydream-scope" / "metrics_buffer.jsonl"


def is_metrics_reporter_enabled() -> bool:
    """Check if the metrics reporter should be started."""
    explicit = os.getenv("SCOPE_METRICS_ENABLED", "").lower()
    if explicit == "false":
        return False
    if explicit == "true":
        return True
    return os.getenv("SCOPE_CLOUD_API_KEY") is not None


def map_envelope_to_event(envelope: dict[str, Any]) -> dict[str, Any]:
    """Map a Scope stream_trace Kafka envelope to a /v1/metrics Event object.

    Scope envelope: {id, type:"stream_trace", timestamp, data:{type, client_source, ...}}
    API Event:      {type, data, id?, timestamp?}
    """
    data = envelope.get("data", {})
    event: dict[str, Any] = {
        "type": data.get("type", envelope.get("type", "unknown")),
        "data": data,
    }
    if envelope.get("id"):
        event["id"] = envelope["id"]
    if envelope.get("timestamp"):
        event["timestamp"] = str(envelope["timestamp"])
    return event


class MetricsReporter:
    """Reliable buffered metrics forwarder to Daydream /v1/metrics.

    Thread-safe: ``enqueue()`` can be called from any thread.
    The async flush loop runs on the event loop captured at ``start()``.
    """

    def __init__(self, api_key: str, url: str = DAYDREAM_METRICS_URL):
        self._api_key = api_key
        self._url = url
        self._buffer: deque[dict[str, Any]] = deque(maxlen=MAX_BUFFER_SIZE)
        self._lock = threading.Lock()
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._flush_task: asyncio.Task | None = None
        self._started = False
        self._paused = False  # Set on 401, cleared on key refresh
        self._backoff_s = INITIAL_BACKOFF_S
        self._client = None
        self._buffer_path = _default_buffer_path()

    async def start(self) -> bool:
        """Start the reporter and its background flush loop."""
        try:
            import httpx

            self._client = httpx.AsyncClient(timeout=30.0)
        except ImportError:
            logger.warning("httpx not installed, metrics reporting disabled")
            return False

        self._event_loop = asyncio.get_running_loop()
        self._started = True

        self._load_disk_buffer()

        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info(
            "Metrics reporter started (url=%s, flush=%.1fs, max_batch=%d)",
            self._url,
            FLUSH_INTERVAL_S,
            MAX_BATCH_SIZE,
        )
        return True

    async def stop(self):
        """Flush remaining events and shut down."""
        self._started = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        # Final flush attempt
        await self._flush_once()

        self._persist_disk_buffer()

        if self._client:
            await self._client.aclose()
            self._client = None

        logger.info(
            "Metrics reporter stopped (remaining_buffered=%d)", len(self._buffer)
        )

    def enqueue(self, envelope: dict[str, Any]) -> None:
        """Thread-safe: add a stream_trace envelope to the buffer."""
        if not self._started or self._paused:
            return
        event = map_envelope_to_event(envelope)
        with self._lock:
            self._buffer.append(event)

    def enqueue_raw_event(self, event: dict[str, Any]) -> None:
        """Thread-safe: add an already-mapped /v1/metrics event to the buffer."""
        if not self._started or self._paused:
            return
        with self._lock:
            self._buffer.append(event)

    @property
    def is_running(self) -> bool:
        return self._started

    @property
    def buffered_count(self) -> int:
        return len(self._buffer)

    def update_api_key(self, api_key: str) -> None:
        """Update the Bearer token (e.g., after key refresh) and unpause."""
        self._api_key = api_key
        self._paused = False
        self._backoff_s = INITIAL_BACKOFF_S
        logger.info("Metrics reporter API key updated, unpaused")

    # -- internal --

    def _take_batch(self, max_size: int = MAX_BATCH_SIZE) -> list[dict[str, Any]]:
        with self._lock:
            count = min(max_size, len(self._buffer))
            batch = [self._buffer.popleft() for _ in range(count)]
        return batch

    def _requeue_batch(self, batch: list[dict[str, Any]]) -> None:
        with self._lock:
            self._buffer.extendleft(reversed(batch))

    async def _flush_loop(self) -> None:
        try:
            while self._started:
                await asyncio.sleep(FLUSH_INTERVAL_S)
                if self._paused or not self._buffer:
                    continue
                await self._flush_once()
        except asyncio.CancelledError:
            pass

    async def _flush_once(self) -> None:
        if not self._buffer or not self._client:
            return

        batch = self._take_batch()
        if not batch:
            return

        body = self._build_body(batch)
        body_bytes = json.dumps(body).encode("utf-8")

        # If body exceeds 1 MiB, split the batch
        if len(body_bytes) > MAX_BODY_BYTES and len(batch) > 1:
            half = len(batch) // 2
            self._requeue_batch(batch[half:])
            batch = batch[:half]
            body = self._build_body(batch)

        try:
            response = await self._client.post(
                self._url,
                json=body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            await self._handle_response(response, batch)
        except Exception as e:
            # Network error: treat as 5xx
            logger.warning("Metrics POST failed (network): %s", e)
            self._requeue_batch(batch)
            await self._wait_backoff()

    async def _handle_response(
        self, response: Any, batch: list[dict[str, Any]]
    ) -> None:
        status = response.status_code

        if status == 200:
            self._backoff_s = INITIAL_BACKOFF_S
            logger.debug("Metrics batch accepted (%d events)", len(batch))
            return

        if status == 400:
            logger.error(
                "Metrics batch rejected (400 malformed), dropping %d events: %s",
                len(batch),
                response.text[:200],
            )
            return

        if status == 401:
            logger.error(
                "Metrics reporter unauthorized (401), pausing until key refresh"
            )
            self._paused = True
            self._requeue_batch(batch)
            return

        if status == 413:
            logger.warning("Metrics batch too large (413), halving batch size")
            self._requeue_batch(batch)
            # Next flush will naturally try a smaller batch since _take_batch
            # gets called again; force a smaller max on immediate retry.
            half_batch = self._take_batch(max(1, len(batch) // 2))
            if half_batch:
                body = self._build_body(half_batch)
                try:
                    retry_resp = await self._client.post(
                        self._url,
                        json=body,
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                    )
                    await self._handle_response(retry_resp, half_batch)
                except Exception:
                    self._requeue_batch(half_batch)
            return

        if status == 429:
            logger.warning(
                "Metrics rate limited (429), waiting %.0fs", RATE_LIMIT_WINDOW_S
            )
            self._requeue_batch(batch)
            await asyncio.sleep(RATE_LIMIT_WINDOW_S)
            return

        # 5xx or unexpected status
        logger.warning("Metrics POST failed (HTTP %d), retrying with backoff", status)
        self._requeue_batch(batch)
        await self._wait_backoff()

    async def _wait_backoff(self) -> None:
        await asyncio.sleep(self._backoff_s)
        self._backoff_s = min(self._backoff_s * 2, MAX_BACKOFF_S)

    def _build_body(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "app": "scope",
            "host": _HOSTNAME,
            "events": batch,
        }

    # -- disk persistence --

    def _load_disk_buffer(self) -> None:
        if not self._buffer_path or not self._buffer_path.exists():
            return
        loaded = 0
        try:
            with open(self._buffer_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        self._buffer.append(event)
                        loaded += 1
                    except json.JSONDecodeError:
                        continue
            # Clear the file after loading
            self._buffer_path.write_text("")
            if loaded:
                logger.info("Loaded %d events from disk buffer", loaded)
        except Exception as e:
            logger.warning("Failed to load disk buffer: %s", e)

    def _persist_disk_buffer(self) -> None:
        if not self._buffer_path or not self._buffer:
            return
        try:
            self._buffer_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._buffer_path, "a") as f:
                with self._lock:
                    for event in self._buffer:
                        f.write(json.dumps(event) + "\n")
            logger.info("Persisted %d events to disk buffer", len(self._buffer))
        except Exception as e:
            logger.warning("Failed to persist disk buffer: %s", e)


# -- Global accessor pattern (mirrors kafka_publisher.py) --

_reporter: MetricsReporter | None = None


def get_metrics_reporter() -> MetricsReporter | None:
    """Get the global MetricsReporter instance."""
    return _reporter


def set_metrics_reporter(reporter: MetricsReporter | None) -> None:
    """Set the global MetricsReporter instance."""
    global _reporter
    _reporter = reporter


def report_event(envelope: dict[str, Any]) -> None:
    """Convenience: enqueue an envelope for reporting. No-op if disabled."""
    reporter = _reporter
    if reporter and reporter.is_running:
        reporter.enqueue(envelope)
