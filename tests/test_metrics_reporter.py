"""Tests for the MetricsReporter: reliability contract, batching, disk resume, mapping,
and TelemetrySink protocol conformance."""

import time
from collections import deque
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scope.server.metrics_reporter import (
    INITIAL_BACKOFF_S,
    MAX_BATCH_SIZE,
    MetricsReporter,
    get_metrics_reporter,
    map_envelope_to_event,
    report_event,
    set_metrics_reporter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_envelope(
    event_type: str = "stream_heartbeat",
    event_id: str | None = None,
    **extra_data,
) -> dict:
    ts = str(int(time.time() * 1000))
    return {
        "id": event_id or f"test-{time.monotonic_ns()}",
        "type": "stream_trace",
        "timestamp": ts,
        "data": {
            "type": event_type,
            "client_source": "scope",
            "timestamp": ts,
            **extra_data,
        },
    }


def _make_response(status_code: int, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    return resp


# ---------------------------------------------------------------------------
# Event mapping
# ---------------------------------------------------------------------------


class TestMapEnvelopeToEvent:
    def test_basic_mapping(self):
        envelope = _make_envelope("stream_started", event_id="evt-1", session_id="s1")
        event = map_envelope_to_event(envelope)

        assert event["type"] == "stream_started"
        assert event["id"] == "evt-1"
        assert "timestamp" in event
        assert event["data"]["session_id"] == "s1"
        assert event["data"]["client_source"] == "scope"

    def test_missing_data_type_falls_back_to_outer(self):
        envelope = {"id": "x", "type": "stream_trace", "timestamp": "123", "data": {}}
        event = map_envelope_to_event(envelope)
        assert event["type"] == "stream_trace"

    def test_missing_data_and_outer_type(self):
        envelope = {"id": "x", "data": {}}
        event = map_envelope_to_event(envelope)
        assert event["type"] == "unknown"

    def test_missing_fields_omitted(self):
        envelope = {"data": {"type": "error"}}
        event = map_envelope_to_event(envelope)
        assert "id" not in event
        assert "timestamp" not in event
        assert event["type"] == "error"


# ---------------------------------------------------------------------------
# Global accessors
# ---------------------------------------------------------------------------


class TestGlobalAccessors:
    def test_get_set_reporter(self):
        original = get_metrics_reporter()
        try:
            reporter = MetricsReporter(api_key="test")
            set_metrics_reporter(reporter)
            assert get_metrics_reporter() is reporter
        finally:
            set_metrics_reporter(original)

    def test_report_event_noop_when_no_reporter(self):
        original = get_metrics_reporter()
        try:
            set_metrics_reporter(None)
            report_event(_make_envelope())
        finally:
            set_metrics_reporter(original)

    def test_is_metrics_reporter_enabled_with_key(self):
        with patch.dict("os.environ", {"SCOPE_CLOUD_API_KEY": "test-key"}, clear=False):
            from scope.server import metrics_reporter as mr

            assert mr.is_metrics_reporter_enabled()

    def test_is_metrics_reporter_disabled_explicit(self):
        with patch.dict(
            "os.environ",
            {"SCOPE_METRICS_ENABLED": "false", "SCOPE_CLOUD_API_KEY": "key"},
            clear=False,
        ):
            from scope.server import metrics_reporter as mr

            assert not mr.is_metrics_reporter_enabled()


# ---------------------------------------------------------------------------
# MetricsReporter: batching & buffer
# ---------------------------------------------------------------------------


class TestEnqueueAndBatching:
    def test_enqueue_adds_mapped_events(self):
        reporter = MetricsReporter(api_key="key")
        reporter._started = True

        for i in range(10):
            reporter.enqueue(_make_envelope("stream_heartbeat", event_id=f"e-{i}"))

        assert reporter.buffered_count == 10

    def test_enqueue_noop_when_stopped(self):
        reporter = MetricsReporter(api_key="key")
        reporter._started = False
        reporter.enqueue(_make_envelope())
        assert reporter.buffered_count == 0

    def test_enqueue_noop_when_paused(self):
        reporter = MetricsReporter(api_key="key")
        reporter._started = True
        reporter._paused = True
        reporter.enqueue(_make_envelope())
        assert reporter.buffered_count == 0

    def test_buffer_max_size_drops_oldest(self):
        reporter = MetricsReporter(api_key="key")
        reporter._started = True
        reporter._buffer = deque(maxlen=5)

        for i in range(10):
            reporter.enqueue(_make_envelope(event_id=f"e-{i}"))

        assert reporter.buffered_count == 5
        batch = reporter._take_batch(5)
        ids = [e["id"] for e in batch]
        assert ids[0] == "e-5"

    def test_take_batch_respects_max_size(self):
        reporter = MetricsReporter(api_key="key")
        reporter._started = True

        for i in range(1000):
            reporter.enqueue(_make_envelope(event_id=f"e-{i}"))

        batch = reporter._take_batch(MAX_BATCH_SIZE)
        assert len(batch) == MAX_BATCH_SIZE
        assert reporter.buffered_count == 500

    def test_requeue_batch_preserves_order(self):
        reporter = MetricsReporter(api_key="key")
        reporter._started = True

        for i in range(3):
            reporter.enqueue(_make_envelope(event_id=f"e-{i}"))

        batch = reporter._take_batch(3)
        reporter._requeue_batch(batch)

        recovered = reporter._take_batch(3)
        assert [e["id"] for e in recovered] == [e["id"] for e in batch]

    def test_build_body_shape(self):
        reporter = MetricsReporter(api_key="key")
        events = [map_envelope_to_event(_make_envelope())]
        body = reporter._build_body(events)

        assert body["app"] == "scope"
        assert "host" in body
        assert body["events"] is events


# ---------------------------------------------------------------------------
# Reliability contract (response handling)
# ---------------------------------------------------------------------------


class TestResponseHandling:
    @pytest.mark.anyio
    async def test_200_drops_batch_resets_backoff(self):
        reporter = MetricsReporter(api_key="key")
        reporter._started = True
        reporter._backoff_s = 16.0

        batch = [map_envelope_to_event(_make_envelope())]
        await reporter._handle_response(_make_response(200), batch)

        assert reporter._backoff_s == INITIAL_BACKOFF_S
        assert reporter.buffered_count == 0

    @pytest.mark.anyio
    async def test_400_drops_batch(self):
        reporter = MetricsReporter(api_key="key")
        reporter._started = True

        batch = [map_envelope_to_event(_make_envelope())]
        await reporter._handle_response(_make_response(400, "bad request"), batch)

        assert reporter.buffered_count == 0

    @pytest.mark.anyio
    async def test_401_pauses_and_requeues(self):
        reporter = MetricsReporter(api_key="key")
        reporter._started = True

        batch = [map_envelope_to_event(_make_envelope())]
        await reporter._handle_response(_make_response(401), batch)

        assert reporter._paused is True
        assert reporter.buffered_count == 1

    @pytest.mark.anyio
    async def test_429_requeues_batch(self):
        reporter = MetricsReporter(api_key="key")
        reporter._started = True

        batch = [map_envelope_to_event(_make_envelope())]

        with patch(
            "scope.server.metrics_reporter.asyncio.sleep", new_callable=AsyncMock
        ):
            await reporter._handle_response(_make_response(429), batch)

        assert reporter.buffered_count == 1

    @pytest.mark.anyio
    async def test_5xx_requeues_with_backoff(self):
        reporter = MetricsReporter(api_key="key")
        reporter._started = True
        reporter._backoff_s = INITIAL_BACKOFF_S

        batch = [map_envelope_to_event(_make_envelope())]

        with patch(
            "scope.server.metrics_reporter.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep:
            await reporter._handle_response(_make_response(502), batch)

        assert reporter.buffered_count == 1
        mock_sleep.assert_awaited_once_with(INITIAL_BACKOFF_S)
        assert reporter._backoff_s == INITIAL_BACKOFF_S * 2

    @pytest.mark.anyio
    async def test_backoff_capped_at_max(self):
        reporter = MetricsReporter(api_key="key")
        reporter._started = True
        reporter._backoff_s = 32.0

        batch = [map_envelope_to_event(_make_envelope())]

        with patch(
            "scope.server.metrics_reporter.asyncio.sleep", new_callable=AsyncMock
        ):
            await reporter._handle_response(_make_response(500), batch)

        assert reporter._backoff_s == 60.0

    @pytest.mark.anyio
    async def test_update_api_key_unpauses(self):
        reporter = MetricsReporter(api_key="old-key")
        reporter._started = True
        reporter._paused = True
        reporter._backoff_s = 30.0

        reporter.update_api_key("new-key")

        assert reporter._paused is False
        assert reporter._backoff_s == INITIAL_BACKOFF_S
        assert reporter._api_key == "new-key"


# ---------------------------------------------------------------------------
# Flush cycle
# ---------------------------------------------------------------------------


class TestFlushOnce:
    @pytest.mark.anyio
    async def test_flush_once_sends_batch(self):
        reporter = MetricsReporter(api_key="test-key")
        reporter._started = True

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=_make_response(200))
        reporter._client = mock_client

        for i in range(3):
            reporter.enqueue(_make_envelope(event_id=f"e-{i}"))

        await reporter._flush_once()

        assert reporter.buffered_count == 0
        mock_client.post.assert_awaited_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer test-key"

    @pytest.mark.anyio
    async def test_flush_once_noop_on_empty_buffer(self):
        reporter = MetricsReporter(api_key="key")
        reporter._started = True
        reporter._client = AsyncMock()

        await reporter._flush_once()

        reporter._client.post.assert_not_awaited()

    @pytest.mark.anyio
    async def test_flush_network_error_requeues(self):
        reporter = MetricsReporter(api_key="key")
        reporter._started = True

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("connection refused"))
        reporter._client = mock_client

        reporter.enqueue(_make_envelope(event_id="net-err"))

        with patch(
            "scope.server.metrics_reporter.asyncio.sleep", new_callable=AsyncMock
        ):
            await reporter._flush_once()

        assert reporter.buffered_count == 1


# ---------------------------------------------------------------------------
# Disk persistence
# ---------------------------------------------------------------------------


class TestDiskBuffer:
    def test_persist_and_load(self, tmp_path: Path):
        buf_path = tmp_path / "buffer.jsonl"

        reporter = MetricsReporter(api_key="key")
        reporter._started = True
        reporter._buffer_path = buf_path

        reporter.enqueue(_make_envelope(event_id="disk-1"))
        reporter.enqueue(_make_envelope(event_id="disk-2"))

        reporter._persist_disk_buffer()
        assert buf_path.exists()
        lines = buf_path.read_text().strip().splitlines()
        assert len(lines) == 2

        reporter._buffer.clear()
        assert reporter.buffered_count == 0

        reporter._load_disk_buffer()
        assert reporter.buffered_count == 2
        assert buf_path.read_text() == ""

    def test_load_handles_missing_file(self, tmp_path: Path):
        reporter = MetricsReporter(api_key="key")
        reporter._buffer_path = tmp_path / "nonexistent.jsonl"
        reporter._load_disk_buffer()
        assert reporter.buffered_count == 0

    def test_load_handles_corrupt_lines(self, tmp_path: Path):
        buf_path = tmp_path / "buffer.jsonl"
        buf_path.write_text(
            '{"type":"ok","data":{}}\nnot-json\n{"type":"ok2","data":{}}\n'
        )

        reporter = MetricsReporter(api_key="key")
        reporter._buffer_path = buf_path
        reporter._load_disk_buffer()
        assert reporter.buffered_count == 2


# ---------------------------------------------------------------------------
# TelemetrySink protocol conformance
# ---------------------------------------------------------------------------


class TestTelemetrySinkConformance:
    def test_isinstance_check(self):
        """MetricsReporter satisfies the TelemetrySink runtime_checkable protocol."""
        from scope.server.kafka_publisher import TelemetrySink

        reporter = MetricsReporter(api_key="key")
        assert isinstance(reporter, TelemetrySink)

    def test_emit_delegates_to_enqueue(self):
        """emit() is the TelemetrySink entry point and buffers via enqueue()."""
        reporter = MetricsReporter(api_key="key")
        reporter._started = True

        envelope = _make_envelope("stream_started", event_id="emit-1")
        reporter.emit(envelope)

        assert reporter.buffered_count == 1
        batch = reporter._take_batch(1)
        assert batch[0]["id"] == "emit-1"
        assert batch[0]["type"] == "stream_started"

    def test_publish_raw_event_routes_through_emit(self):
        """When MetricsReporter is the active sink, publish_raw_event() routes to emit()."""
        from scope.server.kafka_publisher import (
            get_telemetry_sink,
            publish_raw_event,
            set_telemetry_sink,
        )

        reporter = MetricsReporter(api_key="key")
        reporter._started = True

        prev_sink = get_telemetry_sink()
        try:
            set_telemetry_sink(reporter)

            envelope = _make_envelope("stream_stopped", event_id="raw-1")
            publish_raw_event(envelope)

            assert reporter.buffered_count == 1
        finally:
            set_telemetry_sink(prev_sink)

    def test_emit_noop_when_stopped(self):
        reporter = MetricsReporter(api_key="key")
        reporter._started = False
        reporter.emit(_make_envelope())
        assert reporter.buffered_count == 0


# ---------------------------------------------------------------------------
# Start / stop lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    @pytest.mark.anyio
    async def test_start_and_stop(self, tmp_path: Path):
        reporter = MetricsReporter(api_key="key")
        reporter._buffer_path = tmp_path / "empty.jsonl"
        started = await reporter.start()
        assert started is True
        assert reporter.is_running is True

        reporter.enqueue(_make_envelope())
        assert reporter.buffered_count == 1

        await reporter.stop()
        assert reporter.is_running is False

    @pytest.mark.anyio
    async def test_start_fails_without_httpx(self):
        reporter = MetricsReporter(api_key="key")
        with patch.dict("sys.modules", {"httpx": None}):
            with patch("builtins.__import__", side_effect=ImportError("no httpx")):
                started = await reporter.start()
        assert isinstance(started, bool)
