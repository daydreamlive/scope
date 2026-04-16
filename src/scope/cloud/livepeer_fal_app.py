"""fal.ai deployment wrapper for the Livepeer runner.

This runs the Livepeer runner (``livepeer-runner`` / ``scope.cloud.livepeer_app``)
as a subprocess and provides:
- fal container/image configuration
- runner subprocess lifecycle management
- WebSocket proxying between fal `/ws` and local runner `/ws`
"""

import asyncio
import os
import subprocess as _subprocess
import time
from contextlib import suppress
from pathlib import Path

import fal
from fal.container import ContainerImage
from fastapi import WebSocket, WebSocketDisconnect

RUNNER_HOST = "127.0.0.1"
RUNNER_PORT = int(os.getenv("LIVEPEER_RUNNER_PORT", "8001"))
RUNNER_LOCAL_WS_URL = f"ws://{RUNNER_HOST}:{RUNNER_PORT}/ws"
RUNNER_LOCAL_HTTP_URL = f"http://{RUNNER_HOST}:{RUNNER_PORT}"
RUNNER_STARTUP_TIMEOUT_SECONDS = 90
RUNNER_RETRY_DELAY_SECONDS = 2.5
RUNNER_MAX_FAILURES_PER_WINDOW = 20
RUNNER_FAILURE_WINDOW_SECONDS = 60.0
ASSETS_DIR_PATH = "/tmp/.daydream-scope/assets"


# ---------------------------------------------------------------------------
# Kafka publisher — matches fal_app.py KafkaPublisher for event parity
# ---------------------------------------------------------------------------

kafka_publisher: "KafkaPublisher | None" = None


class KafkaPublisher:
    """Async Kafka event publisher for fal.ai websocket events."""

    def __init__(self):
        self._producer = None
        self._started = False
        self._topic = None

    async def start(self) -> bool:
        """Start the Kafka producer."""
        import json as _json  # noqa: F811

        bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
        self._topic = os.getenv("KAFKA_TOPIC", "network_events")
        sasl_username = os.getenv("KAFKA_SASL_USERNAME")
        sasl_password = os.getenv("KAFKA_SASL_PASSWORD")

        print(
            f"[Kafka] Starting publisher (KAFKA_BOOTSTRAP_SERVERS={bootstrap_servers})"
        )
        if not bootstrap_servers:
            print("[Kafka] Not configured, event publishing disabled")
            return False

        try:
            from aiokafka import AIOKafkaProducer

            config = {
                "bootstrap_servers": bootstrap_servers,
                "value_serializer": lambda v: _json.dumps(v).encode("utf-8"),
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
            print(f"[Kafka] Publisher started, topic: {self._topic}")
            return True

        except ImportError:
            print("[Kafka] aiokafka not installed, Kafka disabled")
            return False
        except Exception as e:
            print(f"[Kafka] Failed to start producer: {e}")
            return False

    async def stop(self):
        """Stop the Kafka producer."""
        if self._producer and self._started:
            try:
                await self._producer.stop()
                print("[Kafka] Publisher stopped")
            except Exception as e:
                print(f"[Kafka] Error stopping producer: {e}")
            finally:
                self._started = False
                self._producer = None

    async def publish(self, event_type: str, data: dict) -> bool:
        """Publish an event to Kafka."""
        import uuid as _uuid

        if not self._started or not self._producer:
            return False

        event_id = str(_uuid.uuid4())
        timestamp_ms = str(int(time.time() * 1000))

        event = {
            "id": event_id,
            "type": "stream_trace",
            "timestamp": timestamp_ms,
            "data": {
                "type": event_type,
                "client_source": "scope",
                "timestamp": timestamp_ms,
                **data,
            },
        }

        try:
            await self._producer.send_and_wait(self._topic, value=event, key=event_id)
            print(f"[Kafka] Published event: {event_type}")
            return True
        except Exception as e:
            print(f"[Kafka] Failed to publish event {event_type}: {e}")
            return False

    @property
    def is_running(self) -> bool:
        return self._started


# Gates startup cleanup so only one cleanup run executes at a time.
_cleanup_event: asyncio.Event | None = None


def _get_cleanup_event() -> asyncio.Event:
    global _cleanup_event
    if _cleanup_event is None:
        _cleanup_event = asyncio.Event()
        _cleanup_event.set()
    return _cleanup_event


async def cleanup_runner_session() -> None:
    """Request full session cleanup from the local runner endpoint."""
    import httpx

    cleanup_url = f"{RUNNER_LOCAL_HTTP_URL}/internal/cleanup-session"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(cleanup_url, timeout=180.0)
            if response.status_code != 200:
                print(
                    "Warning: Runner cleanup endpoint failed: "
                    f"{response.status_code} {response.text[:200]}"
                )
                return

            payload = response.json()
            if not payload.get("ok", False):
                print(f"Warning: Runner cleanup completed with issues: {payload}")
            else:
                print("Runner cleanup completed successfully")
    except Exception as exc:
        print(f"Warning: Runner cleanup request failed: {exc}")


async def run_cleanup() -> None:
    """Run full cleanup and release waiting websocket sessions."""
    event = _get_cleanup_event()
    event.clear()
    try:
        await cleanup_runner_session()
    finally:
        event.set()


def _get_git_sha() -> str:
    """Get deploy tag from env var SCOPE_DEPLOY_TAG or derive from git SHA."""
    deploy_tag = os.environ.get("SCOPE_DEPLOY_TAG")
    if deploy_tag:
        if deploy_tag.endswith("-cloud"):
            return deploy_tag
        normalized_tag = f"{deploy_tag}-cloud"
        print(
            "SCOPE_DEPLOY_TAG did not include '-cloud' suffix; "
            f"using cloud image tag: {normalized_tag}"
        )
        return normalized_tag

    try:
        result = _subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        tag = result.stdout.strip()[:7] + "-cloud"
        print(f"Deploying with tag: {tag}")
        return tag
    except Exception as exc:
        print(f"Warning: could not get git SHA: {exc}")
        return "unknown"


GIT_SHA = _get_git_sha()
DOCKER_IMAGE = f"daydreamlive/scope:{GIT_SHA}"
dockerfile_str = f"""
FROM {DOCKER_IMAGE}
WORKDIR /app
COPY pyproject.toml uv.lock README.md patches.pth /app/
COPY src/ /app/src/
"""
custom_image = ContainerImage.from_dockerfile_str(
    dockerfile_str,
    context_dir=Path(__file__).resolve().parents[3],
    dockerignore=[
        "frontend",
        "docs",
        "tests",
        "app",
        "**/__pycache__",
        "*.pyc",
        "**/*.pyc",
        "*.swp",
        "**/*.swp",
        "*.swo",
        "**/*.swo",
    ],
)


def _runner_is_ready() -> bool:
    """Return True when the local runner HTTP server responds."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{RUNNER_LOCAL_HTTP_URL}/docs", timeout=2):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _build_runner_command() -> list[str]:
    """Build the runner startup command using the package script entrypoint."""
    return [
        "uv",
        "run",
        "--extra",
        "livepeer",
        "--extra",
        "kafka",
        "livepeer-runner",
        "--host",
        RUNNER_HOST,
        "--port",
        str(RUNNER_PORT),
    ]


async def _handle_first_client_text(text_data: str, metadata: dict | None) -> None:
    """Parse the first client text message, print manifest_id, publish connected.

    Best-effort — any parsing or publishing failure is logged with the
    ``[KAFKA-DEBUG]`` prefix but never raises, so the proxy loop is unaffected.
    """
    import json

    if metadata is None:
        return

    try:
        parsed = json.loads(text_data)
    except (json.JSONDecodeError, TypeError) as exc:
        print(f"[KAFKA-DEBUG] First client text not JSON: {exc}")
        return

    manifest_id = parsed.get("manifest_id") if isinstance(parsed, dict) else None
    params = parsed.get("params") if isinstance(parsed, dict) else None
    user_id = params.get("daydream_user_id") if isinstance(params, dict) else None

    metadata["manifest_id"] = manifest_id
    metadata["user_id"] = user_id

    print(
        f"[KAFKA-DEBUG] First client message parsed manifest_id={manifest_id} "
        f"user_id={user_id}"
    )

    if kafka_publisher is None or not kafka_publisher.is_running:
        print("[KAFKA-DEBUG] Skipping websocket_connected: Kafka not running")
        return

    connection_info = metadata.get("connection_info") or {}
    ok = await kafka_publisher.publish(
        "websocket_connected",
        {
            "user_id": user_id,
            "connection_id": manifest_id,
            "connection_info": connection_info,
        },
    )
    print(f"[KAFKA-DEBUG] websocket_connected publish result: ok={ok}")


async def _proxy_ws(client_ws: WebSocket, metadata: dict | None = None) -> None:
    """Connect to the local runner and proxy traffic bidirectionally.

    If *metadata* is provided, the first client→runner text message is
    parsed (best-effort) to extract ``manifest_id`` and ``user_id`` and a
    ``websocket_connected`` Kafka event is published inline.

    Raises WebSocketDisconnect if the client disconnects.
    Returns normally if the runner connection drops.
    """

    import websockets
    from websockets.exceptions import ConnectionClosed

    # NOTE: Previously we sniffed the two-message handshake explicitly
    # (recv runner ready, then receive client job_info) before starting
    # the parallel proxy loop. That is disabled here for debugging —
    # keep the proxy transparent and sniff inline in client_to_runner.
    #
    # async with websockets.connect(RUNNER_LOCAL_WS_URL) as runner_ws:
    #     ready_msg = await runner_ws.recv()
    #     ...
    #     job_msg = await client_ws.receive()
    #     ...

    async with websockets.connect(RUNNER_LOCAL_WS_URL) as runner_ws:
        first_client_text_seen = False

        async def client_to_runner() -> None:
            nonlocal first_client_text_seen
            while True:
                message = await client_ws.receive()
                msg_type = message.get("type")
                if msg_type == "websocket.receive":
                    text_data = message.get("text")
                    bytes_data = message.get("bytes")
                    if text_data is not None:
                        if not first_client_text_seen:
                            first_client_text_seen = True
                            await _handle_first_client_text(text_data, metadata)
                        await runner_ws.send(text_data)
                    elif bytes_data is not None:
                        await runner_ws.send(bytes_data)
                elif msg_type == "websocket.disconnect":
                    raise WebSocketDisconnect()

        async def runner_to_client() -> None:
            while True:
                message = await runner_ws.recv()
                if isinstance(message, bytes):
                    await client_ws.send_bytes(message)
                else:
                    await client_ws.send_text(message)

        c2r = asyncio.create_task(client_to_runner())
        r2c = asyncio.create_task(runner_to_client())
        done, pending = await asyncio.wait(
            {c2r, r2c},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

        # WebSocketDisconnect is used as a signal to tell the caller
        # the client is gone (normal shutdown) so prioritize that.
        # Otherwise, re-raise for other types of unexpected errors.
        disconnect_exc: WebSocketDisconnect | None = None
        unexpected_exc: Exception | None = None
        for task in (*done, *pending):
            try:
                await task
            except (asyncio.CancelledError, ConnectionClosed):
                pass
            except WebSocketDisconnect as exc:
                disconnect_exc = disconnect_exc or exc
            except Exception as exc:
                unexpected_exc = unexpected_exc or exc

        if disconnect_exc is not None:
            raise disconnect_exc
        if unexpected_exc is not None:
            raise unexpected_exc


class LivepeerScopeApp(fal.App, keep_alive=300):
    """fal entrypoint that runs and proxies the existing Livepeer Scope runner."""

    image = custom_image
    machine_type = "GPU-H100"
    requirements = [
        "websockets",
        "httpx",
        "aiokafka",
    ]

    def setup(self):
        """Start the Livepeer runner as a background subprocess."""
        import subprocess

        print(f"Starting Livepeer runner wrapper setup... (version: {GIT_SHA})")

        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                check=True,
            )
            print(f"GPU Status:\n{result.stdout}")
        except Exception as exc:
            print(f"GPU check failed: {exc}")
            raise

        env_allowlist = [
            "PATH",
            "HOME",
            "USER",
            "LANG",
            "LC_ALL",
            "PYTHONPATH",
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "NVIDIA_DRIVER_CAPABILITIES",
            "LD_LIBRARY_PATH",
            "DAYDREAM_API_BASE",
            "HF_TOKEN",
            "HF_HOME",
            "HUGGINGFACE_HUB_CACHE",
            "DAYDREAM_SCOPE_BUNDLED_PLUGINS_FILE",
            "LIVEPEER_DEBUG",
            "UV_CACHE_DIR",
            # Kafka (for scope.server.kafka_publisher in the runner subprocess)
            "KAFKA_BOOTSTRAP_SERVERS",
            "KAFKA_TOPIC",
            "KAFKA_SASL_USERNAME",
            "KAFKA_SASL_PASSWORD",
        ]
        runner_env = {k: os.environ[k] for k in env_allowlist if k in os.environ}
        runner_env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
        runner_env.setdefault("DAYDREAM_SCOPE_MODELS_DIR", "/data/models")
        runner_env.setdefault("DAYDREAM_SCOPE_LORA_SHARED_DIR", "/data/models/lora")
        runner_env.setdefault("DAYDREAM_SCOPE_ASSETS_DIR", ASSETS_DIR_PATH)
        runner_env.setdefault("DAYDREAM_SCOPE_LORA_DIR", ASSETS_DIR_PATH + "/lora")
        runner_env.setdefault("DAYDREAM_SCOPE_LOGS_DIR", ASSETS_DIR_PATH + "/logs")
        runner_env.setdefault(
            "DAYDREAM_SCOPE_PLUGINS_DIR", ASSETS_DIR_PATH + "/plugins"
        )
        runner_env.setdefault("PYTHONUNBUFFERED", "1")

        runner_cmd = _build_runner_command()
        print(f"Starting Livepeer runner with command: {' '.join(runner_cmd)}")

        process = subprocess.Popen(
            runner_cmd,
            env=runner_env,
        )
        self.runner_process = process

        start = time.time()
        while time.time() - start < RUNNER_STARTUP_TIMEOUT_SECONDS:
            if process.poll() is not None:
                raise RuntimeError(
                    "Livepeer runner process exited during startup "
                    f"(code={process.returncode})"
                )
            if _runner_is_ready():
                print(f"Livepeer runner ready at {RUNNER_LOCAL_WS_URL}")
                return
            time.sleep(1)

        raise RuntimeError(
            f"Timed out waiting for Livepeer runner on {RUNNER_LOCAL_HTTP_URL}"
        )

    @fal.endpoint("/ws", is_websocket=True)
    async def websocket_handler(self, client_ws: WebSocket) -> None:
        """WebSocket endpoint for Livepeer signaling and control traffic."""
        print("Livepeer fal websocket_handler invoked for /ws")

        from websockets.exceptions import (
            ConnectionClosed,
            InvalidHandshake,
            InvalidStatus,
        )

        await client_ws.accept()

        # Initialize Kafka publisher (lazy, once per process)
        global kafka_publisher
        if kafka_publisher is None:
            print("[KAFKA-DEBUG] kafka_publisher is None, initializing")
            kafka_publisher = KafkaPublisher()
            started = await kafka_publisher.start()
            print(f"[KAFKA-DEBUG] KafkaPublisher.start() returned: {started}")
        else:
            print(
                f"[KAFKA-DEBUG] kafka_publisher already initialized "
                f"(is_running={kafka_publisher.is_running})"
            )

        # Smoke-test publish: verify basic Kafka write path works end-to-end
        # before any real events. Filter logs on "[KAFKA-DEBUG]" to check.
        if kafka_publisher is not None and kafka_publisher.is_running:
            print("[KAFKA-DEBUG] Attempting smoke-test publish (kafka_smoke_test)")
            smoke_ok = await kafka_publisher.publish(
                "kafka_smoke_test",
                {"note": "livepeer_fal_app websocket_handler startup ping"},
            )
            print(f"[KAFKA-DEBUG] Smoke-test publish result: ok={smoke_ok}")
        else:
            print("[KAFKA-DEBUG] Skipping smoke-test publish: Kafka not running")

        connection_start_time = time.time()
        metadata: dict = {}

        import json

        fal_log_labels_raw = os.getenv("FAL_LOG_LABELS", "unknown")
        try:
            fal_log_labels = json.loads(fal_log_labels_raw)
        except (json.JSONDecodeError, TypeError):
            fal_log_labels = fal_log_labels_raw

        connection_info = {
            "gpu_type": LivepeerScopeApp.machine_type,
            "fal_region": os.getenv("NOMAD_DC", "unknown"),
            "fal_runner_id": os.getenv(
                "FAL_JOB_ID", os.getenv("FAL_RUNNER_ID", "unknown")
            ),
            "fal_log_labels": fal_log_labels,
        }
        metadata["connection_info"] = connection_info

        # Ensure any previous session data is cleaned up
        event = _get_cleanup_event()
        await event.wait()
        event.clear()

        failure_timestamps: list[float] = []

        try:
            while True:
                print(f"Connecting proxy to runner websocket at {RUNNER_LOCAL_WS_URL}")
                try:
                    await _proxy_ws(client_ws, metadata=metadata)
                except (
                    ConnectionClosed,
                    InvalidStatus,
                    InvalidHandshake,
                    OSError,
                ) as exc:
                    print(f"Livepeer fal ws runner connection failed: {exc}")

                # websocket_connected is published inline from
                # _handle_first_client_text the first time the client sends a
                # text message through the proxy — no post-proxy publish here.

                now = time.monotonic()
                cutoff = now - RUNNER_FAILURE_WINDOW_SECONDS
                failure_timestamps.append(now)
                failure_timestamps = [t for t in failure_timestamps if t > cutoff]
                if len(failure_timestamps) > RUNNER_MAX_FAILURES_PER_WINDOW:
                    print(
                        "Livepeer fal ws proxy: too many runner failures in rolling window; "
                        "closing outer websocket"
                    )
                    break

                print(
                    f"Runner websocket disconnected, retrying in "
                    f"{RUNNER_RETRY_DELAY_SECONDS * 1000:.0f}ms..."
                )
                await asyncio.sleep(RUNNER_RETRY_DELAY_SECONDS)
        except (WebSocketDisconnect, ConnectionClosed):
            pass
        except Exception as exc:
            print(f"Livepeer fal ws proxy error: {type(exc).__name__}: {exc}")
        finally:
            # Publish websocket_disconnected event
            if kafka_publisher and kafka_publisher.is_running:
                end_time = time.time()
                elapsed_ms = int((end_time - connection_start_time) * 1000)
                print(
                    "[KAFKA-DEBUG] Attempting websocket_disconnected publish "
                    f"(manifest_id={metadata.get('manifest_id')} "
                    f"user_id={metadata.get('user_id')} "
                    f"duration_ms={elapsed_ms})"
                )
                disc_ok = await kafka_publisher.publish(
                    "websocket_disconnected",
                    {
                        "user_id": metadata.get("user_id"),
                        "connection_id": metadata.get("manifest_id"),
                        "connection_info": connection_info,
                        "duration_ms": elapsed_ms,
                        "session_start_time_ms": int(connection_start_time * 1000),
                        "session_end_time_ms": int(end_time * 1000),
                    },
                )
                print(
                    f"[KAFKA-DEBUG] websocket_disconnected publish result: ok={disc_ok}"
                )
            else:
                print(
                    "[KAFKA-DEBUG] Skipping websocket_disconnected: "
                    "Kafka publisher not running"
                )

            await run_cleanup()
            with suppress(Exception):
                await client_ws.close()
            print("Livepeer fal ws client disconnected")
