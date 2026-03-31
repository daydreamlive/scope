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
RUNNER_STARTUP_TIMEOUT_SECONDS = 600
RUNNER_RETRY_DELAY_SECONDS = 2.5
RUNNER_MAX_FAILURES_PER_WINDOW = 20
RUNNER_FAILURE_WINDOW_SECONDS = 60.0
ASSETS_DIR_PATH = "/tmp/.daydream-scope/assets"

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
# Pre-install livepeer extras at image build time so the runner can start
# immediately on cold workers without downloading ~1.4GB of CUDA packages.
RUN uv sync --extra livepeer --frozen
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
        "livepeer-runner",
        "--host",
        RUNNER_HOST,
        "--port",
        str(RUNNER_PORT),
    ]


async def _proxy_ws(client_ws: WebSocket) -> None:
    """Connect to the local runner and proxy traffic bidirectionally.

    Raises WebSocketDisconnect if the client disconnects.
    Returns normally if the runner connection drops.
    """
    import websockets
    from websockets.exceptions import ConnectionClosed

    async with websockets.connect(RUNNER_LOCAL_WS_URL) as runner_ws:

        async def client_to_runner() -> None:
            while True:
                message = await client_ws.receive()
                msg_type = message.get("type")
                if msg_type == "websocket.receive":
                    text_data = message.get("text")
                    bytes_data = message.get("bytes")
                    if text_data is not None:
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
            "LIVEPEER_DEBUG",
            "UV_CACHE_DIR",
        ]
        runner_env = {k: os.environ[k] for k in env_allowlist if k in os.environ}
        runner_env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
        runner_env.setdefault("DAYDREAM_SCOPE_MODELS_DIR", "/data/models")
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

        # Ensure any previous session data is cleaned up
        event = _get_cleanup_event()
        await event.wait()
        event.clear()

        failure_timestamps: list[float] = []

        try:
            while True:
                print(f"Connecting proxy to runner websocket at {RUNNER_LOCAL_WS_URL}")
                try:
                    await _proxy_ws(client_ws)
                except (
                    ConnectionClosed,
                    InvalidStatus,
                    InvalidHandshake,
                    OSError,
                ) as exc:
                    print(f"Livepeer fal ws runner connection failed: {exc}")

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
            await run_cleanup()
            with suppress(Exception):
                await client_ws.close()
            print("Livepeer fal ws client disconnected")
