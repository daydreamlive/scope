"""fal.ai deployment wrapper for the Livepeer runner.

This wraps the Livepeer runner implementation in
`scope.cloud.livepeer_app` unchanged and provides:
- fal container/image configuration
- runner subprocess lifecycle management
- WebSocket proxying between fal `/ws` and local runner `/ws`
"""

import asyncio
import importlib.util
import os
import subprocess as _subprocess
import time
from contextlib import suppress
from pathlib import Path

import fal
from fal.container import ContainerImage
from fastapi import WebSocket, WebSocketDisconnect

RUNNER_HOST = "127.0.0.1"
RUNNER_BIND_HOST = "0.0.0.0"
RUNNER_PORT = int(os.getenv("LIVEPEER_RUNNER_PORT", "8001"))
RUNNER_LOCAL_WS_URL = f"ws://{RUNNER_HOST}:{RUNNER_PORT}/ws"
RUNNER_LOCAL_HTTP_URL = f"http://{RUNNER_HOST}:{RUNNER_PORT}"
RUNNER_STARTUP_TIMEOUT_SECONDS = 90
SCOPE_IMPORT_ROOT_CANDIDATES = (
    Path("/app/src"),
    Path.cwd() / "src",
    Path(__file__).resolve().parents[2],
)
RUNNER_APP_FILE_CANDIDATES = (
    Path("/app/src/scope/cloud/livepeer_app.py"),
    Path.cwd() / "src/scope/cloud/livepeer_app.py",
    Path(__file__).resolve().with_name("livepeer_app.py"),
)


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
    """Build a runner startup command for source and installed layouts."""
    for runner_app_file in RUNNER_APP_FILE_CANDIDATES:
        if runner_app_file.exists():
            return [
                "uv",
                "run",
                "--extra",
                "livepeer",
                "python",
                str(runner_app_file),
                "--host",
                RUNNER_BIND_HOST,
                "--port",
                str(RUNNER_PORT),
            ]

    module_spec = None
    try:
        module_spec = importlib.util.find_spec("scope.cloud.livepeer_app")
    except ModuleNotFoundError:
        module_spec = None

    if module_spec is not None:
        return [
            "uv",
            "run",
            "--extra",
            "livepeer",
            "python",
            "-m",
            "scope.cloud.livepeer_app",
            "--host",
            RUNNER_BIND_HOST,
            "--port",
            str(RUNNER_PORT),
        ]

    deploy_tag = os.environ.get("SCOPE_DEPLOY_TAG", "<unset>")
    checked_paths = ", ".join(str(path) for path in RUNNER_APP_FILE_CANDIDATES)
    raise RuntimeError(
        "Could not locate Livepeer runner entrypoint in this container image. "
        "Neither module `scope.cloud.livepeer_app` nor any runner file is available. "
        f"Checked paths: {checked_paths}. "
        f"SCOPE_DEPLOY_TAG={deploy_tag}. "
        "Build and use a newer cloud image tag that includes "
        "`src/scope/cloud/livepeer_app.py`."
    )


class LivepeerScopeApp(fal.App, keep_alive=300):
    """fal entrypoint that runs and proxies the existing Livepeer Scope runner."""

    image = custom_image
    machine_type = "GPU-H100"
    app_auth = "public"
    requirements = [
        "requests",
        "websockets",
    ]

    def setup(self):
        """Start `scope.cloud.livepeer_app` as a background subprocess."""
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

        env_whitelist = [
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
        runner_env = {k: os.environ[k] for k in env_whitelist if k in os.environ}
        runner_env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
        runner_env.setdefault("DAYDREAM_SCOPE_MODELS_DIR", "/data/models")
        runner_env.setdefault("PYTHONUNBUFFERED", "1")
        pythonpath_parts = [
            str(path) for path in SCOPE_IMPORT_ROOT_CANDIDATES if path.exists()
        ]
        existing_pythonpath = runner_env.get("PYTHONPATH")
        if existing_pythonpath:
            pythonpath_parts.append(existing_pythonpath)
        if pythonpath_parts:
            runner_env["PYTHONPATH"] = ":".join(pythonpath_parts)

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

    async def _proxy_ws(self, client_ws: WebSocket) -> None:
        """Proxy fal WebSocket traffic to the local Livepeer runner WebSocket."""
        print("Livepeer fal websocket_handler invoked for /ws")

        import websockets
        from websockets.exceptions import (
            ConnectionClosed,
            InvalidHandshake,
            InvalidStatus,
        )

        await client_ws.accept()
        print(f"Connecting proxy to runner websocket at {RUNNER_LOCAL_WS_URL}")

        try:
            async with websockets.connect(RUNNER_LOCAL_WS_URL) as runner_ws:

                async def client_to_runner():
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
                            break

                async def runner_to_client():
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
                for task in done:
                    with suppress(asyncio.CancelledError, WebSocketDisconnect):
                        await task
        except (WebSocketDisconnect, ConnectionClosed):
            pass
        except InvalidStatus as exc:
            print(f"Livepeer fal ws handshake rejected by runner: {exc}")
        except InvalidHandshake as exc:
            print(f"Livepeer fal ws handshake failed before upgrade: {exc}")
        except Exception as exc:
            print(f"Livepeer fal ws proxy error: {type(exc).__name__}: {exc}")
        finally:
            with suppress(Exception):
                await client_ws.close()
            print("Livepeer fal ws client disconnected")

    @fal.endpoint("/ws", is_websocket=True)
    async def websocket_handler(self, client_ws: WebSocket):
        """WebSocket endpoint for Livepeer signaling and control traffic."""
        await self._proxy_ws(client_ws)
