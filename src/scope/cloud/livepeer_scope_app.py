"""Livepeer live-runner wrapper for Scope.

This app is the outer Livepeer runner registered with the orchestrator. It
accepts the SDK ``/scope`` live-runner request, creates the control/events
trickle channels, then bridges the current Scope websocket runner protocol to
the new live-runner session model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import aiohttp
import click
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from livepeer_gateway.live_runner import (
    LiveRunnerRegistration,
    create_trickle_channels,
    register_runner,
    remove_trickle_channels,
    stop_runner_session,
)

logger = logging.getLogger(__name__)

LIVEPEER_APP_NAME = "live-video-to-video/scope"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8989
DEFAULT_INNER_PORT = 8001
DEFAULT_ORCHESTRATOR_URL = "http://localhost:8935"
DEFAULT_INNER_STARTUP_TIMEOUT_SECONDS = 90.0
INNER_WS_HANDSHAKE_TIMEOUT_SECONDS = 20.0
DEFAULT_RETRY_DELAY_SECONDS = 2.5
DEFAULT_MAX_FAILURES_PER_WINDOW = 20
DEFAULT_FAILURE_WINDOW_SECONDS = 60.0
DEFAULT_PRICE_PER_UNIT = 0
DEFAULT_PIXELS_PER_UNIT = 1
DEFAULT_PRICE_UNIT = "USD"
CHANNEL_MIME_JSONL = "application/jsonl"
DEFAULT_SCOPE_HOME_DIRNAME = ".daydream-scope"
DEFAULT_SESSION_ASSETS_DIR = Path("/tmp/.daydream-scope/assets")

_registration: LiveRunnerRegistration | None = None
_inner_process: subprocess.Popen | None = None
_sessions: dict[str, ScopeBridgeSession] = {}
_cleanup_event: asyncio.Event | None = None


@dataclass(slots=True)
class ScopeRunnerSettings:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    orchestrator_url: str = DEFAULT_ORCHESTRATOR_URL
    orch_secret: str = ""
    runner_url: str = ""
    inner_ws_url: str = f"ws://127.0.0.1:{DEFAULT_INNER_PORT}/ws"
    inner_host: str = DEFAULT_HOST
    inner_port: int = DEFAULT_INNER_PORT
    inner_startup_timeout_seconds: float = DEFAULT_INNER_STARTUP_TIMEOUT_SECONDS
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS
    max_failures_per_window: int = DEFAULT_MAX_FAILURES_PER_WINDOW
    failure_window_seconds: float = DEFAULT_FAILURE_WINDOW_SECONDS
    price_per_unit: int = DEFAULT_PRICE_PER_UNIT
    pixels_per_unit: int = DEFAULT_PIXELS_PER_UNIT
    price_unit: str = DEFAULT_PRICE_UNIT

    def resolved_runner_url(self) -> str:
        if self.runner_url:
            return self.runner_url
        return f"http://{self.host}:{self.port}"


@dataclass(slots=True)
class ScopeBridgeSession:
    session_id: str
    headers: dict[str, str]
    job_info: dict[str, Any]
    settings: ScopeRunnerSettings
    task: asyncio.Task | None = None
    websocket: Any | None = None
    channel_url_to_name: dict[str, str] = field(default_factory=dict)
    stop_requested: bool = False
    cleanup_started: bool = False
    channel_counter: int = 0


settings = ScopeRunnerSettings()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _validate_price_settings(price_per_unit: int, pixels_per_unit: int) -> None:
    if pixels_per_unit <= 0:
        raise ValueError("LIVEPEER_RUNNER_PIXELS_PER_UNIT must be positive")


def _settings_from_env() -> ScopeRunnerSettings:
    port = int(os.getenv("LIVEPEER_RUNNER_PORT", str(DEFAULT_PORT)))
    inner_port = int(os.getenv("LIVEPEER_INNER_PORT", str(DEFAULT_INNER_PORT)))
    host = os.getenv("LIVEPEER_RUNNER_HOST", DEFAULT_HOST)
    runner_url = os.getenv("LIVEPEER_RUNNER_URL", "")
    price_per_unit = _env_int(
        "LIVEPEER_RUNNER_PRICE_PER_UNIT",
        DEFAULT_PRICE_PER_UNIT,
    )
    pixels_per_unit = _env_int(
        "LIVEPEER_RUNNER_PIXELS_PER_UNIT",
        DEFAULT_PIXELS_PER_UNIT,
    )
    _validate_price_settings(price_per_unit, pixels_per_unit)
    inner_ws_url = os.getenv(
        "LIVEPEER_INNER_WS_URL",
        f"ws://127.0.0.1:{inner_port}/ws",
    )
    inner_host, parsed_inner_port = _inner_bind_from_ws_url(inner_ws_url)
    return ScopeRunnerSettings(
        host=host,
        port=port,
        orchestrator_url=os.getenv("LIVEPEER_ORCH_URL", DEFAULT_ORCHESTRATOR_URL),
        orch_secret=os.getenv("LIVEPEER_ORCH_SECRET", ""),
        runner_url=runner_url,
        inner_ws_url=inner_ws_url,
        inner_host=os.getenv("LIVEPEER_INNER_HOST", inner_host),
        inner_port=inner_port
        if "LIVEPEER_INNER_PORT" in os.environ
        else parsed_inner_port,
        price_per_unit=price_per_unit,
        pixels_per_unit=pixels_per_unit,
        price_unit=os.getenv("LIVEPEER_RUNNER_PRICE_UNIT", DEFAULT_PRICE_UNIT),
    )


def _inner_bind_from_ws_url(inner_ws_url: str) -> tuple[str, int]:
    parsed = urlparse(inner_ws_url)
    return parsed.hostname or DEFAULT_HOST, parsed.port or DEFAULT_INNER_PORT


def _header_snapshot(request: Request) -> dict[str, str]:
    return {
        "Livepeer-Runner-Route": request.headers.get("Livepeer-Runner-Route", ""),
        "Livepeer-Session-Id": request.headers.get("Livepeer-Session-Id", ""),
        "Livepeer-Session-Token": request.headers.get("Livepeer-Session-Token", ""),
        "Livepeer-Session-Control": request.headers.get("Livepeer-Session-Control", ""),
    }


def _session_id(headers: dict[str, str]) -> str:
    session_id = headers.get("Livepeer-Session-Id", "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="missing Livepeer-Session-Id")
    return session_id


async def _register_runner(
    runner_settings: ScopeRunnerSettings,
) -> LiveRunnerRegistration:
    if not runner_settings.orch_secret:
        raise RuntimeError("Livepeer orchestrator secret is required")
    return await register_runner(
        runner_settings.orchestrator_url,
        secret=runner_settings.orch_secret,
        runner_url=runner_settings.resolved_runner_url(),
        app=LIVEPEER_APP_NAME,
        price_per_unit=runner_settings.price_per_unit,
        pixels_per_unit=runner_settings.pixels_per_unit,
        price_unit=runner_settings.price_unit,
        mode="persistent",
        capacity=1,
        on_session_release=_on_session_release,
    )


def _build_inner_scope_env() -> dict[str, str]:
    env = os.environ.copy()

    shared_dir = Path.home() / DEFAULT_SCOPE_HOME_DIRNAME / "models"
    session_dir = DEFAULT_SESSION_ASSETS_DIR

    # Shared / persistent data.
    env.setdefault("DAYDREAM_SCOPE_MODELS_DIR", str(shared_dir))
    env.setdefault("DAYDREAM_SCOPE_LORA_SHARED_DIR", str(shared_dir / "lora"))

    # Session local data, removed after each session.
    env.setdefault("DAYDREAM_SCOPE_ASSETS_DIR", str(session_dir))
    env.setdefault("DAYDREAM_SCOPE_LORA_DIR", str(session_dir / "lora"))
    env.setdefault("DAYDREAM_SCOPE_LOGS_DIR", str(session_dir / "logs"))
    env.setdefault("DAYDREAM_SCOPE_PLUGINS_DIR", str(session_dir / "plugins"))

    return env


def _start_inner_scope(runner_settings: ScopeRunnerSettings) -> subprocess.Popen:
    command = [
        "livepeer-runner",
        "--host",
        runner_settings.inner_host,
        "--port",
        str(runner_settings.inner_port),
    ]
    logger.info("Starting inner Scope runner: %s", " ".join(command))
    return subprocess.Popen(command, env=_build_inner_scope_env())


async def _wait_for_inner_scope(runner_settings: ScopeRunnerSettings) -> None:
    deadline = (
        asyncio.get_running_loop().time()
        + runner_settings.inner_startup_timeout_seconds
    )
    status_url = _inner_status_url(runner_settings.inner_ws_url)
    while True:
        try:
            async with aiohttp.ClientSession() as client:
                async with client.get(
                    status_url,
                    timeout=aiohttp.ClientTimeout(total=2.0),
                ) as response:
                    if response.status == 200:
                        status = await response.json()
                        if (
                            status.get("listener_ready") is True
                            and status.get("connection_active") is False
                        ):
                            return
                        logger.info(
                            "Inner Scope runner not ready for new websocket: %s",
                            status,
                        )
                    else:
                        text = await response.text()
                        logger.info(
                            "Inner Scope status endpoint returned HTTP %s: %s",
                            response.status,
                            text[:500],
                        )
        except Exception as exc:
            logger.info("Inner Scope status check failed: %s", exc)

        if asyncio.get_running_loop().time() >= deadline:
            raise RuntimeError(
                f"Timed out waiting for inner Scope runner at {status_url}"
            )
        await asyncio.sleep(1.0)


def _inner_status_url(inner_ws_url: str) -> str:
    parsed = urlparse(inner_ws_url)
    scheme = "https" if parsed.scheme == "wss" else "http"
    path = "/internal/status"
    return urlunparse((scheme, parsed.netloc, path, "", "", ""))


def _inner_cleanup_url(inner_ws_url: str) -> str:
    parsed = urlparse(inner_ws_url)
    scheme = "https" if parsed.scheme == "wss" else "http"
    path = "/internal/cleanup-session"
    return urlunparse((scheme, parsed.netloc, path, "", "", ""))


def _get_cleanup_event() -> asyncio.Event:
    global _cleanup_event
    if _cleanup_event is None:
        _cleanup_event = asyncio.Event()
        _cleanup_event.set()
    return _cleanup_event


async def _cleanup_inner_scope_session(
    runner_settings: ScopeRunnerSettings,
) -> None:
    cleanup_url = _inner_cleanup_url(runner_settings.inner_ws_url)
    try:
        async with aiohttp.ClientSession() as client:
            async with client.post(
                cleanup_url,
                timeout=aiohttp.ClientTimeout(total=180.0),
            ) as response:
                text = await response.text()
                if response.status != 200:
                    logger.warning(
                        "Inner runner cleanup endpoint failed: HTTP %s: %s",
                        response.status,
                        text[:200],
                    )
                    return

                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    logger.warning(
                        "Inner runner cleanup endpoint returned invalid JSON: %s",
                        text[:200],
                    )
                    return

                if not isinstance(payload, dict) or not payload.get("ok", False):
                    logger.warning(
                        "Inner runner cleanup completed with issues: %s",
                        payload,
                    )
                else:
                    logger.info("Inner runner cleanup completed successfully")
    except Exception as exc:
        logger.warning("Inner runner cleanup request failed: %s", exc)


async def _run_inner_cleanup(runner_settings: ScopeRunnerSettings) -> None:
    event = _get_cleanup_event()
    event.clear()
    try:
        await _cleanup_inner_scope_session(runner_settings)
    except Exception as exc:
        logger.warning("Inner runner cleanup request failed: %s", exc)
    finally:
        event.set()


async def _stop_inner_scope() -> None:
    global _inner_process
    process = _inner_process
    _inner_process = None
    if process is None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=10.0)
    except TimeoutError:
        process.kill()
        await asyncio.to_thread(process.wait)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _registration
    global _inner_process

    _inner_process = _start_inner_scope(settings)
    try:
        await _wait_for_inner_scope(settings)
        _registration = await _register_runner(settings)
    except Exception:
        await _stop_inner_scope()
        raise
    logger.info(
        "Registered Scope live runner app=%s runner_url=%s",
        LIVEPEER_APP_NAME,
        settings.resolved_runner_url(),
    )
    try:
        yield
    finally:
        await _cleanup_all_sessions(notify_orchestrator=True)
        if _registration is not None:
            await _registration.close()
            _registration = None
        await _stop_inner_scope()


app = FastAPI(
    lifespan=lifespan,
    title="Livepeer Scope Runner App",
    description="Registers Scope as a Livepeer live runner and bridges to the inner Scope websocket runner.",
)


@app.post("/scope")
async def scope_endpoint(request: Request) -> dict[str, Any]:
    # Ensure previous sessions are fully torn down before accepting a new one.
    await _get_cleanup_event().wait()

    headers = _header_snapshot(request)
    session_id = _session_id(headers)
    if session_id in _sessions:
        raise HTTPException(status_code=409, detail="session already active")

    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="request body must be an object")

    bridge_session = ScopeBridgeSession(
        session_id=session_id,
        headers=headers,
        job_info={},
        settings=settings,
    )

    channels = await create_trickle_channels(
        bridge_session,
        [
            {"name": "control", "mime_type": CHANNEL_MIME_JSONL},
            {"name": "events", "mime_type": CHANNEL_MIME_JSONL},
        ],
    )
    by_name = {channel["name"]: channel for channel in channels}
    control = by_name.get("control")
    events = by_name.get("events")
    if control is None or events is None:
        raise HTTPException(
            status_code=502,
            detail="orchestrator did not return control/events channels",
        )

    bridge_session.job_info = {
        "manifest_id": session_id,
        "control_url": control.get("internal_url") or control["url"],
        "events_url": events.get("internal_url") or events["url"],
        "params": body.get("params") if isinstance(body.get("params"), dict) else {},
    }
    bridge_session.channel_url_to_name[control["url"]] = control["name"]
    bridge_session.channel_url_to_name[events["url"]] = events["name"]
    _sessions[session_id] = bridge_session

    try:
        first_ws = await _connect_inner_once(bridge_session)
    except Exception:
        _sessions.pop(session_id, None)
        with suppress(Exception):
            await remove_trickle_channels(bridge_session, ["control", "events"])
        raise

    bridge_session.task = asyncio.create_task(
        _bridge_loop(bridge_session, first_ws=first_ws)
    )

    return {
        "manifest_id": session_id,
        "session_id": session_id,
        "control_url": control["url"],
        "events_url": events["url"],
    }


async def _bridge_loop(
    session: ScopeBridgeSession,
    *,
    first_ws: Any | None = None,
) -> None:
    reconnectable = _reconnectable_exceptions()
    failure_timestamps: list[float] = []
    ws = first_ws

    try:
        while not session.stop_requested:
            try:
                if ws is None:
                    ws = await _connect_inner_once(session)
                await _read_ws(session, ws)
                session.stop_requested = True
                break
            except reconnectable as exc:
                # The inner runner uses close code 4000 when the Livepeer control
                # channel has ended. That is an intentional session shutdown, not
                # a transport failure to heal with another websocket connection.
                if _is_control_channel_close(exc):
                    session.stop_requested = True
                    break
                logger.warning(
                    "Inner Scope websocket disconnected, reconnecting: %s", exc
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Inner Scope bridge failed")
                session.stop_requested = True

            with suppress(Exception):
                if ws is not None:
                    await ws.close()
            ws = None

            if session.stop_requested:
                break

            now = asyncio.get_running_loop().time()
            cutoff = now - session.settings.failure_window_seconds
            failure_timestamps.append(now)
            failure_timestamps = [ts for ts in failure_timestamps if ts > cutoff]
            if len(failure_timestamps) > session.settings.max_failures_per_window:
                logger.error(
                    "Inner Scope websocket failed too often; stopping session %s",
                    session.session_id,
                )
                break
            await asyncio.sleep(session.settings.retry_delay_seconds)
    finally:
        await _cleanup_session(session, notify_orchestrator=True)


async def _connect_inner_once(session: ScopeBridgeSession) -> Any:
    import websockets

    await _wait_for_inner_scope(session.settings)
    ws = await websockets.connect(session.settings.inner_ws_url)
    session.websocket = ws

    ready = parse_json(
        await asyncio.wait_for(ws.recv(), timeout=INNER_WS_HANDSHAKE_TIMEOUT_SECONDS)
    )
    if ready.get("type") != "ready":
        await ws.close()
        raise RuntimeError(f"inner runner did not send ready: {ready!r}")

    await ws.send(json.dumps(session.job_info))
    started = parse_json(
        await asyncio.wait_for(ws.recv(), timeout=INNER_WS_HANDSHAKE_TIMEOUT_SECONDS)
    )
    if started.get("type") == "error":
        await ws.close()
        _raise_inner_start_error(started)
    if started.get("type") != "started":
        await ws.close()
        raise RuntimeError(f"inner runner did not start: {started!r}")
    return ws


async def _read_ws(session: ScopeBridgeSession, ws: Any) -> None:
    async for raw in ws:
        await _handle_ws_message(session, ws, parse_json(raw))


def parse_json(raw: str) -> dict[str, Any]:
    message = json.loads(raw)
    if not isinstance(message, dict):
        raise RuntimeError(f"expected websocket JSON object, got {message!r}")
    return message


def _raise_inner_start_error(message: dict[str, Any]) -> None:
    error = message.get("error")
    detail = error if isinstance(error, str) and error else "inner runner did not start"
    if message.get("code") == "ACCESS_DENIED":
        raise HTTPException(status_code=401, detail=detail)
    raise RuntimeError(f"inner runner did not start: {message!r}")


async def _handle_ws_message(
    session: ScopeBridgeSession,
    ws: Any,
    message: dict[str, Any],
) -> None:
    msg_type = message.get("type")
    if msg_type == "create_channels":
        await _handle_create_channels(session, ws, message)
        return
    if msg_type == "close_channels":
        await _handle_close_channels(session, message)
        return
    if msg_type == "ping":
        await ws.send(
            json.dumps(
                {
                    "type": "pong",
                    "request_id": message.get("request_id"),
                    "timestamp": message.get("timestamp"),
                }
            )
        )
        return
    if msg_type == "restarting":
        request_id = message.get("request_id")
        if not request_id:
            logger.warning(
                "Ignoring inner restarting message without request_id for session %s",
                session.session_id,
            )
            return
        await ws.send(
            json.dumps(
                {
                    "type": "response",
                    "request_id": request_id,
                }
            )
        )
        logger.info(
            "Acknowledged inner restarting message for session %s",
            session.session_id,
        )
        return

    logger.debug("Ignoring inner websocket message type=%s", msg_type)


async def _handle_create_channels(
    session: ScopeBridgeSession,
    ws: Any,
    message: dict[str, Any],
) -> None:
    request_id = message.get("request_id")
    direction = message.get("direction")
    if direction not in {"in", "out"}:
        await _send_ws_error(ws, request_id, "create_channels direction must be in/out")
        return

    mime_type = message.get("mime_type")
    if not isinstance(mime_type, str) or not mime_type:
        mime_type = "video/MP2T"
    channel_name = _next_channel_name(session, direction, request_id)
    try:
        channels = await create_trickle_channels(
            session,
            [{"name": channel_name, "mime_type": mime_type}],
        )
    except Exception as exc:
        logger.exception("Failed to create stream channel")
        await _send_ws_error(ws, request_id, str(exc))
        return

    response_channels = []
    for channel in channels:
        url = channel.get("url")
        if not isinstance(url, str) or not url:
            continue
        session.channel_url_to_name[url] = channel.get("name", channel_name)
        response_channel = {
            **channel,
            "direction": direction,
            "mime_type": channel.get("mime_type", mime_type),
        }
        response_channels.append(response_channel)

    await ws.send(
        json.dumps(
            {
                "type": "response",
                "request_id": request_id,
                "channels": response_channels,
            }
        )
    )


async def _handle_close_channels(
    session: ScopeBridgeSession,
    message: dict[str, Any],
) -> None:
    urls = message.get("channels")
    if not isinstance(urls, list):
        return
    names = []
    for url in urls:
        if not isinstance(url, str):
            continue
        name = session.channel_url_to_name.pop(url, None)
        if name is not None:
            names.append(name)
    if not names:
        return
    with suppress(Exception):
        await remove_trickle_channels(session, names)


async def _send_ws_error(ws: Any, request_id: Any, error: str) -> None:
    await ws.send(
        json.dumps(
            {
                "type": "response",
                "request_id": request_id,
                "error": error,
                "channels": [],
            }
        )
    )


def _next_channel_name(
    session: ScopeBridgeSession,
    direction: str,
    request_id: Any,
) -> str:
    session.channel_counter += 1
    request_part = str(request_id or session.channel_counter)
    request_part = re.sub(r"[^A-Za-z0-9_.-]+", "-", request_part).strip("-")
    if not request_part:
        request_part = str(session.channel_counter)
    return f"{direction}-{request_part}-{session.channel_counter}"


async def _cleanup_all_sessions(*, notify_orchestrator: bool) -> None:
    sessions = list(_sessions.values())
    await asyncio.gather(
        *(
            _cleanup_session(session, notify_orchestrator=notify_orchestrator)
            for session in sessions
        ),
        return_exceptions=True,
    )


async def _cleanup_session(
    session: ScopeBridgeSession,
    *,
    notify_orchestrator: bool,
) -> None:
    if session.cleanup_started:
        return
    session.cleanup_started = True
    session.stop_requested = True
    _sessions.pop(session.session_id, None)

    task = session.task
    if task is not None and task is not asyncio.current_task() and not task.done():
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    websocket = session.websocket
    session.websocket = None
    if websocket is not None:
        with suppress(Exception):
            await websocket.close()

    channel_names = list(set(session.channel_url_to_name.values()))
    session.channel_url_to_name.clear()
    with suppress(Exception):
        await remove_trickle_channels(session, channel_names)

    if notify_orchestrator:
        with suppress(Exception):
            await stop_runner_session(session)

    await _run_inner_cleanup(session.settings)


def _reconnectable_exceptions() -> tuple[type[BaseException], ...]:
    from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus

    return (ConnectionClosed, InvalidStatus, InvalidHandshake, OSError)


def _is_control_channel_close(exc: BaseException) -> bool:
    for close in (getattr(exc, "rcvd", None), getattr(exc, "sent", None)):
        if getattr(close, "code", None) == 4000:
            return True
    return False


def _on_session_release(event: Any) -> None:
    session = _sessions.get(event.session_id)
    if session is None:
        return
    asyncio.create_task(_cleanup_session(session, notify_orchestrator=False))


@click.command()
@click.option("--host", default=DEFAULT_HOST, show_default=True, help="Host to bind to")
@click.option("--port", default=DEFAULT_PORT, show_default=True, help="Port to bind to")
@click.option(
    "--orchestrator",
    default=DEFAULT_ORCHESTRATOR_URL,
    show_default=True,
    help="Livepeer orchestrator URL",
)
@click.option(
    "--orch-secret",
    envvar="LIVEPEER_ORCH_SECRET",
    required=True,
    help="Livepeer runner bootstrap secret",
)
@click.option(
    "--runner-url",
    default="",
    help="Public runner URL registered with the orchestrator",
)
@click.option(
    "--inner-ws-url",
    default=f"ws://127.0.0.1:{DEFAULT_INNER_PORT}/ws",
    show_default=True,
    help="Inner Scope runner websocket URL",
)
@click.option(
    "--price-per-unit",
    envvar="LIVEPEER_RUNNER_PRICE_PER_UNIT",
    default=DEFAULT_PRICE_PER_UNIT,
    show_default=True,
    type=int,
    help="Runner price per unit advertised to the orchestrator",
)
@click.option(
    "--pixels-per-unit",
    envvar="LIVEPEER_RUNNER_PIXELS_PER_UNIT",
    default=DEFAULT_PIXELS_PER_UNIT,
    show_default=True,
    type=int,
    help="Pixels per advertised price unit",
)
@click.option(
    "--price-unit",
    envvar="LIVEPEER_RUNNER_PRICE_UNIT",
    default=DEFAULT_PRICE_UNIT,
    show_default=True,
    help="Currency unit for advertised runner price",
)
def main(
    host: str,
    port: int,
    orchestrator: str,
    orch_secret: str,
    runner_url: str,
    inner_ws_url: str,
    price_per_unit: int,
    pixels_per_unit: int,
    price_unit: str,
) -> None:
    """Run the Livepeer Scope live-runner app."""
    global settings
    logging.basicConfig(level=logging.INFO)
    _validate_price_settings(price_per_unit, pixels_per_unit)
    inner_host, inner_port = _inner_bind_from_ws_url(inner_ws_url)
    settings = ScopeRunnerSettings(
        host=host,
        port=port,
        orchestrator_url=orchestrator,
        orch_secret=orch_secret,
        runner_url=runner_url,
        inner_ws_url=inner_ws_url,
        inner_host=inner_host,
        inner_port=inner_port,
        price_per_unit=price_per_unit,
        pixels_per_unit=pixels_per_unit,
        price_unit=price_unit,
    )
    uvicorn.run(
        "scope.cloud.livepeer_scope_app:app",
        host=host,
        port=port,
        log_config=None,
    )


settings = _settings_from_env()


if __name__ == "__main__":
    main()
