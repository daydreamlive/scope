"""Always-on OSC UDP server that shares the same numeric port as the HTTP API.

The server binds a UDP socket on the configured API port (TCP and UDP can coexist
on the same port number). It runs for the full application lifetime and dispatches
incoming OSC messages to the active pipeline's parameter update path.

Every received message is validated against the known path inventory and logged
as either valid or invalid before being forwarded.
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import AsyncIOOSCUDPServer

if TYPE_CHECKING:
    from .pipeline_manager import PipelineManager
    from .webrtc import WebRTCManager

logger = logging.getLogger(__name__)

# How long (seconds) to cache the known-path inventory before rebuilding.
_PATH_CACHE_TTL = 5.0


class OSCServer:
    """Manages the always-on OSC UDP listener with path validation."""

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._server: AsyncIOOSCUDPServer | None = None
        self._transport: asyncio.DatagramTransport | None = None
        self._listening = False
        self._log_all_messages = False
        self._pipeline_manager: PipelineManager | None = None
        self._webrtc_manager: WebRTCManager | None = None
        # SSE subscribers — each is an asyncio.Queue fed by _handle_osc_message.
        self._sse_queues: list[asyncio.Queue] = []
        # Cached path inventory to avoid rebuilding on every OSC message.
        self._known_paths_cache: dict[str, dict[str, Any]] | None = None
        self._known_paths_cache_time: float = 0.0

    @property
    def port(self) -> int:
        return self._port

    @property
    def host(self) -> str:
        return self._host

    @property
    def listening(self) -> bool:
        return self._listening

    @property
    def log_all_messages(self) -> bool:
        return self._log_all_messages

    @log_all_messages.setter
    def log_all_messages(self, value: bool) -> None:
        self._log_all_messages = value

    def set_managers(
        self,
        pipeline_manager: "PipelineManager",
        webrtc_manager: "WebRTCManager",
    ) -> None:
        self._pipeline_manager = pipeline_manager
        self._webrtc_manager = webrtc_manager
        # Invalidate the path cache when managers change.
        self._known_paths_cache = None

    def subscribe(self) -> "asyncio.Queue[dict[str, Any]]":
        """Register a new SSE subscriber and return its event queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._sse_queues.append(q)
        return q

    def unsubscribe(self, q: "asyncio.Queue") -> None:
        """Deregister an SSE subscriber."""
        try:
            self._sse_queues.remove(q)
        except ValueError:
            pass

    def _get_known_paths(self) -> dict[str, dict[str, Any]]:
        """Return the known OSC paths, rebuilding from the registry when stale."""
        now = time.monotonic()
        if (
            self._known_paths_cache is None
            or now - self._known_paths_cache_time > _PATH_CACHE_TTL
        ):
            from .osc_docs import get_all_known_paths

            self._known_paths_cache = get_all_known_paths(self._pipeline_manager)
            self._known_paths_cache_time = now
        return self._known_paths_cache

    def _build_dispatcher(self) -> Dispatcher:
        dispatcher = Dispatcher()
        dispatcher.map("/scope/*", self._handle_osc_message)
        return dispatcher

    def _handle_osc_message(self, address: str, *args) -> None:
        """Validate and forward an incoming OSC message."""
        parts = address.split("/")
        if len(parts) < 3:
            logger.info(
                "OSC INVALID  %s  reason=address too short  args=%r",
                address,
                args,
            )
            return

        key = "/".join(parts[2:])
        value = args[0] if len(args) == 1 else list(args)

        known = self._get_known_paths()
        path_info = known.get(key)

        if path_info is None:
            logger.info(
                "OSC UNKNOWN  %s = %r",
                address,
                value,
            )
            return

        from .osc_docs import validate_osc_value

        reason = validate_osc_value(path_info, value)
        if reason:
            logger.info(
                "OSC INVALID  %s = %r  reason=%s",
                address,
                value,
                reason,
            )
            return

        if self._log_all_messages:
            logger.info("OSC OK  %s = %r", address, value)

        # Apply the parameter immediately to all active local sessions so
        # the pipeline effect takes place without waiting for the frontend
        # round-trip.
        if self._webrtc_manager:
            self._webrtc_manager.broadcast_parameter_update({key: value})

        # Push to all SSE subscribers so the frontend can sync its UI state.
        event: dict[str, Any] = {"type": "osc_command", "key": key, "value": value}
        for q in list(self._sse_queues):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    async def start(self) -> None:
        dispatcher = self._build_dispatcher()
        try:
            self._server = AsyncIOOSCUDPServer(
                (self._host, self._port),
                dispatcher,
                asyncio.get_running_loop(),
            )
            self._transport, _protocol = await self._server.create_serve_endpoint()
            self._listening = True
            logger.info("OSC server listening on udp://%s:%d", self._host, self._port)
        except Exception:
            logger.exception(
                "Failed to start OSC server on udp://%s:%d",
                self._host,
                self._port,
            )

    async def stop(self) -> None:
        if self._transport:
            self._transport.close()
            self._transport = None
        self._listening = False
        logger.info("OSC server stopped")

    def status(self) -> dict:
        return {
            "enabled": True,
            "listening": self._listening,
            "port": self._port,
            "host": self._host,
            "log_all_messages": self._log_all_messages,
        }
