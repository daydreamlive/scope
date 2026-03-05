"""Always-on OSC UDP server that shares the same numeric port as the HTTP API.

The server binds a UDP socket on the configured API port (TCP and UDP can coexist
on the same port number). It runs for the full application lifetime and dispatches
incoming OSC messages to the active pipeline's parameter update path.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import AsyncIOOSCUDPServer

if TYPE_CHECKING:
    from .pipeline_manager import PipelineManager
    from .webrtc import WebRTCManager

logger = logging.getLogger(__name__)


class OSCServer:
    """Manages the always-on OSC UDP listener."""

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._server: AsyncIOOSCUDPServer | None = None
        self._transport: asyncio.DatagramTransport | None = None
        self._listening = False
        self._pipeline_manager: PipelineManager | None = None
        self._webrtc_manager: WebRTCManager | None = None

    @property
    def port(self) -> int:
        return self._port

    @property
    def host(self) -> str:
        return self._host

    @property
    def listening(self) -> bool:
        return self._listening

    def set_managers(
        self,
        pipeline_manager: "PipelineManager",
        webrtc_manager: "WebRTCManager",
    ) -> None:
        self._pipeline_manager = pipeline_manager
        self._webrtc_manager = webrtc_manager

    def _build_dispatcher(self) -> Dispatcher:
        dispatcher = Dispatcher()
        dispatcher.map("/scope/*", self._handle_osc_message)
        return dispatcher

    def _handle_osc_message(self, address: str, *args) -> None:
        """Route an incoming OSC message to active WebRTC sessions as a parameter update."""
        if not self._webrtc_manager:
            logger.debug("OSC message ignored – no WebRTC manager available")
            return

        parts = address.split("/")
        # address is e.g. "/scope/noise_scale" → key = "noise_scale"
        # or "/scope/some_plugin/param" → key = "some_plugin/param"
        if len(parts) < 3:
            logger.warning("OSC address too short: %s", address)
            return

        key = "/".join(parts[2:])
        value = args[0] if len(args) == 1 else list(args)

        logger.debug("OSC → %s = %r", key, value)

        try:
            self._webrtc_manager.broadcast_parameter_update({key: value})
        except Exception:
            logger.exception("Error forwarding OSC message %s", address)

    async def start(self) -> None:
        dispatcher = self._build_dispatcher()
        try:
            self._server = AsyncIOOSCUDPServer(
                (self._host, self._port),
                dispatcher,
                asyncio.get_event_loop(),
            )
            self._transport, _protocol = await self._server.create_serve_endpoint()
            self._listening = True
            logger.info("OSC server listening on udp://%s:%d", self._host, self._port)
        except Exception:
            logger.exception(
                "Failed to start OSC server on udp://%s:%d", self._host, self._port
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
        }
