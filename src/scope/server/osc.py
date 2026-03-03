"""OSC (Open Sound Control) server for real-time parameter control.

Allows external applications (TouchDesigner, Resolume, Max/MSP, MIDI controllers,
etc.) to control Scope's pipeline parameters over UDP using the OSC protocol.

The OSC server is started/stopped at runtime via the HTTP API
(POST /api/v1/osc/config) and binds to the same host/port as the main HTTP server
(UDP and TCP can share a port number).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import AsyncIOOSCUDPServer

if TYPE_CHECKING:
    from .webrtc import WebRTCManager

logger = logging.getLogger(__name__)


class OSCManager:
    """Manages an async OSC UDP server that maps OSC messages to pipeline parameter updates.

    The server listens on a UDP port and dispatches incoming OSC messages to a
    generic handler that translates them into Scope parameter updates, which are
    then pushed to all active WebRTC sessions' frame processors.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._webrtc_manager: WebRTCManager | None = None
        self._server: AsyncIOOSCUDPServer | None = None
        self._transport: asyncio.BaseTransport | None = None
        self._listening = False

    @property
    def listening(self) -> bool:
        return self._listening

    def _get_active_frame_processors(self) -> list:
        """Get frame processors from all active WebRTC sessions."""
        if not self._webrtc_manager:
            return []

        processors = []
        for session in self._webrtc_manager.sessions.values():
            if (
                session.video_track
                and hasattr(session.video_track, "frame_processor")
                and session.video_track.frame_processor is not None
            ):
                processors.append(session.video_track.frame_processor)
        return processors

    def _push_parameters(self, parameters: dict[str, Any]) -> None:
        """Push parameter updates to all active sessions."""
        processors = self._get_active_frame_processors()
        if not processors:
            logger.debug("OSC message received but no active sessions")
            return

        for processor in processors:
            try:
                processor.update_parameters(parameters)
            except Exception as e:
                logger.error(f"Error pushing OSC parameter update: {e}")

    def _setup_dispatcher(self) -> Dispatcher:
        """Configure the OSC dispatcher with a single generic handler."""
        dispatcher = Dispatcher()
        dispatcher.set_default_handler(self._handle_message)
        return dispatcher

    def _handle_message(self, address: str, *args) -> None:
        """Handle any OSC message by forwarding address/value as a parameter update.

        Messages under the /scope/ namespace have their address path (minus the
        leading /scope/) used as the parameter key. Messages outside the namespace
        are ignored.
        """
        parts = address.strip("/").split("/")
        if len(parts) < 2 or parts[0] != "scope":
            logger.debug(f"OSC ignoring address outside /scope/ namespace: {address}")
            return

        key = "/".join(parts[1:])
        if not args:
            logger.warning(f"OSC {address}: no value provided")
            return

        value = args[0] if len(args) == 1 else list(args)
        logger.info(f"OSC: {key} = {value}")
        self._push_parameters({key: value})

    # --- Lifecycle ---

    async def start(self, webrtc_manager: WebRTCManager) -> bool:
        """Start the OSC UDP server.

        Returns True if started successfully, False otherwise.
        """
        self._webrtc_manager = webrtc_manager

        try:
            dispatcher = self._setup_dispatcher()
            self._server = AsyncIOOSCUDPServer(
                (self.host, self.port),
                dispatcher,
                asyncio.get_running_loop(),
            )
            self._transport, _ = await self._server.create_serve_endpoint()
            self._listening = True
            logger.info(f"OSC server listening on UDP {self.host}:{self.port}")
            return True
        except OSError as e:
            logger.error(f"Failed to start OSC server on {self.host}:{self.port}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error starting OSC server: {e}")
            return False

    def stop(self) -> None:
        """Stop the OSC server and release resources."""
        if self._transport:
            self._transport.close()
            self._transport = None
        self._server = None
        self._listening = False
        self._webrtc_manager = None
        logger.info("OSC server stopped")
