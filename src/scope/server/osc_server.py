"""Always-on OSC UDP server that shares the same numeric port as the HTTP API.

The server binds a UDP socket on the configured API port (TCP and UDP can coexist
on the same port number). It runs for the full application lifetime and dispatches
incoming OSC messages to the active pipeline's parameter update path.

Every received message is validated against the known path inventory and logged
as either valid or invalid before being forwarded.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import AsyncIOOSCUDPServer

if TYPE_CHECKING:
    from .pipeline_manager import PipelineManager
    from .webrtc import WebRTCManager

logger = logging.getLogger(__name__)

# Stored transition settings so that prompt, transition_steps, and
# interpolation_method OSC messages can arrive independently.
_transition_steps: int = 4
_temporal_interpolation_method: str = "linear"
_interpolation_method: str = "linear"


def _transform_osc_param(key: str, value: Any) -> dict[str, Any]:
    """Map an OSC key/value into the parameter dict expected by the pipeline.

    Most keys pass through unchanged, but some Input & Controls keys need
    to be restructured into the format the frame processor expects.
    """
    global _transition_steps, _temporal_interpolation_method, _interpolation_method

    if key == "prompt":
        prompt_item = {"text": str(value), "weight": 1.0}
        if _transition_steps > 0:
            return {
                "transition": {
                    "target_prompts": [prompt_item],
                    "num_steps": _transition_steps,
                    "temporal_interpolation_method": _temporal_interpolation_method,
                },
            }
        return {
            "prompts": [prompt_item],
            "prompt_interpolation_method": _interpolation_method,
        }

    if key == "transition_steps":
        _transition_steps = int(value)
        return {}

    if key == "interpolation_method":
        _interpolation_method = str(value)
        return {}

    if key == "temporal_interpolation_method":
        _temporal_interpolation_method = str(value)
        return {}

    return {key: value}


class OSCServer:
    """Manages the always-on OSC UDP listener with path validation."""

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._server: AsyncIOOSCUDPServer | None = None
        self._transport: asyncio.DatagramTransport | None = None
        self._listening = False
        self._pipeline_manager: PipelineManager | None = None
        self._webrtc_manager: WebRTCManager | None = None
        # Cached path registry; rebuilt on each message to stay current
        self._path_cache: dict[str, dict[str, Any]] | None = None

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

    def _get_known_paths(self) -> dict[str, dict[str, Any]]:
        """Return the current set of known OSC paths, rebuilding each time.

        Rebuilding is cheap (just iterates registry metadata) and ensures we
        always reflect the latest plugins / pipeline changes.
        """
        from .osc_docs import get_all_known_paths

        return get_all_known_paths(self._pipeline_manager)

    def _build_dispatcher(self) -> Dispatcher:
        dispatcher = Dispatcher()
        dispatcher.map("/scope/*", self._handle_osc_message)
        return dispatcher

    def _handle_osc_message(self, address: str, *args) -> None:
        """Validate and optionally forward an incoming OSC message."""
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

        # Look up path in current inventory
        known = self._get_known_paths()
        path_info = known.get(key)

        if path_info is None:
            logger.info(
                "OSC INVALID  %s = %r  reason=unknown path",
                address,
                value,
            )
            return

        # Validate type / range / enum
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

        logger.info("OSC VALID    %s = %r", address, value)

        if not self._webrtc_manager:
            logger.debug("OSC message not forwarded – no WebRTC manager")
            return

        try:
            params = _transform_osc_param(key, value)
            if params:
                self._webrtc_manager.broadcast_parameter_update(params)
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
        }
