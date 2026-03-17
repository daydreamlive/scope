"""Art-Net DMX UDP server for mapping DMX channels to pipeline params.

The server tries to bind on the standard Art-Net port (6454) and falls back to
6455, 6456, 6457 if that port is unavailable.  It runs for the full application
lifetime, parses incoming ArtDMX packets, maps channels to configured parameters
(scaling 0-255 to each parameter's min/max range), and dispatches updates to
the active pipeline.
"""

import asyncio
import logging
import struct
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pipeline_manager import PipelineManager
    from .webrtc import WebRTCManager

logger = logging.getLogger(__name__)

ARTNET_HEADER = b"Art-Net\x00"
ARTNET_OPCODE_DMX = 0x5000
ARTNET_DEFAULT_PORT = 6454
ARTNET_PORT_ATTEMPTS = 4  # try 6454, 6455, 6456, 6457

# Throttle: minimum interval (seconds) between forwarding updates for the same
# parameter to avoid flooding the data-channel at DMX refresh rates (~44 Hz).
_THROTTLE_INTERVAL = 0.016  # ~60 Hz cap


class _ArtNetProtocol(asyncio.DatagramProtocol):
    """Low-level asyncio protocol that hands raw ArtDMX frames to the server."""

    def __init__(self, server: "DMXServer"):
        self._server = server

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        pass

    def datagram_received(self, data: bytes, addr: tuple) -> None:
        if self._server.log_all_messages:
            logger.info("DMX UDP packet received: %d bytes from %s", len(data), addr)
        self._server._handle_packet(data)

    def error_received(self, exc: Exception) -> None:
        logger.debug("Art-Net socket error: %s", exc)


class DMXServer:
    """Manages the Art-Net DMX UDP listener with channel mapping."""

    def __init__(self, host: str, preferred_port: int = ARTNET_DEFAULT_PORT):
        self._host = host
        self._preferred_port = preferred_port
        self._bound_port: int | None = None
        self._transport: asyncio.DatagramTransport | None = None
        self._listening = False
        self._enabled = False
        self._log_all_messages = False

        self._pipeline_manager: PipelineManager | None = None
        self._webrtc_manager: WebRTCManager | None = None

        # SSE subscribers
        self._sse_queues: list[asyncio.Queue] = []

        # Channel -> parameter mappings (loaded from config)
        # Key: (universe, channel), Value: parameter key string
        self._mappings: dict[tuple[int, int], str] = {}

        # Last-value cache per parameter to avoid re-sending identical values
        self._last_values: dict[str, float] = {}
        self._last_send_time: dict[str, float] = {}

        # Full DMX frame buffer per universe (512 channels each, 0-255)
        self._universes: dict[int, bytearray] = {}

        # One-time hint when packets arrive but no mappings configured
        self._logged_no_mappings = False
        # One-time log to confirm packets are arriving
        self._logged_first_packet = False

        # Cached known-path inventory for DMX key validation/range mapping.
        # The cache is refreshed lazily when marked dirty by mapping/pipeline changes.
        self._cached_known_paths: dict[str, dict[str, Any]] = {}
        self._known_paths_version = 0
        self._known_paths_cached_version = -1

    # -- Properties -----------------------------------------------------------

    @property
    def port(self) -> int | None:
        return self._bound_port

    @property
    def preferred_port(self) -> int:
        return self._preferred_port

    @preferred_port.setter
    def preferred_port(self, value: int) -> None:
        self._preferred_port = value

    @property
    def host(self) -> str:
        return self._host

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def listening(self) -> bool:
        return self._listening

    @property
    def log_all_messages(self) -> bool:
        return self._log_all_messages

    @log_all_messages.setter
    def log_all_messages(self, value: bool) -> None:
        self._log_all_messages = value

    @property
    def mappings(self) -> dict[tuple[int, int], str]:
        return dict(self._mappings)

    # -- Manager injection ----------------------------------------------------

    def set_managers(
        self,
        pipeline_manager: "PipelineManager",
        webrtc_manager: "WebRTCManager",
    ) -> None:
        self._pipeline_manager = pipeline_manager
        self._webrtc_manager = webrtc_manager
        self.invalidate_known_paths_cache()

    # -- Mapping management ---------------------------------------------------

    def set_mappings(self, mappings: dict[tuple[int, int], str]) -> None:
        """Replace the active channel->parameter mapping table."""
        self._mappings = dict(mappings)
        self._last_values.clear()
        self._last_send_time.clear()
        self.invalidate_known_paths_cache()

    def invalidate_known_paths_cache(self) -> None:
        """Mark known-path cache stale so it is rebuilt on next packet."""
        self._known_paths_version += 1

    # -- SSE ------------------------------------------------------------------

    def subscribe(self) -> "asyncio.Queue[dict[str, Any]]":
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._sse_queues.append(q)
        return q

    def unsubscribe(self, q: "asyncio.Queue") -> None:
        try:
            self._sse_queues.remove(q)
        except ValueError:
            pass

    # -- Packet handling ------------------------------------------------------

    def _handle_packet(self, data: bytes) -> None:
        if not self._logged_first_packet:
            self._logged_first_packet = True
            logger.info(
                "DMX Art-Net: first UDP packet received (%d bytes, header=%r)",
                len(data),
                data[:8],
            )

        if len(data) < 18:
            return
        if data[:8] != ARTNET_HEADER:
            return
        opcode = struct.unpack_from("<H", data, 8)[0]
        if opcode != ARTNET_OPCODE_DMX:
            return

        # ArtDMX layout after header+opcode:
        #   protver_hi(1) protver_lo(1) sequence(1) physical(1) universe_lo(1) universe_hi(1) length_hi(1) length_lo(1) data...
        sequence = data[12]  # noqa: F841
        universe = data[14] | (data[15] << 8)
        length = (data[16] << 8) | data[17]
        dmx_data = data[18 : 18 + length]

        if not dmx_data:
            return

        # Update universe buffer
        if universe not in self._universes:
            self._universes[universe] = bytearray(512)
        buf = self._universes[universe]
        buf[: len(dmx_data)] = dmx_data

        self._process_mappings(universe, dmx_data)

    def _process_mappings(self, universe: int, dmx_data: bytes) -> None:
        """Check mapped channels for value changes and dispatch updates."""
        if not self._mappings:
            if not self._logged_no_mappings:
                self._logged_no_mappings = True
                logger.info(
                    "DMX Art-Net packets received but no channel mappings configured. "
                    "Add mappings in Settings → DMX."
                )
            return

        now = time.monotonic()
        known_paths = self._get_known_paths()

        for (uni, ch), param_key in self._mappings.items():
            if uni != universe:
                continue
            index = ch - 1
            if ch <= 0 or index >= len(dmx_data):
                continue

            raw_value = dmx_data[index]
            path_info = known_paths.get(param_key)
            if path_info is None:
                continue

            lo = path_info.get("min", 0.0)
            hi = path_info.get("max", 1.0)
            param_type = path_info.get("type", "float")

            # Scale 0-255 -> min..max
            normalized = raw_value / 255.0
            scaled = lo + normalized * (hi - lo)
            if param_type == "integer":
                scaled = round(scaled)
            else:
                scaled = round(scaled, 4)

            # De-duplicate: skip if value unchanged
            if self._last_values.get(param_key) == scaled:
                continue

            # Throttle per-parameter
            last_t = self._last_send_time.get(param_key, 0.0)
            if now - last_t < _THROTTLE_INTERVAL:
                continue

            self._last_values[param_key] = scaled
            self._last_send_time[param_key] = now

            if self._log_all_messages:
                logger.info(
                    "DMX OK  uni=%d ch=%d raw=%d -> %s = %s",
                    universe,
                    ch,
                    raw_value,
                    param_key,
                    scaled,
                )

            # Broadcast to active sessions
            if self._webrtc_manager:
                self._webrtc_manager.broadcast_parameter_update({param_key: scaled})

            # Push SSE event
            event: dict[str, Any] = {
                "type": "dmx_command",
                "key": param_key,
                "value": scaled,
            }
            for q in list(self._sse_queues):
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass

        # Log unmapped channels with non-zero values when verbose logging is on
        if self._log_all_messages:
            for ch, raw in enumerate(dmx_data):
                if raw == 0:
                    continue
                if (universe, ch + 1) not in self._mappings:
                    logger.info(
                        "DMX UNMAPPED  uni=%d ch=%d raw=%d (no mapping)",
                        universe,
                        ch + 1,
                        raw,
                    )

    def _get_known_paths(self) -> dict[str, dict[str, Any]]:
        """Return flat dict of numeric-only runtime parameters."""
        if self._known_paths_cached_version != self._known_paths_version:
            from .dmx_paths import get_all_numeric_paths

            self._cached_known_paths = get_all_numeric_paths(self._pipeline_manager)
            self._known_paths_cached_version = self._known_paths_version
        return self._cached_known_paths

    # -- Lifecycle ------------------------------------------------------------

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        last_exc: Exception | None = None
        self._bound_port = None

        for offset in range(ARTNET_PORT_ATTEMPTS):
            port = self._preferred_port + offset
            try:
                transport, _protocol = await loop.create_datagram_endpoint(
                    lambda: _ArtNetProtocol(self),
                    local_addr=(self._host, port),
                )
                self._transport = transport
                self._bound_port = port
                self._listening = True
                logger.info(
                    "DMX Art-Net server listening on udp://%s:%d",
                    self._host,
                    port,
                )
                return
            except OSError as exc:
                last_exc = exc
                logger.debug("DMX Art-Net port %d unavailable: %s", port, exc)

        self._bound_port = None
        self._listening = False
        logger.warning(
            "Failed to start DMX Art-Net server on ports %d-%d: %s",
            self._preferred_port,
            self._preferred_port + ARTNET_PORT_ATTEMPTS - 1,
            last_exc,
        )

    async def stop(self) -> None:
        if self._transport:
            self._transport.close()
            self._transport = None
        self._bound_port = None
        self._listening = False
        logger.info("DMX Art-Net server stopped")

    def status(self) -> dict:
        return {
            "enabled": self._enabled,
            "listening": self._listening,
            "port": self._bound_port,
            "preferred_port": self._preferred_port,
            "host": self._host,
            "log_all_messages": self._log_all_messages,
            "mapping_count": len(self._mappings),
        }
