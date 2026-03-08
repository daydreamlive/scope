"""Art-Net DMX UDP server for external parameter control.

Receives Art-Net DMX frames and maps channel values to pipeline parameters.
Follows the same architecture as the OSC server but with channel-to-parameter
mapping since DMX doesn't have named addresses like OSC.

Art-Net uses UDP port 6454 by default. Each universe contains 512 channels
with 8-bit values (0-255).
"""

import asyncio
import json
import logging
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pipeline_manager import PipelineManager
    from .webrtc import WebRTCManager

logger = logging.getLogger(__name__)

# Art-Net constants
ARTNET_PORT = 6454
ARTNET_HEADER = b"Art-Net\x00"
ARTNET_OPCODE_DMX = 0x5000

# How often to broadcast updates (rate limiting)
_MIN_BROADCAST_INTERVAL = 0.016  # ~60fps max


@dataclass
class DMXMapping:
    """Maps a DMX channel to a pipeline parameter."""

    id: str
    universe: int
    channel: int  # 1-512 (DMX convention, 1-indexed)
    param_key: str
    min_value: float = 0.0
    max_value: float = 1.0
    enabled: bool = True

    def scale(self, raw: int) -> float:
        """Convert 0-255 DMX value to parameter range."""
        normalized = raw / 255.0
        return self.min_value + normalized * (self.max_value - self.min_value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "universe": self.universe,
            "channel": self.channel,
            "param_key": self.param_key,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DMXMapping":
        return cls(
            id=data["id"],
            universe=data["universe"],
            channel=data["channel"],
            param_key=data["param_key"],
            min_value=data.get("min_value", 0.0),
            max_value=data.get("max_value", 1.0),
            enabled=data.get("enabled", True),
        )


@dataclass
class DMXMappingStore:
    """Persistent storage for DMX channel mappings."""

    mappings: dict[str, DMXMapping] = field(default_factory=dict)
    _config_path: Path | None = None

    @classmethod
    def load(cls, config_dir: Path) -> "DMXMappingStore":
        """Load mappings from config file."""
        config_path = config_dir / "dmx_mappings.json"
        store = cls(_config_path=config_path)

        if config_path.exists():
            try:
                data = json.loads(config_path.read_text())
                for mapping_data in data.get("mappings", []):
                    mapping = DMXMapping.from_dict(mapping_data)
                    store.mappings[mapping.id] = mapping
                logger.info(f"Loaded {len(store.mappings)} DMX mappings from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load DMX mappings: {e}")

        return store

    def save(self) -> None:
        """Save mappings to config file."""
        if self._config_path is None:
            return

        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "mappings": [m.to_dict() for m in self.mappings.values()]
            }
            self._config_path.write_text(json.dumps(data, indent=2))
            logger.debug(f"Saved {len(self.mappings)} DMX mappings")
        except Exception as e:
            logger.error(f"Failed to save DMX mappings: {e}")

    def add(self, mapping: DMXMapping) -> None:
        """Add or update a mapping."""
        self.mappings[mapping.id] = mapping
        self.save()

    def remove(self, mapping_id: str) -> bool:
        """Remove a mapping by ID."""
        if mapping_id in self.mappings:
            del self.mappings[mapping_id]
            self.save()
            return True
        return False

    def get_by_channel(self, universe: int, channel: int) -> DMXMapping | None:
        """Find mapping for a specific universe/channel."""
        for mapping in self.mappings.values():
            if mapping.universe == universe and mapping.channel == channel and mapping.enabled:
                return mapping
        return None


class ArtNetProtocol(asyncio.DatagramProtocol):
    """UDP protocol handler for Art-Net packets."""

    def __init__(self, server: "DMXServer"):
        self._server = server

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        self._transport = transport

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        self._server._handle_artnet_packet(data, addr)


class DMXServer:
    """Manages the Art-Net DMX UDP listener with channel-to-parameter mapping."""

    def __init__(self, port: int = ARTNET_PORT, config_dir: Path | None = None):
        self._port = port
        self._transport: asyncio.DatagramTransport | None = None
        self._listening = False
        self._pipeline_manager: "PipelineManager | None" = None
        self._webrtc_manager: "WebRTCManager | None" = None

        # SSE subscribers for real-time DMX monitoring
        self._sse_queues: list[asyncio.Queue] = []

        # Channel mappings
        if config_dir is None:
            config_dir = Path.home() / ".daydream-scope"
        self._mapping_store = DMXMappingStore.load(config_dir)

        # Rate limiting for broadcasts
        self._last_broadcast_time: float = 0.0
        self._pending_updates: dict[str, Any] = {}

        # Cache last known channel values for monitoring
        self._channel_values: dict[tuple[int, int], int] = {}

    @property
    def port(self) -> int:
        return self._port

    @property
    def listening(self) -> bool:
        return self._listening

    @property
    def mappings(self) -> list[DMXMapping]:
        return list(self._mapping_store.mappings.values())

    def set_managers(
        self,
        pipeline_manager: "PipelineManager",
        webrtc_manager: "WebRTCManager",
    ) -> None:
        self._pipeline_manager = pipeline_manager
        self._webrtc_manager = webrtc_manager

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

    def add_mapping(self, mapping: DMXMapping) -> None:
        """Add or update a channel mapping."""
        self._mapping_store.add(mapping)

    def remove_mapping(self, mapping_id: str) -> bool:
        """Remove a mapping by ID."""
        return self._mapping_store.remove(mapping_id)

    def get_mappings(self) -> list[dict[str, Any]]:
        """Get all mappings as dicts."""
        return [m.to_dict() for m in self._mapping_store.mappings.values()]

    def _handle_artnet_packet(self, data: bytes, addr: tuple[str, int]) -> None:
        """Parse and process an Art-Net packet."""
        # Validate Art-Net header
        if len(data) < 18 or not data.startswith(ARTNET_HEADER):
            return

        # Parse opcode (little-endian)
        opcode = struct.unpack("<H", data[8:10])[0]

        if opcode != ARTNET_OPCODE_DMX:
            return  # We only care about DMX data packets

        # Parse Art-Net DMX packet
        # Bytes 10-11: Protocol version (14)
        # Byte 12: Sequence
        # Byte 13: Physical port
        # Bytes 14-15: Universe (little-endian, but Art-Net spec says low byte first)
        # Bytes 16-17: Length (big-endian)
        # Bytes 18+: DMX data

        universe = struct.unpack("<H", data[14:16])[0]
        length = struct.unpack(">H", data[16:18])[0]

        if len(data) < 18 + length:
            return

        dmx_data = data[18:18 + length]

        self._process_dmx_frame(universe, dmx_data)

    def _process_dmx_frame(self, universe: int, dmx_data: bytes) -> None:
        """Process a DMX frame and apply any mapped parameters."""
        now = time.monotonic()
        updates: dict[str, Any] = {}
        changed_channels: list[dict[str, Any]] = []

        for i, value in enumerate(dmx_data):
            channel = i + 1  # DMX channels are 1-indexed
            cache_key = (universe, channel)

            # Track if value changed (for SSE monitoring)
            old_value = self._channel_values.get(cache_key)
            if old_value != value:
                self._channel_values[cache_key] = value
                changed_channels.append({
                    "universe": universe,
                    "channel": channel,
                    "value": value,
                })

            # Check for mapping
            mapping = self._mapping_store.get_by_channel(universe, channel)
            if mapping:
                scaled_value = mapping.scale(value)
                updates[mapping.param_key] = scaled_value

        # Rate-limit broadcasts
        if updates:
            self._pending_updates.update(updates)

            if now - self._last_broadcast_time >= _MIN_BROADCAST_INTERVAL:
                if self._webrtc_manager and self._pending_updates:
                    self._webrtc_manager.broadcast_parameter_update(self._pending_updates)
                    self._pending_updates = {}
                    self._last_broadcast_time = now

        # Push channel updates to SSE subscribers (for monitoring UI)
        if changed_channels:
            event = {
                "type": "dmx_channels",
                "universe": universe,
                "channels": changed_channels[:50],  # Limit to avoid flooding
            }
            for q in list(self._sse_queues):
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass

    async def start(self) -> None:
        """Start the Art-Net UDP listener."""
        try:
            loop = asyncio.get_running_loop()
            transport, _ = await loop.create_datagram_endpoint(
                lambda: ArtNetProtocol(self),
                local_addr=("0.0.0.0", self._port),
            )
            self._transport = transport
            self._listening = True
            logger.info(f"DMX (Art-Net) server listening on udp://0.0.0.0:{self._port}")
        except OSError as e:
            if e.errno == 98:  # Address already in use
                logger.warning(
                    f"DMX server: Port {self._port} already in use. "
                    "Another application may be using Art-Net."
                )
            else:
                logger.exception(f"Failed to start DMX server on port {self._port}")
            self._listening = False

    async def stop(self) -> None:
        """Stop the DMX server."""
        if self._transport:
            self._transport.close()
            self._transport = None
        self._listening = False
        logger.info("DMX server stopped")

    def status(self) -> dict[str, Any]:
        """Return current server status."""
        return {
            "enabled": True,
            "listening": self._listening,
            "port": self._port,
            "mapping_count": len(self._mapping_store.mappings),
        }
