"""Art-Net DMX server for external parameter control and fixture output.

Supports both DMX In (console → Scope) and DMX Out (Scope → fixtures).
Art-Net uses UDP port 6454 by default. Each universe contains 512 channels
with 8-bit values (0-255).
"""

import asyncio
import json
import logging
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
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
ARTNET_OPCODE_POLL = 0x2000
ARTNET_OPCODE_POLL_REPLY = 0x2100


class MergeMode(str, Enum):
    """DMX merge mode for output."""

    HTP = "htp"  # Highest Takes Precedence (safer default)
    LTP = "ltp"  # Latest Takes Precedence


class ParameterCategory(str, Enum):
    """Categories for parameter grouping in UI."""

    GENERATION = "generation"
    LORA = "lora"
    COLOR = "color"
    ANALYSIS = "analysis"


@dataclass
class DMXInputMapping:
    """Maps a DMX input channel to a Scope parameter."""

    id: str
    universe: int
    channel: int  # 1-512 (DMX convention, 1-indexed)
    param_key: str
    category: ParameterCategory
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
            "category": self.category.value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DMXInputMapping":
        return cls(
            id=data["id"],
            universe=data["universe"],
            channel=data["channel"],
            param_key=data["param_key"],
            category=ParameterCategory(data.get("category", "generation")),
            min_value=data.get("min_value", 0.0),
            max_value=data.get("max_value", 1.0),
            enabled=data.get("enabled", True),
        )


@dataclass
class DMXOutputMapping:
    """Maps a Scope analysis value to a DMX output channel."""

    id: str
    universe: int
    channel: int  # 1-512
    source_key: str  # e.g., "color_r", "motion", "brightness", "beat"
    category: ParameterCategory
    min_value: float = 0.0  # Source value that maps to DMX 0
    max_value: float = 1.0  # Source value that maps to DMX 255
    enabled: bool = True

    def scale(self, value: float) -> int:
        """Convert source value to 0-255 DMX value."""
        # Clamp and normalize
        clamped = max(self.min_value, min(self.max_value, value))
        if self.max_value == self.min_value:
            normalized = 0.0
        else:
            normalized = (clamped - self.min_value) / (self.max_value - self.min_value)
        return int(normalized * 255)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "universe": self.universe,
            "channel": self.channel,
            "source_key": self.source_key,
            "category": self.category.value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DMXOutputMapping":
        return cls(
            id=data["id"],
            universe=data["universe"],
            channel=data["channel"],
            source_key=data["source_key"],
            category=ParameterCategory(data.get("category", "analysis")),
            min_value=data.get("min_value", 0.0),
            max_value=data.get("max_value", 1.0),
            enabled=data.get("enabled", True),
        )


@dataclass
class DMXConfig:
    """DMX configuration with input/output mappings."""

    input_mappings: dict[str, DMXInputMapping] = field(default_factory=dict)
    output_mappings: dict[str, DMXOutputMapping] = field(default_factory=dict)
    input_universe: int = 0
    input_start_channel: int = 1
    output_universe: int = 0
    output_enabled: bool = False  # Safe default: don't send until explicitly enabled
    output_merge_mode: MergeMode = MergeMode.HTP
    _config_path: Path | None = None

    @classmethod
    def load(cls, config_dir: Path) -> "DMXConfig":
        """Load config from file."""
        config_path = config_dir / "dmx_config.json"
        config = cls(_config_path=config_path)

        if config_path.exists():
            try:
                data = json.loads(config_path.read_text())

                for mapping_data in data.get("input_mappings", []):
                    mapping = DMXInputMapping.from_dict(mapping_data)
                    config.input_mappings[mapping.id] = mapping

                for mapping_data in data.get("output_mappings", []):
                    mapping = DMXOutputMapping.from_dict(mapping_data)
                    config.output_mappings[mapping.id] = mapping

                config.input_universe = data.get("input_universe", 0)
                config.input_start_channel = data.get("input_start_channel", 1)
                config.output_universe = data.get("output_universe", 0)
                config.output_enabled = data.get("output_enabled", False)
                config.output_merge_mode = MergeMode(
                    data.get("output_merge_mode", "htp")
                )

                logger.info(
                    f"Loaded DMX config: {len(config.input_mappings)} input mappings, "
                    f"{len(config.output_mappings)} output mappings"
                )
            except Exception as e:
                logger.warning(f"Failed to load DMX config: {e}")

        return config

    def save(self) -> None:
        """Save config to file."""
        if self._config_path is None:
            return

        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "input_mappings": [m.to_dict() for m in self.input_mappings.values()],
                "output_mappings": [m.to_dict() for m in self.output_mappings.values()],
                "input_universe": self.input_universe,
                "input_start_channel": self.input_start_channel,
                "output_universe": self.output_universe,
                "output_enabled": self.output_enabled,
                "output_merge_mode": self.output_merge_mode.value,
            }
            self._config_path.write_text(json.dumps(data, indent=2))
            logger.debug("Saved DMX config")
        except Exception as e:
            logger.error(f"Failed to save DMX config: {e}")


class ArtNetProtocol(asyncio.DatagramProtocol):
    """UDP protocol handler for Art-Net packets."""

    def __init__(self, server: "DMXServer"):
        self._server = server
        self._transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        self._transport = transport

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        self._server._handle_artnet_packet(data, addr)


class DMXServer:
    """Manages Art-Net DMX input and output."""

    def __init__(self, port: int = ARTNET_PORT, config_dir: Path | None = None):
        self._port = port
        self._transport: asyncio.DatagramTransport | None = None
        self._listening = False
        self._input_active = False  # True when receiving signal
        self._last_signal_time: float = 0.0
        self._pipeline_manager: PipelineManager | None = None
        self._webrtc_manager: WebRTCManager | None = None

        # SSE subscribers for real-time monitoring
        self._sse_queues: list[asyncio.Queue] = []

        # Configuration
        if config_dir is None:
            config_dir = Path.home() / ".daydream-scope"
        self._config = DMXConfig.load(config_dir)

        # Rate limiting for parameter broadcasts
        self._last_broadcast_time: float = 0.0
        self._pending_updates: dict[str, Any] = {}
        self._min_broadcast_interval = 0.016  # ~60fps max

        # Cache for channel values (input monitoring)
        self._input_channel_values: dict[tuple[int, int], int] = {}

        # Output state
        self._output_values: dict[tuple[int, int], int] = {}  # (universe, channel) → value

        # Analysis values from pipeline (for DMX Out)
        self._analysis_values: dict[str, float] = {}

    @property
    def port(self) -> int:
        return self._port

    @property
    def listening(self) -> bool:
        return self._listening

    @property
    def input_active(self) -> bool:
        """True if receiving Art-Net signal recently."""
        if not self._listening:
            return False
        return time.monotonic() - self._last_signal_time < 5.0

    @property
    def config(self) -> DMXConfig:
        return self._config

    def set_managers(
        self,
        pipeline_manager: PipelineManager,
        webrtc_manager: WebRTCManager,
    ) -> None:
        self._pipeline_manager = pipeline_manager
        self._webrtc_manager = webrtc_manager

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Register a new SSE subscriber."""
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._sse_queues.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Deregister an SSE subscriber."""
        try:
            self._sse_queues.remove(q)
        except ValueError:
            pass

    # -------------------------------------------------------------------------
    # Configuration API
    # -------------------------------------------------------------------------

    def get_config(self) -> dict[str, Any]:
        """Get current configuration."""
        return {
            "input_mappings": [m.to_dict() for m in self._config.input_mappings.values()],
            "output_mappings": [m.to_dict() for m in self._config.output_mappings.values()],
            "input_universe": self._config.input_universe,
            "input_start_channel": self._config.input_start_channel,
            "output_universe": self._config.output_universe,
            "output_enabled": self._config.output_enabled,
            "output_merge_mode": self._config.output_merge_mode.value,
        }

    def update_config(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Update configuration fields."""
        if "input_universe" in updates:
            self._config.input_universe = int(updates["input_universe"])
        if "input_start_channel" in updates:
            ch = int(updates["input_start_channel"])
            if 1 <= ch <= 512:
                self._config.input_start_channel = ch
        if "output_universe" in updates:
            self._config.output_universe = int(updates["output_universe"])
        if "output_enabled" in updates:
            self._config.output_enabled = bool(updates["output_enabled"])
        if "output_merge_mode" in updates:
            self._config.output_merge_mode = MergeMode(updates["output_merge_mode"])

        self._config.save()
        return self.get_config()

    def add_input_mapping(self, mapping: DMXInputMapping) -> None:
        """Add or update an input mapping."""
        self._config.input_mappings[mapping.id] = mapping
        self._config.save()

    def remove_input_mapping(self, mapping_id: str) -> bool:
        """Remove an input mapping."""
        if mapping_id in self._config.input_mappings:
            del self._config.input_mappings[mapping_id]
            self._config.save()
            return True
        return False

    def add_output_mapping(self, mapping: DMXOutputMapping) -> None:
        """Add or update an output mapping."""
        self._config.output_mappings[mapping.id] = mapping
        self._config.save()

    def remove_output_mapping(self, mapping_id: str) -> bool:
        """Remove an output mapping."""
        if mapping_id in self._config.output_mappings:
            del self._config.output_mappings[mapping_id]
            self._config.save()
            return True
        return False

    # -------------------------------------------------------------------------
    # Input handling
    # -------------------------------------------------------------------------

    def _handle_artnet_packet(self, data: bytes, addr: tuple[str, int]) -> None:
        """Parse and process an Art-Net packet."""
        if len(data) < 12 or not data.startswith(ARTNET_HEADER):
            return

        opcode = struct.unpack("<H", data[8:10])[0]

        if opcode == ARTNET_OPCODE_DMX:
            self._handle_dmx_packet(data)
        # Could handle ARTNET_OPCODE_POLL for discovery in future

    def _handle_dmx_packet(self, data: bytes) -> None:
        """Process Art-Net DMX packet."""
        if len(data) < 18:
            return

        # Parse header
        # Bytes 14-15: Universe (little-endian, subnet << 4 | universe)
        # Bytes 16-17: Length (big-endian)
        universe = struct.unpack("<H", data[14:16])[0]
        length = struct.unpack(">H", data[16:18])[0]

        if len(data) < 18 + length:
            return

        dmx_data = data[18 : 18 + length]
        self._last_signal_time = time.monotonic()
        self._input_active = True

        self._process_dmx_input(universe, dmx_data)

    def _process_dmx_input(self, universe: int, dmx_data: bytes) -> None:
        """Process incoming DMX frame and apply mappings."""
        now = time.monotonic()
        updates: dict[str, Any] = {}
        changed_channels: list[dict[str, Any]] = []

        # Check configured input universe
        if universe != self._config.input_universe:
            return

        for mapping in self._config.input_mappings.values():
            if not mapping.enabled:
                continue
            if mapping.universe != universe:
                continue

            idx = mapping.channel - 1  # 0-indexed
            if idx < 0 or idx >= len(dmx_data):
                continue

            value = dmx_data[idx]
            cache_key = (universe, mapping.channel)

            # Track changes for monitoring
            old_value = self._input_channel_values.get(cache_key)
            if old_value != value:
                self._input_channel_values[cache_key] = value
                changed_channels.append(
                    {
                        "universe": universe,
                        "channel": mapping.channel,
                        "value": value,
                        "param": mapping.param_key,
                    }
                )

            # Scale and queue parameter update
            scaled = mapping.scale(value)
            updates[mapping.param_key] = scaled

        # Rate-limited broadcast to pipeline
        if updates:
            self._pending_updates.update(updates)

            if now - self._last_broadcast_time >= self._min_broadcast_interval:
                if self._webrtc_manager and self._pending_updates:
                    self._webrtc_manager.broadcast_parameter_update(self._pending_updates)
                    self._pending_updates = {}
                    self._last_broadcast_time = now

        # Push to SSE subscribers for UI monitoring
        if changed_channels:
            event = {
                "type": "dmx_input",
                "universe": universe,
                "channels": changed_channels[:50],
            }
            for q in list(self._sse_queues):
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass

    # -------------------------------------------------------------------------
    # Output handling
    # -------------------------------------------------------------------------

    def update_analysis_values(self, values: dict[str, float]) -> None:
        """Update analysis values from pipeline (for DMX Out)."""
        self._analysis_values.update(values)

        if self._config.output_enabled and self._config.output_mappings:
            self._send_dmx_output()

    def _send_dmx_output(self) -> None:
        """Send DMX output based on analysis values and mappings."""
        if not self._transport or not self._config.output_enabled:
            return

        # Build universe data
        universes: dict[int, bytearray] = {}

        for mapping in self._config.output_mappings.values():
            if not mapping.enabled:
                continue

            source_value = self._analysis_values.get(mapping.source_key, 0.0)
            dmx_value = mapping.scale(source_value)

            universe = mapping.universe
            if universe not in universes:
                universes[universe] = bytearray(512)

            idx = mapping.channel - 1
            if 0 <= idx < 512:
                current = universes[universe][idx]
                if self._config.output_merge_mode == MergeMode.HTP:
                    universes[universe][idx] = max(current, dmx_value)
                else:  # LTP
                    universes[universe][idx] = dmx_value

        # Send Art-Net packets
        for universe, data in universes.items():
            packet = self._build_artnet_dmx_packet(universe, bytes(data))
            # Broadcast to network
            self._transport.sendto(packet, ("255.255.255.255", ARTNET_PORT))

    def _build_artnet_dmx_packet(self, universe: int, dmx_data: bytes) -> bytes:
        """Build an Art-Net DMX packet."""
        header = ARTNET_HEADER
        opcode = struct.pack("<H", ARTNET_OPCODE_DMX)
        protocol_version = struct.pack(">H", 14)  # Art-Net protocol version
        sequence = b"\x00"  # Sequence number (0 = disabled)
        physical = b"\x00"  # Physical port
        universe_bytes = struct.pack("<H", universe)
        length = struct.pack(">H", len(dmx_data))

        return header + opcode + protocol_version + sequence + physical + universe_bytes + length + dmx_data

    def test_output(self) -> None:
        """Send a test ramp (0→255→0) over 2 seconds."""
        if not self._transport or not self._config.output_enabled:
            return

        async def _ramp():
            for i in range(256):
                await asyncio.sleep(2.0 / 512)
                self._send_test_value(i)
            for i in range(255, -1, -1):
                await asyncio.sleep(2.0 / 512)
                self._send_test_value(i)

        asyncio.create_task(_ramp())

    def _send_test_value(self, value: int) -> None:
        """Send test value to all mapped output channels."""
        if not self._transport:
            return

        universes: dict[int, bytearray] = {}
        for mapping in self._config.output_mappings.values():
            if not mapping.enabled:
                continue
            universe = mapping.universe
            if universe not in universes:
                universes[universe] = bytearray(512)
            idx = mapping.channel - 1
            if 0 <= idx < 512:
                universes[universe][idx] = value

        for universe, data in universes.items():
            packet = self._build_artnet_dmx_packet(universe, bytes(data))
            self._transport.sendto(packet, ("255.255.255.255", ARTNET_PORT))

    # -------------------------------------------------------------------------
    # Server lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start the Art-Net UDP server."""
        try:
            loop = asyncio.get_running_loop()
            transport, _ = await loop.create_datagram_endpoint(
                lambda: ArtNetProtocol(self),
                local_addr=("0.0.0.0", self._port),
                allow_broadcast=True,
            )
            self._transport = transport
            self._listening = True
            logger.info(f"DMX (Art-Net) server listening on UDP port {self._port}")
        except OSError as e:
            if e.errno == 98:  # Address already in use
                logger.warning(
                    f"DMX server: Port {self._port} already in use. "
                    "Another application may be using Art-Net."
                )
            else:
                logger.exception(f"Failed to start DMX server: {e}")
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
            "input_active": self.input_active,
            "input_mapping_count": len(self._config.input_mappings),
            "output_enabled": self._config.output_enabled,
            "output_mapping_count": len(self._config.output_mappings),
            "output_merge_mode": self._config.output_merge_mode.value,
        }
