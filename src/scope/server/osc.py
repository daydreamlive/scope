"""OSC (Open Sound Control) server for real-time parameter control.

Allows external applications (TouchDesigner, Resolume, Max/MSP, MIDI controllers,
etc.) to control Scope's pipeline parameters over UDP using the OSC protocol.

Environment Variables:
    DAYDREAM_SCOPE_OSC: Set to "1" to enable OSC server (default: disabled)
    DAYDREAM_SCOPE_OSC_PORT: UDP port for OSC server (default: 9000)
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .webrtc import WebRTCManager

logger = logging.getLogger(__name__)

DEFAULT_OSC_PORT = 9000


def is_osc_available() -> bool:
    """Check if python-osc is installed."""
    try:
        import pythonosc  # noqa: F401

        return True
    except ImportError:
        return False


def is_osc_enabled() -> bool:
    """Check if OSC is enabled via environment variable or CLI flag."""
    return os.getenv("DAYDREAM_SCOPE_OSC", "").strip() in ("1", "true", "yes")


def get_osc_port() -> int:
    """Get the configured OSC port from environment or default."""
    try:
        return int(os.getenv("DAYDREAM_SCOPE_OSC_PORT", str(DEFAULT_OSC_PORT)))
    except ValueError:
        return DEFAULT_OSC_PORT


class OSCManager:
    """Manages an async OSC UDP server that maps OSC messages to pipeline parameter updates.

    The server listens on a configurable UDP port and dispatches incoming OSC
    messages to handler methods that translate them into Scope parameter updates,
    which are then pushed to all active WebRTC sessions' frame processors.
    """

    def __init__(self, port: int = DEFAULT_OSC_PORT):
        self.port = port
        self._webrtc_manager: WebRTCManager | None = None
        self._server: Any = None
        self._transport: Any = None
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

    def _setup_dispatcher(self):
        """Configure the OSC dispatcher with address handlers."""
        from pythonosc.dispatcher import Dispatcher

        dispatcher = Dispatcher()

        dispatcher.map("/scope/prompt", self._handle_prompt)
        dispatcher.map("/scope/prompt/weight", self._handle_prompt_weight)
        dispatcher.map("/scope/noise", self._handle_noise)
        dispatcher.map("/scope/denoise", self._handle_denoise)
        dispatcher.map("/scope/cache/reset", self._handle_cache_reset)
        dispatcher.map("/scope/cache/bias", self._handle_cache_bias)
        dispatcher.map("/scope/output/*/enable", self._handle_output_enable)
        dispatcher.map("/scope/output/*/disable", self._handle_output_disable)
        dispatcher.map("/scope/param/*", self._handle_generic_param)

        dispatcher.set_default_handler(self._handle_unknown)

        return dispatcher

    # --- OSC Address Handlers ---

    def _handle_prompt(self, address: str, *args) -> None:
        """Handle /scope/prompt <string> -- set the first prompt text."""
        if not args or not isinstance(args[0], str):
            logger.warning(f"OSC {address}: expected string argument, got {args}")
            return

        prompt_text = args[0]
        logger.info(f"OSC prompt: {prompt_text[:80]}...")
        self._push_parameters({"prompts": [{"text": prompt_text, "weight": 100}]})

    def _handle_prompt_weight(self, address: str, *args) -> None:
        """Handle /scope/prompt/weight <float> -- set first prompt weight."""
        if not args:
            return
        try:
            weight = float(args[0])
            weight = max(0.0, min(100.0, weight))
            logger.info(f"OSC prompt weight: {weight}")
            self._push_parameters({"prompts": [{"text": "", "weight": weight}]})
        except (ValueError, TypeError):
            logger.warning(f"OSC {address}: invalid weight value: {args[0]}")

    def _handle_noise(self, address: str, *args) -> None:
        """Handle /scope/noise <float 0.0-1.0> -- set noise scale."""
        if not args:
            return
        try:
            noise = float(args[0])
            noise = max(0.0, min(1.0, noise))
            logger.info(f"OSC noise scale: {noise}")
            self._push_parameters({"noise_scale": noise})
        except (ValueError, TypeError):
            logger.warning(f"OSC {address}: invalid noise value: {args[0]}")

    def _handle_denoise(self, address: str, *args) -> None:
        """Handle /scope/denoise <int> [<int>...] -- set denoising step list."""
        if not args:
            return
        try:
            steps = [int(a) for a in args]
            logger.info(f"OSC denoising steps: {steps}")
            self._push_parameters({"denoising_step_list": steps})
        except (ValueError, TypeError):
            logger.warning(f"OSC {address}: invalid step values: {args}")

    def _handle_cache_reset(self, address: str, *args) -> None:
        """Handle /scope/cache/reset -- trigger cache reset."""
        logger.info("OSC cache reset")
        self._push_parameters({"reset_cache": True})

    def _handle_cache_bias(self, address: str, *args) -> None:
        """Handle /scope/cache/bias <float 0.01-1.0> -- set KV cache attention bias."""
        if not args:
            return
        try:
            bias = float(args[0])
            bias = max(0.01, min(1.0, bias))
            logger.info(f"OSC cache bias: {bias}")
            self._push_parameters({"kv_cache_attention_bias": bias})
        except (ValueError, TypeError):
            logger.warning(f"OSC {address}: invalid bias value: {args[0]}")

    def _handle_output_enable(self, address: str, *args) -> None:
        """Handle /scope/output/<sink_type>/enable -- enable an output sink."""
        parts = address.split("/")
        if len(parts) < 5:
            return
        sink_type = parts[3]
        name = args[0] if args and isinstance(args[0], str) else "ScopeOSC"
        logger.info(f"OSC enable output: {sink_type} (name={name})")
        self._push_parameters(
            {"output_sinks": {sink_type: {"enabled": True, "name": name}}}
        )

    def _handle_output_disable(self, address: str, *args) -> None:
        """Handle /scope/output/<sink_type>/disable -- disable an output sink."""
        parts = address.split("/")
        if len(parts) < 5:
            return
        sink_type = parts[3]
        logger.info(f"OSC disable output: {sink_type}")
        self._push_parameters(
            {"output_sinks": {sink_type: {"enabled": False, "name": ""}}}
        )

    def _handle_generic_param(self, address: str, *args) -> None:
        """Handle /scope/param/<key> <value> -- generic parameter passthrough."""
        parts = address.split("/")
        if len(parts) < 4:
            return
        key = parts[3]
        if not args:
            logger.warning(f"OSC {address}: no value provided")
            return

        value = args[0] if len(args) == 1 else list(args)
        logger.info(f"OSC param: {key} = {value}")
        self._push_parameters({key: value})

    def _handle_unknown(self, address: str, *args) -> None:
        """Log unrecognized OSC addresses for debugging."""
        logger.debug(f"OSC unknown address: {address} args={args}")

    # --- Lifecycle ---

    async def start(self, webrtc_manager: WebRTCManager) -> bool:
        """Start the OSC UDP server.

        Returns True if started successfully, False otherwise.
        """
        if not is_osc_available():
            logger.warning("python-osc is not installed, OSC server cannot start")
            return False

        self._webrtc_manager = webrtc_manager

        try:
            from pythonosc.osc_server import AsyncIOOSCUDPServer

            dispatcher = self._setup_dispatcher()
            self._server = AsyncIOOSCUDPServer(
                ("0.0.0.0", self.port),
                dispatcher,
                asyncio.get_event_loop(),
            )
            self._transport, _ = await self._server.create_serve_endpoint()
            self._listening = True
            logger.info(f"OSC server listening on UDP port {self.port}")
            return True
        except OSError as e:
            logger.error(f"Failed to start OSC server on port {self.port}: {e}")
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
