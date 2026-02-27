"""OSC (Open Sound Control) server for MIDI/controller integration.

This module provides an optional OSC server that can receive parameter updates
from external controllers and inject them directly into the FrameProcessor.
"""

import asyncio
import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Lazy import - python-osc is optional
_osc_available = False
try:
    from pythonosc import dispatcher
    from pythonosc import osc_server

    _osc_available = True
except ImportError:
    logger.warning(
        "python-osc not available. Install with: pip install python-osc"
    )


class OSCServer:
    """OSC server that maps OSC address patterns to parameter updates."""

    def __init__(
        self,
        port: int = 8000,
        get_frame_processor: Callable[[], Any] | None = None,
    ):
        """Initialize OSC server.

        Args:
            port: UDP port to listen on (default: 8000)
            get_frame_processor: Callable that returns the active FrameProcessor instance
        """
        if not _osc_available:
            raise RuntimeError(
                "python-osc is not installed. Install with: pip install python-osc"
            )

        self.port = port
        self.get_frame_processor = get_frame_processor
        self._server: osc_server.ThreadingOSCUDPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._running = False

    def _handle_parameter(self, address: str, *args: Any) -> None:
        """Handle OSC message for a parameter.

        Address format: /scope/<parameter_name>
        Example: /scope/noise_scale with float value
        """
        if not address.startswith("/scope/"):
            return

        parameter_name = address[len("/scope/") :]
        if not parameter_name:
            return

        # Extract value from args
        if len(args) == 0:
            logger.warning(f"OSC message for {address} has no value")
            return

        value = args[0]

        # Normalize parameter name (OSC uses /, we use snake_case)
        # Already normalized since we strip /scope/

        update = {parameter_name: value}

        # Inject update into active FrameProcessor
        if self.get_frame_processor:
            try:
                frame_processor = self.get_frame_processor()
                if frame_processor:
                    frame_processor.update_parameters(update)
                    logger.debug(f"OSC update: {parameter_name} = {value}")
                else:
                    logger.warning("No active FrameProcessor for OSC update")
            except Exception as e:
                logger.error(f"Error applying OSC update: {e}")

    def start(self) -> None:
        """Start the OSC server in a background thread."""
        if self._running:
            logger.warning("OSC server is already running")
            return

        if not _osc_available:
            logger.error("Cannot start OSC server: python-osc not available")
            return

        # Create dispatcher
        disp = dispatcher.Dispatcher()
        disp.map("/scope/*", self._handle_parameter)

        # Create server
        try:
            self._server = osc_server.ThreadingOSCUDPServer(
                ("127.0.0.1", self.port), disp
            )
            logger.info(f"Starting OSC server on port {self.port}")

            # Start server in background thread
            self._server_thread = threading.Thread(
                target=self._server.serve_forever, daemon=True
            )
            self._server_thread.start()
            self._running = True

            logger.info(f"OSC server started on port {self.port}")
        except OSError as e:
            logger.error(f"Failed to start OSC server: {e}")
            raise

    def stop(self) -> None:
        """Stop the OSC server."""
        if not self._running:
            return

        if self._server:
            self._server.shutdown()
            self._server = None

        if self._server_thread:
            self._server_thread.join(timeout=1.0)
            self._server_thread = None

        self._running = False
        logger.info("OSC server stopped")

    def is_running(self) -> bool:
        """Check if the server is running."""
        return self._running


# Global OSC server instance (singleton)
_osc_server_instance: OSCServer | None = None


def get_osc_server() -> OSCServer | None:
    """Get the global OSC server instance."""
    return _osc_server_instance


def start_osc_server(
    port: int = 8000,
    get_frame_processor: Callable[[], Any] | None = None,
) -> OSCServer:
    """Start the global OSC server.

    Args:
        port: UDP port to listen on
        get_frame_processor: Callable that returns the active FrameProcessor instance

    Returns:
        OSCServer instance
    """
    global _osc_server_instance

    if _osc_server_instance is not None:
        _osc_server_instance.stop()

    if not _osc_available:
        logger.warning("OSC server not available (python-osc not installed)")
        return None  # type: ignore

    _osc_server_instance = OSCServer(
        port=port, get_frame_processor=get_frame_processor
    )
    _osc_server_instance.start()
    return _osc_server_instance


def stop_osc_server() -> None:
    """Stop the global OSC server."""
    global _osc_server_instance

    if _osc_server_instance:
        _osc_server_instance.stop()
        _osc_server_instance = None
