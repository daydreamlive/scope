"""
Syphon Receiver - Receives textures from Syphon servers on macOS.

This module provides a simple interface for receiving textures from
Syphon-compatible applications like TouchDesigner, Resolume, etc.

Uses the syphon-python library which wraps Syphon.framework via PyObjC.

Raises:
    ImportError: If syphon-python is not installed.
        Install with: pip install syphon-python
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any

import Metal
import numpy as np
import objc
import syphon
from syphon.server_directory import SyphonServerDescription
from syphon.utils.numpy import copy_mtl_texture_to_image

logger = logging.getLogger(__name__)

# Syphon server discovery uses NSDistributedNotificationCenter which delivers
# notifications exclusively to the main thread's NSRunLoop.  In a FastAPI /
# uvicorn server the main thread runs an asyncio event-loop, so we must
# explicitly pump the NSRunLoop on the main thread for server announcements to
# be received.  ensure_directory_initialized() creates the ObjC singleton
# (registering its notification observers) and pump_run_loop() must be called
# periodically on the main thread to keep the server list up-to-date.
_directory_initialized = False
_directory_lock = threading.Lock()
_SyphonServerDirectoryObjC = objc.lookUpClass("SyphonServerDirectory")


def ensure_directory_initialized():
    """Create the SyphonServerDirectory ObjC singleton.

    Safe to call from any thread.  The singleton registers its notification
    observers on NSDistributedNotificationCenter which delivers to the main
    thread regardless of which thread creates the singleton.

    After calling this, ``pump_run_loop()`` must be called on the **main
    thread** at least once for server announcements to be received.
    """
    global _directory_initialized
    if _directory_initialized:
        return
    with _directory_lock:
        if _directory_initialized:
            return
        _SyphonServerDirectoryObjC.sharedDirectory()
        _directory_initialized = True
        logger.debug("Syphon shared directory singleton created")


def pump_run_loop(duration_s: float = 0.1):
    """Pump the current thread's NSRunLoop to receive Syphon notifications.

    **Must be called on the main thread** — Syphon server announcements are
    delivered via NSDistributedNotificationCenter which only fires on the
    main thread's run-loop.
    """
    from Cocoa import NSDate, NSDefaultRunLoopMode, NSRunLoop

    NSRunLoop.currentRunLoop().runMode_beforeDate_(
        NSDefaultRunLoopMode,
        NSDate.dateWithTimeIntervalSinceNow_(duration_s),
    )


def _get_raw_servers() -> list:
    """Read the current server list from the ObjC shared directory."""
    shared = _SyphonServerDirectoryObjC.sharedDirectory()
    return list(shared.servers() or [])


def _raw_to_description(raw: Any) -> SyphonServerDescription:
    from Cocoa import NSImage

    return SyphonServerDescription(
        uuid=str(raw.get("SyphonServerDescriptionUUIDKey", "")),
        name=str(raw.get("SyphonServerDescriptionNameKey", "")),
        app_name=str(raw.get("SyphonServerDescriptionAppNameKey", "")),
        icon=raw.get("SyphonServerDescriptionIconKey", NSImage.alloc().init()),
        raw=raw,
    )


@dataclass
class SyphonServerInfo:
    """Information about a discovered Syphon server."""

    uuid: str
    name: str
    app_name: str

    @property
    def display_name(self) -> str:
        parts = [p for p in (self.app_name, self.name) if p]
        return " - ".join(parts) if parts else self.uuid


class SyphonReceiver:
    """
    Receives textures from a Syphon server.

    Example usage:
        receiver = SyphonReceiver()
        servers = receiver.discover()
        if servers:
            receiver.connect(servers[0].uuid)
            while running:
                frame = receiver.receive()
                if frame is not None:
                    # Process frame (H, W, 3) numpy array in [0, 255] uint8
                    pass
            receiver.release()
    """

    def __init__(self):
        self._client: Any | None = None
        self._connected_server: SyphonServerInfo | None = None
        self._frame_count = 0

    def discover(self) -> list[SyphonServerInfo]:
        """Discover all available Syphon servers on the system."""
        try:
            raw_servers = _get_raw_servers()
            return [
                SyphonServerInfo(
                    uuid=str(s.get("SyphonServerDescriptionUUIDKey", "")),
                    name=str(s.get("SyphonServerDescriptionNameKey", "")),
                    app_name=str(s.get("SyphonServerDescriptionAppNameKey", "")),
                )
                for s in raw_servers
            ]
        except Exception as e:
            logger.warning(f"Could not discover Syphon servers: {e}", exc_info=True)
            return []

    def connect(self, identifier: str) -> bool:
        """Connect to a Syphon server by UUID or app_name.

        Args:
            identifier: UUID or app_name of the Syphon server to connect to.
                        Empty string connects to the first available server.

        Returns:
            True if connection was successful.
        """
        try:
            self.release()

            raw_servers = _get_raw_servers()
            if not raw_servers:
                logger.warning("No Syphon servers available")
                return False

            target_raw = None
            if not identifier:
                target_raw = raw_servers[0]
            else:
                for s in raw_servers:
                    uuid = str(s.get("SyphonServerDescriptionUUIDKey", ""))
                    app = str(s.get("SyphonServerDescriptionAppNameKey", ""))
                    if uuid == identifier or app == identifier:
                        target_raw = s
                        break

            if target_raw is None:
                logger.warning(f"Syphon server not found: '{identifier}'")
                return False

            desc = _raw_to_description(target_raw)
            self._client = syphon.SyphonMetalClient(desc)
            if not self._client.is_valid:
                logger.error("SyphonMetalClient created but not valid")
                self._client = None
                return False

            self._connected_server = SyphonServerInfo(
                uuid=desc.uuid,
                name=desc.name,
                app_name=desc.app_name,
            )
            logger.info(
                f"SyphonReceiver connected to '{self._connected_server.display_name}'"
            )
            return True

        except Exception as e:
            logger.error(f"Error connecting SyphonReceiver: {e}")
            self._client = None
            return False

    def receive(self, as_rgb: bool = False) -> np.ndarray | None:
        """Receive a frame from the Syphon server.

        Args:
            as_rgb: If True, return RGB (3 channels) instead of RGBA (4 channels).

        Returns:
            Frame as numpy array (H, W, C) in uint8 [0, 255] format,
            or None if no frame is available.
        """
        if self._client is None:
            return None

        try:
            if not self._client.has_new_frame:
                return None

            texture = self._client.new_frame_image
            if texture is None:
                return None

            image = copy_mtl_texture_to_image(texture)
            self._frame_count += 1

            # Handle BGRA pixel format (common macOS default)
            pixel_format = texture.pixelFormat()
            if pixel_format == Metal.MTLPixelFormatBGRA8Unorm:
                image = image[:, :, [2, 1, 0, 3]]

            if as_rgb:
                return np.ascontiguousarray(image[:, :, :3])
            else:
                return image.copy()

        except Exception as e:
            logger.error(f"Error receiving Syphon frame: {e}")
            return None

    def is_connected(self) -> bool:
        return self._client is not None and self._client.is_valid

    def get_server_name(self) -> str:
        if self._connected_server:
            return self._connected_server.display_name
        return ""

    def get_frame_count(self) -> int:
        return self._frame_count

    def release(self):
        """Release the Syphon client resources."""
        if self._client is not None:
            try:
                self._client.stop()
                logger.info("SyphonReceiver released")
            except Exception as e:
                logger.error(f"Error releasing SyphonReceiver: {e}")
            finally:
                self._client = None
                self._connected_server = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
