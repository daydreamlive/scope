"""LivepeerConnection for relay mode.

This manager mirrors the subset of CloudConnectionManager's interface used by
CloudTrack/FrameProcessor, but routes media and control over Livepeer.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from av import AudioFrame, VideoFrame

from .livepeer_client import LivepeerClient

logger = logging.getLogger(__name__)

# --- LIVEPEER CONFIGURATION ---
# The token is a base64-encoded JSON payload consumed by livepeer_gateway.
# It can include signer/discovery URLs and optional headers.
LIVEPEER_TOKEN_ENV = "LIVEPEER_TOKEN"
LIVEPEER_MODEL_ID = "scope"

if os.getenv("LIVEPEER_DEBUG"):
    logging.getLogger("livepeer_gateway").setLevel(logging.DEBUG)
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    logging.getLogger("scope.server.livepeer_client").setLevel(logging.DEBUG)


def is_livepeer_enabled() -> bool:
    """Check if Livepeer mode is enabled via environment variables."""
    return os.getenv("SCOPE_CLOUD_MODE", "").lower() == "livepeer"


class LivepeerConnection:
    """Manager for Livepeer-backed relay mode.

    Lifecycle model:
    - configure(): mark backend as available for relay mode
    - start_webrtc(): start an actual Livepeer job (called when stream starts)
    - stop_webrtc(): stop the active Livepeer job (called when stream ends)
    """

    def __init__(self):
        self._configured = False
        self._connecting = False
        self._connect_task: asyncio.Task | None = None
        self._connect_error: str | None = None
        self._client: LivepeerClient | None = None
        self._frame_callbacks: list[Callable[[VideoFrame], None]] = []
        self._audio_callbacks: list[Callable[[AudioFrame], None]] = []
        self._last_close_code: int | None = None
        self._last_close_reason: str | None = None
        self._user_id: str | None = None

        self._stats = {
            "connected_at": None,
        }

    @property
    def is_connected(self) -> bool:
        """Whether Livepeer backend is configured/available for relay mode."""
        return (
            self._configured and self._client is not None and self._client.is_connected
        )

    @property
    def webrtc_connected(self) -> bool:
        """Whether an active Livepeer job is running."""
        return self._client is not None and self._client.media_connected

    def configure(self) -> None:
        """Enable Livepeer backend mode."""
        self._configured = True
        self._connect_error = None
        self._last_close_code = None
        self._last_close_reason = None
        logger.info("Backend configured")

    async def connect(
        self,
        app_id: str | None = None,
        api_key: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Create and connect a persistent Livepeer job."""
        # Keep connect signature compatible with cloud-style connect requests.
        # app_id can be used as optional runner routing config (derived into a
        # fal ws_url in the client). api_key is forwarded so Livepeer startup can
        # include Daydream signer metadata.
        self._user_id = user_id

        if self.is_connected:
            self._connecting = False
            return

        if not self._configured:
            self.configure()

        # TODO Encode signer URL + Daydream API key in token; env var takes precedence
        token = os.environ.get(LIVEPEER_TOKEN_ENV, "e30K")
        if not token:
            raise RuntimeError(
                f"Livepeer token not configured. Set the {LIVEPEER_TOKEN_ENV} environment variable."
            )

        self._connecting = True
        self._connect_error = None
        self._last_close_code = None
        self._last_close_reason = None

        client = LivepeerClient(
            token=token,
            model_id=LIVEPEER_MODEL_ID,
            app_id=app_id,
            api_key=api_key,
        )

        try:
            connect_params: dict[str, Any] = {}
            if self._user_id:
                connect_params["daydream_user_id"] = self._user_id
            await client.connect(initial_parameters=connect_params)
            self._client = client
            self._stats["connected_at"] = time.time()
            logger.info("Livepeer connected")
        except Exception as e:
            self._connect_error = str(e)
            self._last_close_reason = str(e)
            logger.error(f"Failed to connect job: {e}")
            try:
                await client.disconnect()
            except Exception:
                pass
            raise
        finally:
            self._connecting = False

    async def connect_background(
        self,
        app_id: str | None = None,
        api_key: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Start Livepeer connection in the background."""
        logger.info("Cloud connect requested in Livepeer mode")
        if not self._configured:
            self.configure()

        if self._connect_task is not None and not self._connect_task.done():
            self._connect_task.cancel()
            try:
                await self._connect_task
            except (asyncio.CancelledError, Exception):
                pass

        async def _do_connect():
            try:
                await self.connect(app_id=app_id, api_key=api_key, user_id=user_id)
            except Exception as e:
                self._connecting = False
                self._connect_error = str(e)
                self._last_close_reason = str(e)
                logger.exception(f"Background connect failed: {e}")

        self._connecting = True
        self._connect_error = None
        self._connect_task = asyncio.create_task(_do_connect())

    async def start_webrtc(self, initial_parameters: dict | None = None) -> None:
        """Start media channels for the existing Livepeer job."""
        if self._client is None or not self._client.is_connected:
            raise RuntimeError("Livepeer backend is not connected")
        await self._client.start_media(initial_parameters=initial_parameters)
        # Register callbacks after start_media() so handlers are initialized.
        self._client.output_handlers[0].add_callback(self._on_frame_from_livepeer)
        self._client.audio_output_handler.add_callback(self._on_audio_from_livepeer)

    async def stop_webrtc(self) -> None:
        """Stop active media channels for the current Livepeer job."""
        if self._client is None:
            return
        await self._client.stop_media()
        logger.info("Media stopped")

    async def disconnect(self) -> None:
        """Disable Livepeer mode and stop any active job."""
        logger.info("Cloud disconnect requested in Livepeer mode")
        if self._connect_task is not None and not self._connect_task.done():
            self._connect_task.cancel()
            try:
                await self._connect_task
            except (asyncio.CancelledError, Exception):
                pass
            self._connect_task = None

        if self._client is not None:
            try:
                await asyncio.wait_for(self._client.disconnect(), timeout=15.0)
            except TimeoutError:
                logger.warning(
                    "Livepeer client disconnect timed out after 15s, forcing cleanup"
                )
            except Exception as e:
                logger.warning(f"Error during Livepeer client disconnect: {e}")
            self._client = None
        self._configured = False
        self._connecting = False
        logger.info("Backend disconnected")

    def send_frame(self, frame: VideoFrame | np.ndarray) -> bool:
        if self._client is None or not self._client.is_connected:
            return False
        success = self._client.send_frame_to_track(frame, 0)
        return success

    def send_frame_to_track(
        self, frame: VideoFrame | np.ndarray, track_index: int
    ) -> bool:
        if self._client is None or not self._client.is_connected:
            return False
        return self._client.send_frame_to_track(frame, track_index)

    def get_webrtc_client(self):
        return self._client

    def get_source_track_index(self, node_id: str) -> int | None:
        if self._client is None:
            return None
        return self._client.source_node_to_track_index.get(node_id)

    def send_parameters(self, params: dict[str, Any]) -> None:
        if self._client is not None and self._client.is_connected:
            self._client.send_parameters(params)

    async def api_request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        timeout: float = 30.0,
    ) -> dict:
        """Proxy an API request to the Livepeer runner."""
        if self._client is None or not self._client.is_connected:
            raise RuntimeError("Livepeer backend is not connected")
        return await self._client.api_request(method, path, body, timeout)

    def add_frame_callback(self, callback: Callable[[VideoFrame], None]) -> None:
        self._frame_callbacks.append(callback)

    def remove_frame_callback(self, callback: Callable[[VideoFrame], None]) -> None:
        if callback in self._frame_callbacks:
            self._frame_callbacks.remove(callback)

    def add_audio_callback(self, callback: Callable[[AudioFrame], None]) -> None:
        """Register a callback to receive audio frames from Livepeer."""
        self._audio_callbacks.append(callback)

    def remove_audio_callback(self, callback: Callable[[AudioFrame], None]) -> None:
        """Remove an audio callback."""
        if callback in self._audio_callbacks:
            self._audio_callbacks.remove(callback)

    def _on_frame_from_livepeer(self, frame: VideoFrame) -> None:
        for callback in list(self._frame_callbacks):
            try:
                callback(frame)
            except Exception as e:  # pragma: no cover - defensive callback guard
                logger.error(f"Frame callback failed: {e}")

    def _on_audio_from_livepeer(self, frame: AudioFrame) -> None:
        for callback in list(self._audio_callbacks):
            try:
                callback(frame)
            except Exception as e:  # pragma: no cover - defensive callback guard
                logger.error(f"Audio callback failed: {e}")

    def get_status(self) -> dict[str, Any]:
        status = {
            "connected": self.is_connected,
            "connecting": self._connecting,
            "error": self._connect_error,
            "webrtc_connected": self.webrtc_connected,
            "app_id": "livepeer" if self.is_connected else None,
            "connection_id": (
                self._client.connection_id if self._client is not None else None
            ),
            "last_close_code": self._last_close_code,
            "last_close_reason": self._last_close_reason,
        }

        if self.webrtc_connected:
            status["stats"] = {
                "uptime_seconds": (
                    (time.time() - self._stats["connected_at"])
                    if self._stats["connected_at"]
                    else None
                ),
            }
            if self._client is not None:
                status["webrtc_stats"] = self._client.get_stats()

        return status
