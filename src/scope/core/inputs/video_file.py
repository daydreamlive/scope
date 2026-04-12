"""Video file input source implementation using PyAV."""

import logging
import time
from pathlib import Path
from typing import ClassVar

import numpy as np

from ..pacing import MediaPacingConfig, MediaPacingState, compute_pacing_decision
from .interface import InputSource, InputSourceInfo

logger = logging.getLogger(__name__)

# Tighter tolerance than the default (10ms) since decode jitter is negligible.
_DRIFT_TOLERANCE_S = 0.002


class VideoFileInputSource(InputSource):
    """Input source that reads video frames from a local file.

    Uses PyAV (FFmpeg wrapper) to decode frames sequentially.
    Loops back to the beginning when the video ends.
    """

    source_id: ClassVar[str] = "video_file"
    source_name: ClassVar[str] = "Video File"
    source_description: ClassVar[str] = (
        "Read video frames from a local file (MP4, AVI, MOV, MKV, WebM)."
    )

    def __init__(self):
        self._container = None
        self._stream = None
        self._frame_iter = None
        self._connected = False
        self._file_path: str | None = None
        self._pacing_state = MediaPacingState()
        self._pacing_config = MediaPacingConfig(drift_tolerance_s=_DRIFT_TOLERANCE_S)

    @classmethod
    def is_available(cls) -> bool:
        """Always available (PyAV is a core dependency via aiortc)."""
        try:
            import av  # noqa: F401

            return True
        except ImportError:
            return False

    def list_sources(self, timeout_ms: int = 5000) -> list[InputSourceInfo]:
        """List video files from the assets directory."""
        try:
            from scope.server.file_utils import VIDEO_EXTENSIONS, iter_files
            from scope.server.models_config import get_assets_dir

            assets_dir = get_assets_dir()
            sources = []
            for file_path in iter_files(assets_dir, VIDEO_EXTENSIONS):
                sources.append(
                    InputSourceInfo(
                        name=file_path.stem,
                        identifier=str(file_path),
                        metadata={"filename": file_path.name},
                    )
                )
            return sources
        except ImportError:
            logger.warning("Could not import server utilities to list video assets")
            return []
        except Exception as e:
            logger.error(f"Error listing video file sources: {e}")
            return []

    def connect(self, identifier: str) -> bool:
        """Open a video file by path.

        Args:
            identifier: Absolute or relative path to the video file.

        Returns:
            True if the file was opened successfully.
        """
        import av

        self.disconnect()

        path = Path(identifier)
        if not path.is_file():
            # Try resolving by name in the assets directory
            path = self._resolve_in_assets(identifier)
            if path is None:
                logger.error(f"Video file not found: {identifier}")
                return False

        try:
            self._container = av.open(str(path))

            if not self._container.streams.video:
                logger.error(f"No video stream found in: {identifier}")
                self._container.close()
                self._container = None
                return False

            self._stream = self._container.streams.video[0]
            self._stream.thread_type = "AUTO"
            self._frame_iter = self._container.decode(self._stream)
            self._file_path = str(path)
            self._connected = True
            self._reset_pacing()

            logger.info(
                f"VideoFileInputSource connected: {path.name} "
                f"({self._stream.width}x{self._stream.height})"
            )
            return True
        except Exception as e:
            logger.error(f"Error opening video file '{identifier}': {e}")
            if self._container is not None:
                try:
                    self._container.close()
                except Exception:
                    pass
            self._container = None
            self._stream = None
            self._frame_iter = None
            return False

    def receive_frame(self, timeout_ms: int = 100) -> np.ndarray | None:
        """Decode and return the next video frame, paced to real-time.

        Uses the file's PTS to sleep so frames are emitted at the source
        frame rate rather than at unbounded CPU decode speed. On loop (seek
        back to start) the pacing anchor is reset automatically.

        Returns:
            (H, W, 3) uint8 RGB array, or None if no frame is available.
        """
        if not self._connected or self._frame_iter is None:
            return None

        try:
            frame = next(self._frame_iter)
        except StopIteration:
            # End of video: loop back to the beginning
            try:
                self._container.seek(0)
                self._frame_iter = self._container.decode(self._stream)
                frame = next(self._frame_iter)
                self._reset_pacing()
            except (StopIteration, Exception) as e:
                if isinstance(e, StopIteration):
                    logger.warning("Video file appears empty after seek")
                else:
                    logger.error(f"Error seeking video file: {e}")
                return None
        except Exception as e:
            logger.error(f"Error decoding video frame: {e}")
            return None

        self._pace(frame)

        return frame.to_ndarray(format="rgb24")

    def _reset_pacing(self) -> None:
        self._pacing_state = MediaPacingState()

    def _pace(self, frame) -> None:
        """Sleep if wall-clock is ahead of the frame's media timestamp."""
        if frame.pts is None or frame.time_base is None:
            return
        media_ts = float(frame.pts * frame.time_base)
        now = time.monotonic()
        decision = compute_pacing_decision(
            self._pacing_state,
            media_ts=media_ts,
            now_monotonic=now,
            config=self._pacing_config,
        )
        if decision.sleep_s > 0:
            time.sleep(decision.sleep_s)
        self._pacing_state.prev_wall_monotonic = time.monotonic()

    @staticmethod
    def _resolve_in_assets(name: str) -> Path | None:
        """Try to find a video file in the assets directory by name or stem."""
        try:
            from scope.server.file_utils import VIDEO_EXTENSIONS, iter_files
            from scope.server.models_config import get_assets_dir

            assets_dir = get_assets_dir()
            name_path = Path(name)

            for file_path in iter_files(assets_dir, VIDEO_EXTENSIONS):
                if file_path.name == name or file_path.stem == name_path.stem:
                    return file_path
        except Exception:
            pass
        return None

    def disconnect(self):
        """Close the video file and release resources."""
        self._frame_iter = None
        self._stream = None
        if self._container is not None:
            try:
                self._container.close()
            except Exception as e:
                logger.error(f"Error closing video container: {e}")
            finally:
                self._container = None
        self._connected = False
        self._file_path = None
        self._reset_pacing()

    def get_source_resolution(
        self, identifier: str, timeout_ms: int = 5000
    ) -> tuple[int, int] | None:
        """Read the video resolution from stream metadata without decoding frames."""
        import av

        try:
            container = av.open(identifier)
            if not container.streams.video:
                container.close()
                return None
            stream = container.streams.video[0]
            resolution = (stream.width, stream.height)
            container.close()
            return resolution
        except Exception as e:
            logger.error(f"Error reading video resolution: {e}")
            return None
