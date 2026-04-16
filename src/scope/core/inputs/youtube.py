"""YouTube input source.

Downloads a YouTube video to a local cache via yt-dlp, then reuses
VideoFileInputSource to decode and loop frames with PyAV pacing.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import ClassVar

import numpy as np

from .interface import (
    InputSource,
    InputSourceInfo,
    InvalidSourceURLError,
    SourceUnavailableError,
)
from .video_file import VideoFileInputSource

logger = logging.getLogger(__name__)

# yt-dlp format selector: prefer mp4 video+audio ≤1080p, fallback to any ≤1080p.
_YTDLP_FORMAT = (
    "bv*[ext=mp4][height<=1080]+ba[ext=m4a]/b[ext=mp4][height<=1080]/best[height<=1080]"
)

# Only accept URLs on these hosts. Guards against yt-dlp's generic
# extractor being used as an SSRF oracle.
_ALLOWED_HOSTS = re.compile(
    r"^(?:https?://)?(?:www\.|m\.)?(?:youtube\.com|youtu\.be)(?:/|$)",
    re.IGNORECASE,
)

# Patterns to extract the 11-char YouTube video ID.
_VIDEO_ID_PATTERNS = [
    re.compile(r"(?:v=|/shorts/|/embed/|youtu\.be/)([A-Za-z0-9_-]{11})(?:[?&/]|$)"),
]

_VIDEO_ID_LEN = 11

# Per-video-id lock to prevent concurrent downloads of the same URL.
_download_locks_guard = threading.Lock()
_download_locks: dict[str, threading.Lock] = {}


def _extract_video_id(url: str) -> str | None:
    """Return the 11-char video id, or None if the URL isn't a valid YouTube URL."""
    if not isinstance(url, str) or not url.strip():
        return None
    url = url.strip()
    if not _ALLOWED_HOSTS.search(url):
        return None
    for pat in _VIDEO_ID_PATTERNS:
        m = pat.search(url)
        if m:
            vid = m.group(1)
            if len(vid) == _VIDEO_ID_LEN:
                return vid
    return None


def _get_cache_path(video_id: str) -> Path:
    """Return cache file path for a given video id, creating parent dir."""
    from scope.server.models_config import get_assets_dir

    # Put cache next to assets so users don't need a new env var.
    cache_dir = get_assets_dir().parent / "cache" / "youtube"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{video_id}.mp4"


def _classify_download_error(msg: str) -> tuple[str, str]:
    """Classify a yt-dlp error message into (category, human message).

    Categories: private, unavailable, age, geo, rate, live, other.
    """
    low = msg.lower()
    if "private" in low:
        return "private", "Video is private"
    if "members-only" in low or "members only" in low:
        return "private", "Video is members-only"
    if "removed" in low or "not available" in low or "unavailable" in low:
        return "unavailable", "Video is unavailable or has been removed"
    if "age" in low and "restrict" in low:
        return "age", "Video is age-restricted"
    if "country" in low or "geo" in low or "not available in your" in low:
        return "geo", "Video is geo-blocked in this region"
    if "429" in low or "too many requests" in low:
        return "rate", "YouTube rate-limited the request (HTTP 429)"
    if "live" in low and "stream" in low:
        return "live", "Live streams are not supported"
    return "other", msg


class YouTubeInputSource(InputSource):
    """Input source that downloads a YouTube video and plays it on loop."""

    source_id: ClassVar[str] = "youtube"
    source_name: ClassVar[str] = "YouTube"
    source_description: ClassVar[str] = (
        "Download a YouTube video by URL and play it on loop."
    )

    def __init__(self):
        self._inner: VideoFileInputSource | None = None
        self._url: str | None = None
        self._video_id: str | None = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            import av  # noqa: F401
            import yt_dlp  # noqa: F401

            return True
        except ImportError:
            return False

    def list_sources(self, timeout_ms: int = 5000) -> list[InputSourceInfo]:
        # YouTube can't be enumerated; source_name is the URL itself.
        return []

    def connect(self, identifier: str) -> bool:
        """Resolve + download the URL to cache, then open it with PyAV.

        Args:
            identifier: A YouTube URL.

        Returns:
            True if the video is downloaded and opened.
        """
        self.disconnect()

        video_id = _extract_video_id(identifier)
        if video_id is None:
            logger.error(f"Invalid YouTube URL: {identifier!r}")
            return False

        # Reject live streams before downloading.
        is_live = self._probe_is_live(identifier)
        if is_live is True:
            logger.error(f"Refusing to connect to live stream: {identifier}")
            return False

        cache_path = _get_cache_path(video_id)

        # Serialize concurrent downloads of the same video id.
        lock = self._get_download_lock(video_id)
        with lock:
            if not cache_path.exists() or cache_path.stat().st_size == 0:
                if not self._download(identifier, cache_path):
                    return False

            # Open via VideoFileInputSource; on failure retry once after
            # deleting a possibly-corrupted cache.
            inner = VideoFileInputSource()
            if inner.connect(str(cache_path)):
                self._inner = inner
                self._url = identifier
                self._video_id = video_id
                logger.info(
                    f"YouTubeInputSource connected: {identifier} "
                    f"(cache={cache_path.name})"
                )
                return True

            logger.warning(
                f"Cached file failed to open ({cache_path.name}), redownloading once"
            )
            try:
                cache_path.unlink(missing_ok=True)
            except OSError as e:
                logger.error(f"Could not delete corrupted cache {cache_path}: {e}")
                return False

            if not self._download(identifier, cache_path):
                return False

            inner = VideoFileInputSource()
            if inner.connect(str(cache_path)):
                self._inner = inner
                self._url = identifier
                self._video_id = video_id
                return True

            logger.error(f"Failed to open YouTube video after redownload: {identifier}")
            return False

    def receive_frame(self, timeout_ms: int = 100) -> np.ndarray | None:
        if self._inner is None:
            return None
        return self._inner.receive_frame(timeout_ms=timeout_ms)

    def disconnect(self):
        if self._inner is not None:
            try:
                self._inner.close()
            except Exception as e:
                logger.error(f"Error closing inner video source: {e}")
            finally:
                self._inner = None
        self._url = None
        self._video_id = None

    def get_source_resolution(
        self, identifier: str, timeout_ms: int = 5000
    ) -> tuple[int, int] | None:
        """Probe the video's resolution without downloading the full file.

        Uses yt-dlp's metadata-only extract. Returns None if the URL is
        invalid, the video is unavailable, or the metadata lacks dimensions.
        """
        if _extract_video_id(identifier) is None:
            logger.error(f"Invalid YouTube URL for resolution probe: {identifier!r}")
            raise InvalidSourceURLError("URL is not a recognized YouTube video link")

        try:
            import yt_dlp
        except ImportError:
            logger.error("yt-dlp is not installed; cannot probe YouTube source")
            return None

        ydl_opts = {
            "format": _YTDLP_FORMAT,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "socket_timeout": max(1, timeout_ms // 1000),
            "skip_download": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(identifier, download=False)
        except yt_dlp.utils.DownloadError as e:
            cat, msg = _classify_download_error(str(e))
            logger.error(f"YouTube probe failed ({cat}): {msg}")
            if cat in ("private", "unavailable", "age", "geo", "live"):
                raise SourceUnavailableError(msg) from e
            # Rate limit / other → return None (endpoint falls through to 408).
            return None
        except Exception as e:
            logger.error(f"YouTube probe error: {e}")
            return None

        if info is None:
            return None
        if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
            logger.error("Refusing live stream for resolution probe")
            raise SourceUnavailableError("Live streams are not supported")

        width = info.get("width")
        height = info.get("height")
        if (
            isinstance(width, int)
            and isinstance(height, int)
            and width > 0
            and height > 0
        ):
            return (width, height)

        # Fallback: scan the merged format's requested_formats.
        req = info.get("requested_formats") or []
        for fmt in req:
            w = fmt.get("width")
            h = fmt.get("height")
            if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
                return (w, h)
        return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _get_download_lock(video_id: str) -> threading.Lock:
        with _download_locks_guard:
            lock = _download_locks.get(video_id)
            if lock is None:
                lock = threading.Lock()
                _download_locks[video_id] = lock
            return lock

    @staticmethod
    def _probe_is_live(url: str) -> bool | None:
        """Return True if the URL refers to a live stream, False if not, None on error."""
        try:
            import yt_dlp
        except ImportError:
            return None
        try:
            with yt_dlp.YoutubeDL(
                {"quiet": True, "no_warnings": True, "skip_download": True}
            ) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception:
            return None
        if info is None:
            return None
        if info.get("is_live"):
            return True
        if info.get("live_status") in ("is_live", "is_upcoming"):
            return True
        return False

    @staticmethod
    def _download(url: str, cache_path: Path) -> bool:
        """Download the video to ``cache_path``. Returns True on success.

        Retries once on HTTP 429 after a 5s backoff.
        """
        try:
            import yt_dlp
        except ImportError:
            logger.error("yt-dlp is not installed; install it to use YouTube sources")
            return False

        tmp_path = cache_path.with_suffix(cache_path.suffix + ".part")
        ydl_opts = {
            "format": _YTDLP_FORMAT,
            "outtmpl": str(cache_path),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "merge_output_format": "mp4",
            "overwrites": True,
            # Prevent creation of .live_chat.json etc.
            "writesubtitles": False,
            "writeautomaticsub": False,
        }

        def _run() -> tuple[bool, str]:
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                return True, ""
            except yt_dlp.utils.DownloadError as e:
                return False, str(e)
            except OSError as e:
                return False, f"disk error: {e}"
            except Exception as e:
                return False, f"{type(e).__name__}: {e}"

        ok, err = _run()
        if not ok:
            cat, msg = _classify_download_error(err)
            if cat == "rate":
                logger.warning(f"YouTube rate-limited; retrying after 5s: {msg}")
                time.sleep(5.0)
                ok, err = _run()
                if not ok:
                    cat, msg = _classify_download_error(err)

        if not ok:
            logger.error(f"YouTube download failed ({cat}): {msg}")
            # Best-effort cleanup of partials.
            for p in (tmp_path, cache_path):
                try:
                    if p.exists() and p.stat().st_size == 0:
                        p.unlink()
                except OSError:
                    pass
            return False

        if not cache_path.exists() or cache_path.stat().st_size == 0:
            logger.error(
                f"YouTube download reported success but file missing: {cache_path}"
            )
            return False

        logger.info(
            f"YouTube download complete: {cache_path.name} "
            f"({cache_path.stat().st_size // 1024} KB)"
        )
        return True
