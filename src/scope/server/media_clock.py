"""Shared media clock for synchronizing audio and video streams.

Provides a single source of truth for media timing so that audio and video
WebRTC tracks produce correlated PTS values. aiortc's RTCP Sender Reports
then map these to NTP wallclock time for receiver-side A/V sync.
"""

import threading
import time

# Standard WebRTC clock rates
AUDIO_CLOCK_RATE = 48000  # WebRTC audio: 48 kHz
VIDEO_CLOCK_RATE = 90000  # WebRTC video: 90 kHz


class MediaClock:
    """Shared clock for synchronizing audio and video streams.

    Both VideoProcessingTrack and AudioProcessingTrack reference the same
    MediaClock instance. The clock starts when the first media frame is ready
    to play, and get_media_time() returns elapsed wall-clock seconds since then.

    PTS values derived from get_media_time() are correlated across tracks,
    allowing the WebRTC receiver to synchronize audio and video playback.
    """

    def __init__(self):
        self._start_time: float | None = None
        self._lock = threading.Lock()

    def start(self):
        """Start the clock. Call when the first media frame is ready to play.

        Safe to call multiple times; only the first call takes effect.
        """
        with self._lock:
            if self._start_time is None:
                self._start_time = time.time()

    @property
    def is_started(self) -> bool:
        with self._lock:
            return self._start_time is not None

    def get_media_time(self) -> float:
        """Get elapsed media time in seconds since the clock started.

        Returns 0.0 if the clock hasn't been started yet.
        """
        with self._lock:
            if self._start_time is None:
                return 0.0
            return time.time() - self._start_time

    def to_pts(self, clock_rate: int) -> int:
        """Get current media time as PTS in the given clock rate.

        Combines get_media_time() and conversion in a single call to minimize
        drift between the time read and the PTS calculation.
        """
        return int(self.get_media_time() * clock_rate)
