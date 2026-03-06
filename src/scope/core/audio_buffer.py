"""In-memory buffer for audio pipeline output.

Audio pipelines store their WAV output here instead of writing to disk.
The server's HTTP endpoints serve directly from this buffer.
"""

import threading
import time

_lock = threading.Lock()
_wav_data: bytes | None = None
_timestamp: float | None = None


def store_audio(wav_bytes: bytes) -> None:
    """Store encoded WAV bytes in the buffer."""
    global _wav_data, _timestamp
    with _lock:
        _wav_data = wav_bytes
        _timestamp = time.time()


def get_audio() -> tuple[bytes | None, float | None]:
    """Return (wav_bytes, timestamp) or (None, None) if no audio stored."""
    with _lock:
        return _wav_data, _timestamp
