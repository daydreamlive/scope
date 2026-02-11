"""Lightweight frame wrappers for non-WebRTC input sources."""


class RawFrame:
    """Wraps a raw numpy array to match the VideoFrame interface.

    Allows code that expects a VideoFrame (with .to_ndarray()) to work
    with plain numpy arrays from input sources like Spout and NDI.
    """

    __slots__ = ["_data"]

    def __init__(self, data):
        self._data = data

    def to_ndarray(self, format="rgb24"):
        return self._data
