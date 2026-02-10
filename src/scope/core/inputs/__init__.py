"""Input sources for Scope.

Input sources provide video frames from external sources like NDI, Spout, etc.
"""

from .interface import InputSource
from .registry import InputSourceRegistry, input_source_registry

__all__ = ["InputSource", "InputSourceRegistry", "input_source_registry"]

