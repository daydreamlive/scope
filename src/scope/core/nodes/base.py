"""Base classes for the backend node framework.

Defines the abstract node interface, port descriptors, and node definition
metadata.  Every concrete node (built-in or plugin-provided) inherits from
``BaseNode`` and exposes a static ``NodeDefinition`` via ``get_definition()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel

from scope.server.tempo_sync import BeatState


class NodePort(BaseModel):
    """Descriptor for a single input or output port on a node."""

    name: str
    port_type: str  # "number", "string", "boolean", "stream", "any"
    default_value: Any = None
    label: str | None = None
    min_value: float | None = None
    max_value: float | None = None


class NodeDefinition(BaseModel):
    """Static metadata describing a node type (ports, category, flags)."""

    node_type_id: str
    display_name: str
    category: str  # "math", "control", "input", "output", "pipeline", "utility"
    description: str = ""
    inputs: list[NodePort] = []
    outputs: list[NodePort] = []
    is_animated: bool = False  # Needs tick (control patterns, LFOs)
    is_stream_node: bool = False  # Processes video frames (pipeline bridge)


class BaseNode(ABC):
    """Abstract base class for all graph nodes.

    Concrete nodes must set ``node_type_id`` as a ClassVar and implement
    ``get_definition()`` and ``execute()``.
    """

    node_type_id: ClassVar[str]

    def __init__(self, node_id: str, config: dict[str, Any]) -> None:
        self.node_id = node_id
        self.config = config  # Persistent user state (slider value, math op, …)
        self._state: dict[str, Any] = {}  # Ephemeral runtime state

    @classmethod
    @abstractmethod
    def get_definition(cls) -> NodeDefinition:
        """Return the static definition for this node type."""
        ...

    @abstractmethod
    def execute(
        self,
        inputs: dict[str, Any],
        tick_time: float,
        beat_state: BeatState | None,
    ) -> dict[str, Any]:
        """Execute for one tick.

        Args:
            inputs: Mapping of input port name → current value.
            tick_time: Monotonic seconds since graph start.
            beat_state: Current tempo/beat snapshot (may be ``None``).

        Returns:
            Mapping of output port name → computed value.
        """
        ...

    def on_event(self, event_type: str, payload: dict[str, Any]) -> None:  # noqa: B027
        """Handle a frontend interaction (slider drag, button click, etc.)."""

    def reset(self) -> None:
        """Clear ephemeral runtime state."""
        self._state.clear()
