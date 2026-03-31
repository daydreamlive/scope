"""Base interface for all backend nodes.

Nodes are stateful, event-driven processing units that participate in the
execution graph alongside pipelines. They produce and consume discrete values
and trigger events, but do not process video frames.

Thread management is the developer's responsibility: the framework does not
impose any threading model. A scheduler node may run its own timer thread while
a simple math node computes synchronously inside ``update_input``.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Callable

    from .base_schema import BaseNodeConfig


class ConnectorDef(BaseModel):
    """Definition of a typed input or output port on a node."""

    name: str = Field(..., description="Port name used for routing and display")
    type: Literal["float", "int", "string", "bool", "trigger"] = Field(
        ..., description="Data type carried by this connector"
    )
    direction: Literal["input", "output"] = Field(
        ..., description="Whether this is an input or output port"
    )
    default: Any = Field(default=None, description="Default value (None for triggers)")
    ui: dict[str, Any] | None = Field(
        default=None,
        description="Optional UI hints for frontend rendering (min, max, step, widget, etc.)",
    )


class BaseNode(ABC):
    """Abstract base class for all backend nodes.

    Subclasses must implement:
    - ``get_config_class()`` to return the node's config (ports, metadata).
    - ``setup(emit_output)`` to initialise resources; ``emit_output`` is the
      callback the node must call whenever an output value changes.
    - ``teardown()`` to release resources (threads, connections, etc.).
    - ``update_input(name, value)`` to react to incoming values/triggers.

    Optional overrides:
    - ``update_config(config)`` for runtime configuration changes.
    - ``get_state()`` to expose internal state for frontend observation.
    - ``get_current_ports()`` for nodes with ``dynamic_ports = True``.
    """

    @classmethod
    @abstractmethod
    def get_config_class(cls) -> type["BaseNodeConfig"]:
        """Return the Pydantic config class describing this node type."""
        ...

    @abstractmethod
    def setup(self, emit_output: "Callable[[str, Any], None]") -> None:
        """Initialise the node.

        Args:
            emit_output: Callback to fire an output value. The node must call
                ``emit_output(port_name, value)`` whenever an output changes.
                The ``NodeManager`` handles routing to connected nodes/pipelines
                and optional frontend broadcast.
        """
        ...

    @abstractmethod
    def teardown(self) -> None:
        """Release all resources (threads, file handles, connections, etc.)."""
        ...

    @abstractmethod
    def update_input(self, name: str, value: Any) -> None:
        """Receive an input value or trigger event.

        Called by the ``NodeManager`` when a connected upstream node emits a
        value targeting one of this node's input ports, or when an external
        caller (REST, CLI, OSC) sends an input update.

        Args:
            name: Input port name (must match a ``ConnectorDef.name``).
            value: The incoming value. For triggers this is typically ``True``.
        """
        ...

    def on_stream_stop(self) -> None:  # noqa: B027
        """Called by the framework when the pipeline/stream stops.

        Override to stop playback, reset state, kill threads, etc.
        The node instance is NOT destroyed — it may be restarted later.
        Default implementation does nothing.
        """

    def update_config(self, config: dict[str, Any]) -> None:  # noqa: B027
        """Apply a runtime configuration change.

        Override this for nodes that support reconfiguration while running
        (e.g. adding trigger points to a scheduler, changing clock source).
        """

    def get_state(self) -> dict[str, Any]:
        """Return the node's current observable state.

        The returned dict is JSON-serialisable and broadcast to connected
        WebSocket clients for frontend visualisation. Execution does **not**
        depend on this method being called.
        """
        return {}

    def get_current_ports(self) -> dict[str, list[ConnectorDef]] | None:
        """Return the current input/output ports if they differ from the config.

        Only relevant for nodes with ``dynamic_ports = True``. Return
        ``{"inputs": [...], "outputs": [...]}`` or ``None`` to use the static
        config ports.
        """
        return None
