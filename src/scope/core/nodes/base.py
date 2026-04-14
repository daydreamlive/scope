"""Base classes for the Scope node system.

Nodes are lightweight, fine-grained processing units that can be wired into
pipeline graphs alongside pipelines. Each node type declares typed input/
output ports and editable parameters, and subclasses implement their own
execution contract (which may differ between execution backends).

This module intentionally keeps ``BaseNode`` minimal — only a class-level
identifier and a ``get_definition()`` classmethod are required. Concrete
execution backends (graph executor integration, event-driven runtime, etc.)
layer their own abstract methods on top.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field


class NodePort(BaseModel):
    """Describes an input or output port on a node."""

    name: str = Field(..., description="Port identifier (used in edge wiring)")
    port_type: str = Field(
        ...,
        description=(
            "Type of data carried by this port. Built-in types: "
            "'audio', 'video', 'number', 'string', 'boolean', 'trigger'. "
            "Plugins may define custom types (e.g. 'latent', 'model')."
        ),
    )
    required: bool = Field(default=True, description="Whether this input is required")
    description: str = Field(default="", description="Human-readable description")
    default_value: Any = Field(default=None, description="Default value for inputs")


class NodeParam(BaseModel):
    """Describes an editable parameter (widget) on a node.

    Parameters are user-configurable values that live on the node card.
    Like ComfyUI widgets, a parameter may be overridden by connecting
    an incoming wire to the corresponding input port — the widget then
    becomes an input and the default value is ignored.

    Widget-specific hints (number min/max/step, select options, etc.)
    go into the free-form ``ui`` dict so the base schema doesn't grow
    as new widget kinds are added. The frontend renderer dispatches on
    ``param_type`` and reads whichever ``ui`` keys apply.
    """

    name: str = Field(..., description="Parameter identifier")
    param_type: Literal["number", "string", "boolean", "select"] = Field(
        ..., description="Widget type for the frontend"
    )
    default: Any = Field(default=None, description="Default value")
    description: str = Field(default="", description="Human-readable label")
    ui: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Widget-specific hints consumed by the frontend renderer. "
            "Number widgets read ``min``/``max``/``step``; select "
            "widgets read ``options``; plugin-defined widget kinds may "
            "use any keys they like."
        ),
    )
    convertible_to_input: bool = Field(
        default=True,
        description=(
            "If True, this parameter can be overridden by connecting an "
            "input wire (ComfyUI-style widget-to-input conversion)."
        ),
    )


class NodeDefinition(BaseModel):
    """Static metadata describing a node type."""

    node_type_id: str = Field(..., description="Unique node type identifier")
    display_name: str = Field(..., description="Human-readable name")
    category: str = Field(default="general", description="Category for grouping")
    description: str = Field(default="", description="What this node does")
    inputs: list[NodePort] = Field(default_factory=list)
    outputs: list[NodePort] = Field(default_factory=list)
    params: list[NodeParam] = Field(
        default_factory=list,
        description="Editable parameters (widgets) displayed on the node card.",
    )
    continuous: bool = Field(
        default=False,
        description=(
            "If True, source nodes (no inputs) re-execute continuously "
            "instead of executing once. Useful for streaming generators."
        ),
    )
    pipeline_meta: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Rich pipeline-only metadata (config_schema, mode_defaults, "
            "supports_lora, supports_vace, etc.) for nodes that are "
            ":class:`Pipeline` subclasses. ``None`` for plain nodes. "
            "Populated by ``Pipeline.get_definition()`` from the config "
            "class's ``get_schema_with_metadata()``."
        ),
    )


class BaseNode(ABC):
    """Abstract base class for all backend node types.

    Subclasses must set ``node_type_id`` as a ``ClassVar`` and implement
    ``get_definition()`` and ``execute()``. Nodes run inside a
    :class:`NodeProcessor` in the pipeline graph: input port values arrive
    in the ``inputs`` dict and the return dict fans out to downstream
    nodes or pipelines via queue edges.

    Example::

        class MyNode(BaseNode):
            node_type_id: ClassVar[str] = "my-plugin.my-node"

            @classmethod
            def get_definition(cls) -> NodeDefinition:
                return NodeDefinition(
                    node_type_id=cls.node_type_id,
                    display_name="My Node",
                    inputs=[NodePort(name="audio", port_type="audio")],
                    outputs=[NodePort(name="audio", port_type="audio")],
                )

            def execute(self, inputs, **kwargs):
                return {"audio": inputs["audio"]}
    """

    node_type_id: ClassVar[str]

    def __init__(self, node_id: str = "", config: dict[str, Any] | None = None):
        self.node_id = node_id
        self.config = config or {}

    @classmethod
    @abstractmethod
    def get_definition(cls) -> NodeDefinition:
        """Return static metadata for this node type."""

    @abstractmethod
    def execute(
        self,
        inputs: dict[str, Any],
        **kwargs,
    ) -> dict[str, Any]:
        """Execute the node with the given inputs.

        Args:
            inputs: Dict mapping input port names to their values.
                Audio ports receive ``(tensor, sample_rate)`` tuples.
                Video ports receive frame tensors.
            **kwargs: Additional context (pipeline parameters, etc.)

        Returns:
            Dict mapping output port names to their values.
        """
