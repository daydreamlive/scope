"""Backend node system for Scope.

Provides a base class and registry for defining fine-grained processing
nodes that can be wired into pipeline graphs. Nodes are simpler than full
pipelines — they declare typed input/output ports, editable parameters,
and a small execution contract. Built-in nodes and plugin-provided nodes
are discovered here and rendered generically by the frontend via
``GET /api/v1/nodes/definitions``.
"""

from .base import BaseNode, NodeDefinition, NodeParam, NodePort
from .builtins import AudioSourceNode, SchedulerNode
from .registry import NodeRegistry


def register_builtin_nodes() -> None:
    """Register all built-in node types shipped with the foundation."""
    NodeRegistry.register(AudioSourceNode)
    NodeRegistry.register(SchedulerNode)

    # Register built-in input sources so their NodeDefinition appears in
    # /api/v1/nodes/definitions alongside plugin-registered sources. The
    # frontend renders source-node UI dynamically from these definitions.
    from scope.core.inputs import get_input_source_classes

    for source_cls in get_input_source_classes().values():
        NodeRegistry.register(source_cls)


__all__ = [
    "AudioSourceNode",
    "BaseNode",
    "NodeDefinition",
    "NodeParam",
    "NodePort",
    "NodeRegistry",
    "SchedulerNode",
    "register_builtin_nodes",
]
