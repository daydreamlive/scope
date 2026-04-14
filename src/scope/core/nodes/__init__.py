"""Backend node system for Scope.

Provides a base class and registry for defining fine-grained processing
nodes that can be wired into pipeline graphs. Nodes are simpler than full
pipelines — they declare typed input/output ports, editable parameters,
and a small execution contract. Built-in nodes and plugin-provided nodes
are discovered here and rendered generically by the frontend via
``GET /api/v1/nodes/definitions``.
"""

from .base import BaseNode, NodeDefinition, NodeParam, NodePort
from .builtins import AudioSourceNode
from .registry import NodeRegistry


def register_builtin_nodes() -> None:
    """Register all built-in node types shipped with the foundation."""
    NodeRegistry.register(AudioSourceNode)


__all__ = [
    "AudioSourceNode",
    "BaseNode",
    "NodeDefinition",
    "NodeParam",
    "NodePort",
    "NodeRegistry",
    "register_builtin_nodes",
]
