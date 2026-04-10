"""Backend node system for Scope.

Provides a base class and registry for defining fine-grained processing
nodes that can be wired into pipeline graphs. Nodes are simpler than full
pipelines — they declare typed input/output ports, editable parameters,
and a small execution contract. Built-in nodes and custom nodes (from
plugins or local packs) are discovered here and rendered generically by
the frontend via ``GET /api/v1/nodes/definitions``.
"""

from .base import BaseNode, NodeDefinition, NodeParam, NodePort
from .builtins import AudioSinkNode, AudioSourceNode
from .registry import NodeRegistry


def register_builtin_nodes() -> None:
    """Register all built-in node types shipped with the foundation."""
    NodeRegistry.register(AudioSourceNode)
    NodeRegistry.register(AudioSinkNode)


def load_and_register_local_nodes() -> None:
    """Scan ``~/.daydream-scope/custom_nodes/`` for ComfyUI-style node packs."""
    from .loader import load_local_nodes

    for _node_type_id, node_cls in load_local_nodes().items():
        NodeRegistry.register(node_cls)


__all__ = [
    "AudioSinkNode",
    "AudioSourceNode",
    "BaseNode",
    "NodeDefinition",
    "NodeParam",
    "NodePort",
    "NodeRegistry",
    "load_and_register_local_nodes",
    "register_builtin_nodes",
]
