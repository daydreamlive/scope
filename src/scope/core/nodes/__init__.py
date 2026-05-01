"""Unified node abstraction for Scope.

Every graph element — from a math helper to a full video pipeline — is
a :class:`Node`. See :mod:`scope.core.nodes.base` for the base class
and invocation styles. The :class:`NodeRegistry` stores every node type,
and plugins register into the same storage via the plugin hooks.

``BaseNode`` (historical base class name) and ``Pipeline`` (historical
subclass name, also re-exported from :mod:`scope.core.pipelines.interface`)
are both aliases for :class:`Node`.
"""

from .base import BaseNode, Node, NodeDefinition, NodeParam, NodePort, Requirements
from .builtins import SchedulerNode
from .registry import NodeRegistry


def register_builtin_nodes() -> None:
    """Register all built-in node types shipped with the foundation."""
    NodeRegistry.register(SchedulerNode)


__all__ = [
    "BaseNode",
    "Node",
    "NodeDefinition",
    "NodeParam",
    "NodePort",
    "NodeRegistry",
    "Requirements",
    "SchedulerNode",
    "register_builtin_nodes",
]
