"""Unified registry for every node type on the graph (plain nodes + pipelines)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseNode, NodeDefinition

logger = logging.getLogger(__name__)


def _derive_node_type_id(node_class: type) -> str | None:
    """Return the registry key for a node class, or None if not derivable.

    Plain nodes carry the id as the ``node_type_id`` classvar; pipelines
    keep it on their config class as ``pipeline_id``.
    """
    node_type_id = getattr(node_class, "node_type_id", None)
    if node_type_id is not None:
        return node_type_id
    # Lazy import: nodes.registry is loaded before pipelines.interface.
    try:
        from scope.core.pipelines.interface import Pipeline

        if issubclass(node_class, Pipeline):
            return node_class.get_config_class().pipeline_id
    except Exception:
        pass
    return None


class NodeRegistry:
    """Central registry for all available node types."""

    _nodes: dict[str, type[BaseNode]] = {}

    @classmethod
    def register(cls, node_class: type[BaseNode]) -> None:
        """Register a :class:`BaseNode` subclass (plain node or pipeline)."""
        node_type_id = _derive_node_type_id(node_class)
        if node_type_id is None:
            raise ValueError(
                f"Cannot determine node_type_id for {node_class.__name__}; "
                "set a ClassVar[str] `node_type_id` on plain nodes or a "
                "`pipeline_id` on the pipeline config class."
            )
        cls._nodes[node_type_id] = node_class
        logger.debug("Registered node type: %s", node_type_id)

    @classmethod
    def get(cls, node_type_id: str) -> type[BaseNode] | None:
        return cls._nodes.get(node_type_id)

    @classmethod
    def is_registered(cls, node_type_id: str) -> bool:
        return node_type_id in cls._nodes

    @classmethod
    def list_node_types(cls) -> list[str]:
        return list(cls._nodes.keys())

    @classmethod
    def get_all_definitions(cls) -> list[NodeDefinition]:
        return [nc.get_definition() for nc in cls._nodes.values()]

    @classmethod
    def unregister(cls, node_type_id: str) -> bool:
        if node_type_id in cls._nodes:
            del cls._nodes[node_type_id]
            return True
        return False

    @classmethod
    def clear(cls) -> None:
        cls._nodes.clear()
