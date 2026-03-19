"""Node registry for centralized node type management.

Mirrors the ``PipelineRegistry`` pattern: a class-level dictionary keyed by
``node_type_id`` that maps to concrete ``BaseNode`` subclasses.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseNode, NodeDefinition

logger = logging.getLogger(__name__)


class NodeRegistry:
    """Registry for managing available node types."""

    _nodes: dict[str, type[BaseNode]] = {}

    @classmethod
    def register(cls, node_class: type[BaseNode]) -> None:
        """Register a node class using its ``node_type_id``.

        Args:
            node_class: Concrete ``BaseNode`` subclass to register.
        """
        node_type_id = node_class.node_type_id
        cls._nodes[node_type_id] = node_class
        logger.debug("Registered node type: %s", node_type_id)

    @classmethod
    def get(cls, node_type_id: str) -> type[BaseNode] | None:
        """Look up a node class by its type ID.

        Returns:
            The node class if found, ``None`` otherwise.
        """
        return cls._nodes.get(node_type_id)

    @classmethod
    def unregister(cls, node_type_id: str) -> bool:
        """Remove a node type from the registry.

        Returns:
            ``True`` if removed, ``False`` if not found.
        """
        if node_type_id in cls._nodes:
            del cls._nodes[node_type_id]
            return True
        return False

    @classmethod
    def list_nodes(cls) -> list[str]:
        """Return all registered node type IDs."""
        return list(cls._nodes.keys())

    @classmethod
    def get_all_definitions(cls) -> list[NodeDefinition]:
        """Return ``NodeDefinition`` for every registered node type."""
        return [node_cls.get_definition() for node_cls in cls._nodes.values()]
