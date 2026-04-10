"""Registry for custom node types."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseNode, NodeDefinition

logger = logging.getLogger(__name__)


class NodeRegistry:
    """Central registry for all available node types.

    Keyed by ``node_type_id`` (read from the registered class).
    """

    _nodes: dict[str, type[BaseNode]] = {}

    @classmethod
    def register(cls, node_class: type[BaseNode]) -> None:
        node_type_id = node_class.node_type_id
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
