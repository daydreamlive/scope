"""Node registry for centralised backend node management.

Mirrors the pipeline registry pattern: node classes are registered by their
``node_type_id`` and can be looked up at runtime by the ``NodeManager`` to
create instances.  Built-in nodes are registered on module import; plugin
nodes are discovered via pluggy entry points.
"""

import importlib
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .interface import BaseNode

logger = logging.getLogger(__name__)


class NodeRegistry:
    """Registry for managing available backend node types."""

    _nodes: dict[str, type["BaseNode"]] = {}

    @classmethod
    def register(cls, node_type_id: str, node_class: type["BaseNode"]) -> None:
        """Register a node class with its type ID.

        Args:
            node_type_id: Unique identifier for the node type.
            node_class: Node class to register.
        """
        cls._nodes[node_type_id] = node_class
        logger.debug(f"Registered node type: {node_type_id}")

    @classmethod
    def get(cls, node_type_id: str) -> type["BaseNode"] | None:
        """Get a node class by its type ID.

        Args:
            node_type_id: Node type identifier.

        Returns:
            Node class if found, ``None`` otherwise.
        """
        return cls._nodes.get(node_type_id)

    @classmethod
    def unregister(cls, node_type_id: str) -> bool:
        """Remove a node type from the registry.

        Args:
            node_type_id: Node type identifier to remove.

        Returns:
            ``True`` if the node was removed, ``False`` if not found.
        """
        if node_type_id in cls._nodes:
            del cls._nodes[node_type_id]
            return True
        return False

    @classmethod
    def is_registered(cls, node_type_id: str) -> bool:
        """Check if a node type is registered."""
        return node_type_id in cls._nodes

    @classmethod
    def list_nodes(cls) -> list[str]:
        """Return all registered node type IDs."""
        return list(cls._nodes.keys())

    @classmethod
    def get_config_class(cls, node_type_id: str) -> Any:
        """Get the config class for a registered node type.

        Returns:
            The ``BaseNodeConfig`` subclass, or ``None`` if not found.
        """
        node_class = cls.get(node_type_id)
        if node_class is None:
            return None
        return node_class.get_config_class()

    @classmethod
    def get_all_schemas(cls) -> list[dict[str, Any]]:
        """Return frontend-ready schemas for every registered node type."""
        schemas: list[dict[str, Any]] = []
        for node_type_id in cls._nodes:
            config_cls = cls.get_config_class(node_type_id)
            if config_cls is not None:
                schemas.append(config_cls.get_schema_for_frontend())
        return schemas


def _register_builtin_nodes() -> None:
    """Register built-in node types shipped with scope."""
    node_configs: list[tuple[str, str, str]] = [
        ("scheduler", ".scheduler.node", "SchedulerNode"),
    ]

    for node_name, module_path, class_name in node_configs:
        try:
            module = importlib.import_module(module_path, package=__package__)
            node_class = getattr(module, class_name)
            config_class = node_class.get_config_class()
            NodeRegistry.register(config_class.node_type_id, node_class)
        except ImportError as e:
            logger.warning(
                f"Could not import {node_name} node: {e}. "
                f"This node will not be available."
            )
        except Exception as e:
            logger.warning(
                f"Error loading {node_name} node: {e}. This node will not be available."
            )


def _initialize_registry() -> None:
    """Initialise the registry with built-in nodes and plugin nodes."""
    _register_builtin_nodes()

    try:
        from scope.core.plugins import load_plugins, register_plugin_nodes

        load_plugins()
        register_plugin_nodes(NodeRegistry)
    except Exception as e:
        logger.debug(
            f"Plugin node loading skipped or failed: {e}. "
            f"Built-in nodes are still available."
        )

    node_count = len(NodeRegistry.list_nodes())
    logger.info(f"Node registry initialised with {node_count} node type(s)")


_initialize_registry()
