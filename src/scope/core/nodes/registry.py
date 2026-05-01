"""Unified registry for every node type on the graph.

There is one storage shared by plain event-style nodes and config-driven
"pipeline" nodes — they are all just :class:`Node` subclasses now.
:class:`scope.core.pipelines.registry.PipelineRegistry` keeps its old
API as a filtered view over this same storage.
"""

from __future__ import annotations

import logging

from .base import Node, NodeDefinition

logger = logging.getLogger(__name__)


def _derive_node_type_id(node_class: type[Node]) -> str | None:
    """Return the registry key for a node class, or ``None`` if not derivable.

    Event-style nodes carry the id as the ``node_type_id`` classvar;
    config-driven nodes put it on their config class as ``pipeline_id``.
    """
    node_type_id = getattr(node_class, "node_type_id", None)
    if node_type_id:
        return node_type_id
    config_class = node_class.get_config_class()
    if config_class is not None:
        return config_class.pipeline_id
    return None


class NodeRegistry:
    """Central registry for all available node types."""

    _nodes: dict[str, type[Node]] = {}

    @classmethod
    def register(cls, node_class: type[Node]) -> None:
        """Register a :class:`Node` subclass."""
        node_type_id = _derive_node_type_id(node_class)
        if node_type_id is None:
            raise ValueError(
                f"Cannot determine node_type_id for {node_class.__name__}; "
                "set a ClassVar[str] `node_type_id` on event-style nodes or "
                "return a config class (with `pipeline_id`) from "
                "`get_config_class()` for config-driven nodes."
            )
        existing = cls._nodes.get(node_type_id)
        if existing is not None and existing is not node_class:
            logger.warning(
                "Node type '%s' already registered by %s; overwriting with %s. "
                "Check for duplicate register_nodes/register_pipelines hooks "
                "or plugins registering the same class twice.",
                node_type_id,
                existing.__name__,
                node_class.__name__,
            )
        cls._nodes[node_type_id] = node_class
        logger.debug("Registered node type: %s", node_type_id)

    @classmethod
    def get(cls, node_type_id: str) -> type[Node] | None:
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

    # ------------------------------------------------------------------
    # Chain-level helpers (historically on PipelineRegistry).
    # ------------------------------------------------------------------

    @classmethod
    def chain_produces_video(cls, node_type_ids: list[str]) -> bool:
        """Return True unless the last node in the chain declares ``produces_video=False``.

        Only config-driven nodes can opt out of video output (via their
        Pydantic config class). Plain nodes always produce video.
        """
        if not node_type_ids:
            return True
        last = cls.get(node_type_ids[-1])
        if last is None:
            return True
        config_cls = last.get_config_class()
        if config_cls is None:
            return True
        return getattr(config_cls, "produces_video", True)

    @classmethod
    def chain_produces_audio(cls, node_type_ids: list[str]) -> bool:
        """Return True if any config-driven node in the chain declares ``produces_audio=True``.

        Plain nodes cannot declare audio output today.
        """
        for type_id in node_type_ids:
            node_cls = cls.get(type_id)
            if node_cls is None:
                continue
            config_cls = node_cls.get_config_class()
            if config_cls is not None and getattr(config_cls, "produces_audio", False):
                return True
        return False
