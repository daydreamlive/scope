"""Backend node framework for graph-mode execution.

Registers all built-in node types and plugin-provided nodes at import time,
mirroring the ``scope.core.pipelines`` initialization pattern.
"""

import logging

from .registry import NodeRegistry

logger = logging.getLogger(__name__)


def _register_builtin_nodes() -> None:
    """Register all built-in node types."""
    from .builtin.bool_node import BoolNode
    from .builtin.control_node import ControlNode
    from .builtin.image_node import ImageNode
    from .builtin.knobs_node import KnobsNode
    from .builtin.math_node import MathNode
    from .builtin.midi_node import MidiNode
    from .builtin.note_node import NoteNode
    from .builtin.pipeline_bridge_node import PipelineBridgeNode
    from .builtin.primitive_node import PrimitiveNode
    from .builtin.record_node import RecordNode
    from .builtin.reroute_node import RerouteNode
    from .builtin.sink_node import SinkNode
    from .builtin.slider_node import SliderNode
    from .builtin.source_node import SourceNode
    from .builtin.subgraph_io_nodes import SubgraphInputNode, SubgraphOutputNode
    from .builtin.subgraph_node import SubgraphNode
    from .builtin.tuple_node import TupleNode
    from .builtin.xypad_node import XYPadNode

    builtin_nodes = [
        BoolNode,
        ControlNode,
        ImageNode,
        KnobsNode,
        MathNode,
        MidiNode,
        NoteNode,
        PipelineBridgeNode,
        PrimitiveNode,
        RecordNode,
        RerouteNode,
        SinkNode,
        SliderNode,
        SourceNode,
        SubgraphInputNode,
        SubgraphNode,
        SubgraphOutputNode,
        TupleNode,
        XYPadNode,
    ]

    for node_cls in builtin_nodes:
        NodeRegistry.register(node_cls)


def _initialize_registry() -> None:
    """Initialize registry with built-in nodes and plugins."""
    _register_builtin_nodes()

    try:
        from scope.core.plugins import load_plugins, register_plugin_nodes

        load_plugins()
        register_plugin_nodes(NodeRegistry)
    except Exception as e:
        logger.error(
            "Failed to load plugin nodes: %s. Built-in nodes are still available.", e
        )

    node_count = len(NodeRegistry.list_nodes())
    logger.info("Node registry initialized with %d node type(s)", node_count)


_initialize_registry()
