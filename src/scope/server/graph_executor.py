"""Graph executor: builds and runs pipeline graphs from GraphConfig.

Wires source queues (for put()), pipeline processors (one per pipeline node),
and identifies the sink for get(). All frame ports (video, vace_input_frames,
vace_input_masks) use stream edges (queues); no separate parameter path.
"""

from __future__ import annotations

import logging
import queue
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .graph_schema import GraphConfig
from .pipeline_processor import PipelineProcessor

if TYPE_CHECKING:
    from .pipeline_manager import PipelineManager

logger = logging.getLogger(__name__)

# Default queue sizes (match pipeline_processor)
# Use larger size for inter-pipeline queues so downstream can accumulate a full chunk
DEFAULT_INPUT_QUEUE_MAXSIZE = 64
DEFAULT_OUTPUT_QUEUE_MAXSIZE = 8


@dataclass
class GraphRun:
    """Result of building a graph: queues and processors to run."""

    # When put(frame) is called, put to each of these queues (source fan-out)
    source_queues: list[queue.Queue]
    # The processor whose output_queue we read from for get()
    sink_processor: PipelineProcessor | None
    # All pipeline processors (for start/stop/update_parameters)
    processors: list[PipelineProcessor]
    # Pipeline IDs in graph order (for logging/events)
    pipeline_ids: list[str]
    # Node id of the sink (for clarity)
    sink_node_id: str | None
    # Node ids of output/sink nodes (e.g. "output") for preview mapping
    output_node_ids: list[str]


def build_graph(
    graph: GraphConfig,
    pipeline_manager: PipelineManager,
    initial_parameters: dict[str, Any],
    session_id: str | None = None,
    user_id: str | None = None,
    connection_id: str | None = None,
    connection_info: dict | None = None,
) -> GraphRun:
    """Build executable graph: create queues and processors, wire edges.

    Args:
        graph: Parsed graph config (nodes + edges).
        pipeline_manager: Manager to resolve pipeline_id -> instance.
        initial_parameters: Parameters passed to all pipelines.
        session_id, user_id, connection_id, connection_info: For processors.

    Returns:
        GraphRun with source_queues, sink_processor, processors, pipeline_ids.
    """
    # 1) Create one queue per edge (all edges are stream; frame-by-frame)
    stream_queues: dict[tuple[str, str], queue.Queue] = {}
    for e in graph.edges:
        if e.kind == "stream":
            stream_queues[(e.to_node, e.to_port)] = queue.Queue(
                maxsize=DEFAULT_INPUT_QUEUE_MAXSIZE
            )

    # 2) Source queues: all queues that receive from a source node
    source_queues: list[queue.Queue] = []
    for node_id in graph.get_source_node_ids():
        for e in graph.stream_edges_from(node_id):
            q = stream_queues.get((e.to_node, e.to_port))
            if q is not None:
                source_queues.append(q)

    # 3) Create a processor per pipeline node and wire input_queues per port
    node_processors: dict[str, PipelineProcessor] = {}
    pipeline_ids: list[str] = []

    for node in graph.nodes:
        if node.type != "pipeline" or node.pipeline_id is None:
            continue
        pipeline = pipeline_manager.get_pipeline_by_id(node.pipeline_id)
        processor = PipelineProcessor(
            pipeline=pipeline,
            pipeline_id=node.pipeline_id,
            initial_parameters=initial_parameters.copy(),
            session_id=session_id,
            user_id=user_id,
            connection_id=connection_id,
            connection_info=connection_info,
            node_id=node.id,
        )
        node_processors[node.id] = processor
        pipeline_ids.append(node.pipeline_id)

        for e in graph.edges_to(node.id):
            if e.kind != "stream":
                continue
            q = stream_queues.get((node.id, e.to_port))
            if q is not None:
                processor.input_queues[e.to_port] = q

    # 4) Set each producer's output_queues per port and wire consumer input to same queue
    for node in graph.nodes:
        if node.type != "pipeline" or node.id not in node_processors:
            continue
        proc = node_processors[node.id]
        out_by_port: dict[str, list[queue.Queue]] = {}
        for e in graph.edges_from(node.id):
            if e.kind != "stream":
                continue
            q = stream_queues.get((e.to_node, e.to_port))
            if q is not None and q not in out_by_port.get(e.from_port, []):
                out_by_port.setdefault(e.from_port, []).append(q)
                # Symmetric wiring: ensure consumer reads from this queue
                consumer = node_processors.get(e.to_node)
                if consumer is not None:
                    consumer.input_queues[e.to_port] = q
        for port, qlist in out_by_port.items():
            proc.output_queues[port] = qlist

    # 4b) Remove unwired default input queues so processors only wait for
    # ports that are actually connected via stream edges.
    for node in graph.nodes:
        if node.type != "pipeline" or node.id not in node_processors:
            continue
        proc = node_processors[node.id]
        wired_ports = {e.to_port for e in graph.edges_to(node.id) if e.kind == "stream"}
        if wired_ports:
            proc.input_queues = {
                port: q for port, q in proc.input_queues.items() if port in wired_ports
            }

    # 4c) Resize inter-pipeline queues based on downstream pipeline requirements
    _resize_graph_queues(graph, node_processors)

    # 5) Identify sink: node that has an edge to "output" (or type sink)
    sink_node_id: str | None = None
    for e in graph.edges:
        if e.to_node == "output" and e.kind == "stream":
            sink_node_id = e.from_node
            break
    if sink_node_id is None:
        # Check other sink nodes
        for sink_id in graph.get_sink_node_ids():
            for e in graph.edges:
                if e.to_node == sink_id and e.kind == "stream":
                    sink_node_id = e.from_node
                    break
            if sink_node_id:
                break
    if sink_node_id is None:
        # No explicit output node: treat last pipeline node as sink (linear)
        pipeline_node_ids = graph.get_pipeline_node_ids()
        if pipeline_node_ids:
            sink_node_id = pipeline_node_ids[-1]

    sink_processor = node_processors.get(sink_node_id) if sink_node_id else None
    if sink_node_id and sink_processor is None:
        logger.warning(
            "Graph sink node %s not found in processors (missing pipeline?)",
            sink_node_id,
        )
    elif sink_node_id:
        logger.info("Graph sink for playback: node_id=%s", sink_node_id)

    # 5b) Resize sink's default output queues (initially 8) to match
    # inter-pipeline queue size so output batches don't get dropped.
    if sink_processor is not None:
        for port, qlist in list(sink_processor.output_queues.items()):
            if qlist and qlist[0].maxsize < DEFAULT_INPUT_QUEUE_MAXSIZE:
                sink_processor.output_queues[port] = [
                    queue.Queue(maxsize=DEFAULT_INPUT_QUEUE_MAXSIZE)
                ]

    # Collect output/sink node IDs for preview mapping
    output_node_ids = [n.id for n in graph.nodes if n.type == "sink"]

    return GraphRun(
        source_queues=source_queues,
        sink_processor=sink_processor,
        processors=list(node_processors.values()),
        pipeline_ids=pipeline_ids,
        sink_node_id=sink_node_id,
        output_node_ids=output_node_ids,
    )


def _resize_graph_queues(
    graph: GraphConfig,
    node_processors: dict[str, PipelineProcessor],
) -> None:
    """Resize inter-pipeline queues based on downstream pipeline requirements.

    For each pipeline node, calls prepare(video=True) to determine input_size,
    then ensures its input queues are large enough to buffer
    input_size * OUTPUT_QUEUE_MAX_SIZE_FACTOR frames.
    """
    from .pipeline_processor import OUTPUT_QUEUE_MAX_SIZE_FACTOR

    for node in graph.nodes:
        if node.type != "pipeline" or node.id not in node_processors:
            continue
        proc = node_processors[node.id]
        pipeline = proc.pipeline
        if not hasattr(pipeline, "prepare"):
            continue
        try:
            requirements = pipeline.prepare(video=True)
        except Exception:
            continue
        if requirements is None:
            continue

        target_size = max(
            DEFAULT_INPUT_QUEUE_MAXSIZE,
            requirements.input_size * OUTPUT_QUEUE_MAX_SIZE_FACTOR,
        )

        for port, old_q in list(proc.input_queues.items()):
            if old_q.maxsize >= target_size:
                continue
            new_q = queue.Queue(maxsize=target_size)
            proc.input_queues[port] = new_q
            logger.info(
                "Node %s port '%s': resized queue %d -> %d",
                node.id,
                port,
                old_q.maxsize,
                target_size,
            )
            # Update the producing processor's output_queues reference
            for e in graph.edges_to(node.id):
                if e.to_port != port or e.kind != "stream":
                    continue
                producer = node_processors.get(e.from_node)
                if producer is None:
                    continue
                out_list = producer.output_queues.get(e.from_port, [])
                for i, oq in enumerate(out_list):
                    if oq is old_q:
                        out_list[i] = new_q
