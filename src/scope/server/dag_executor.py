"""DAG executor: builds and runs pipeline graphs from DagConfig.

Wires source queues (for put()), pipeline processors (one per pipeline node),
and identifies the sink for get(). All frame ports (video, vace_input_frames,
vace_input_masks) use stream edges (queues); no separate parameter path.
"""

from __future__ import annotations

import logging
import queue
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .dag_schema import DagConfig, DagEdge, DagNode
from .pipeline_processor import PipelineProcessor

if TYPE_CHECKING:
    from .pipeline_manager import PipelineManager

logger = logging.getLogger(__name__)

# Default queue sizes (match pipeline_processor)
# Use larger size for inter-pipeline queues so downstream can accumulate a full chunk
DEFAULT_INPUT_QUEUE_MAXSIZE = 64
DEFAULT_OUTPUT_QUEUE_MAXSIZE = 8


@dataclass
class DagRun:
    """Result of building a DAG: queues and processors to run."""

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


def build_dag(
    dag: DagConfig,
    pipeline_manager: PipelineManager,
    initial_parameters: dict[str, Any],
    session_id: str | None = None,
    user_id: str | None = None,
    connection_id: str | None = None,
    connection_info: dict | None = None,
) -> DagRun:
    """Build executable DAG: create queues and processors, wire edges.

    Args:
        dag: Parsed DAG config (nodes + edges).
        pipeline_manager: Manager to resolve pipeline_id -> instance.
        initial_parameters: Parameters passed to all pipelines.
        session_id, user_id, connection_id, connection_info: For processors.

    Returns:
        DagRun with source_queues, sink_processor, processors, pipeline_ids.
    """
    # 1) Create one queue per edge (all edges are stream; frame-by-frame)
    stream_queues: dict[tuple[str, str], queue.Queue] = {}
    for e in dag.edges:
        if e.kind == "stream":
            stream_queues[(e.to_node, e.to_port)] = queue.Queue(
                maxsize=DEFAULT_INPUT_QUEUE_MAXSIZE
            )

    # 2) Source queues: all queues that receive from a source node
    source_queues: list[queue.Queue] = []
    for node_id in dag.get_source_node_ids():
        for e in dag.stream_edges_from(node_id):
            q = stream_queues.get((e.to_node, e.to_port))
            if q is not None:
                source_queues.append(q)

    # 3) Create a processor per pipeline node and wire input_queues per port
    node_processors: dict[str, PipelineProcessor] = {}
    pipeline_ids: list[str] = []

    for node in dag.nodes:
        if node.type != "pipeline" or node.pipeline_id is None:
            continue
        pipeline = pipeline_manager.get_pipeline_by_id(node.pipeline_id)
        processor = PipelineProcessor(
            pipeline=pipeline,
            pipeline_id=node.id,
            initial_parameters=initial_parameters.copy(),
            session_id=session_id,
            user_id=user_id,
            connection_id=connection_id,
            connection_info=connection_info,
        )
        node_processors[node.id] = processor
        pipeline_ids.append(node.pipeline_id)

        for e in dag.edges_to(node.id):
            if e.kind != "stream":
                continue
            q = stream_queues.get((node.id, e.to_port))
            if q is not None:
                processor.input_queues[e.to_port] = q
        with processor.input_queue_lock:
            processor.input_queue = processor.input_queues.get("video")

    # 4) Set each producer's output_queues per port and wire consumer input to same queue
    for node in dag.nodes:
        if node.type != "pipeline" or node.id not in node_processors:
            continue
        proc = node_processors[node.id]
        out_by_port: dict[str, list[queue.Queue]] = {}
        for e in dag.edges_from(node.id):
            if e.kind != "stream":
                continue
            q = stream_queues.get((e.to_node, e.to_port))
            if q is not None and q not in out_by_port.get(e.from_port, []):
                out_by_port.setdefault(e.from_port, []).append(q)
                # Symmetric wiring: ensure consumer reads from this queue (fixes chained pipelines)
                consumer = node_processors.get(e.to_node)
                if consumer is not None:
                    consumer.input_queues[e.to_port] = q
                    with consumer.input_queue_lock:
                        consumer.input_queue = consumer.input_queues.get("video")
        for port, qlist in out_by_port.items():
            proc.output_queues[port] = qlist

    # 4) Identify sink: node that has an edge to "output" (or type sink)
    sink_node_id: str | None = None
    for e in dag.edges:
        if e.to_node == "output" and e.kind == "stream":
            sink_node_id = e.from_node
            break
    if sink_node_id is None:
        # No explicit output node: treat last pipeline node as sink (linear)
        pipeline_node_ids = dag.get_pipeline_node_ids()
        if pipeline_node_ids:
            sink_node_id = pipeline_node_ids[-1]

    sink_processor = node_processors.get(sink_node_id) if sink_node_id else None
    if sink_node_id and sink_processor is None:
        logger.warning(
            "DAG sink node %s not found in processors (missing pipeline?)",
            sink_node_id,
        )
    elif sink_node_id:
        logger.info("DAG sink for playback: node_id=%s", sink_node_id)

    # Collect output/sink node IDs for preview mapping
    output_node_ids = [n.id for n in dag.nodes if n.type == "sink"]

    return DagRun(
        source_queues=source_queues,
        sink_processor=sink_processor,
        processors=list(node_processors.values()),
        pipeline_ids=pipeline_ids,
        sink_node_id=sink_node_id,
        output_node_ids=output_node_ids,
    )


def linear_dag_from_pipeline_ids(pipeline_ids: list[str]) -> DagConfig:
    """Build a linear DAG from a list of pipeline IDs (backward compatibility).

    Produces: source -> p0 -> p1 -> ... -> sink.
    """
    nodes = [
        DagNode(id="input", type="source"),
        *[DagNode(id=pid, type="pipeline", pipeline_id=pid) for pid in pipeline_ids],
        DagNode(id="output", type="sink"),
    ]
    edges: list[DagEdge] = []
    prev = "input"
    for pid in pipeline_ids:
        edges.append(
            DagEdge(
                from_node=prev,
                from_port="video",
                to_node=pid,
                to_port="video",
                kind="stream",
            )
        )
        prev = pid
    edges.append(
        DagEdge(
            from_node=prev,
            from_port="video",
            to_node="output",
            to_port="video",
            kind="stream",
        )
    )
    # Vace streams: preprocessor outputs (vace_*) to next in chain (frame-by-frame queues)
    for i in range(len(pipeline_ids) - 1):
        for port in ("vace_input_frames", "vace_input_masks"):
            edges.append(
                DagEdge(
                    from_node=pipeline_ids[i],
                    from_port=port,
                    to_node=pipeline_ids[i + 1],
                    to_port=port,
                    kind="stream",
                )
            )
    return DagConfig(nodes=nodes, edges=edges)
