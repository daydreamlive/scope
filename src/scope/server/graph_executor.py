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

from scope.core.nodes.registry import NodeRegistry

from .graph_schema import GraphConfig
from .pipeline_processor import PipelineProcessor

if TYPE_CHECKING:
    from .pipeline_manager import PipelineManager

logger = logging.getLogger(__name__)

# Default queue sizes (match pipeline_processor)
# Use larger size for inter-pipeline queues so downstream can accumulate a full chunk
DEFAULT_INPUT_QUEUE_MAXSIZE = 30
DEFAULT_OUTPUT_QUEUE_MAXSIZE = 8


@dataclass
class GraphRun:
    """Result of building a graph: queues and processors to run."""

    # When put(frame) is called, put to each of these queues (source fan-out)
    source_queues: list[queue.Queue]
    # Per-source-node queues: source_node_id -> list of queues fed by that source
    source_queues_by_node: dict[str, list[queue.Queue]]
    # The processor whose output_queue we read from for get() (first sink, backward compat)
    sink_processor: PipelineProcessor | None
    # Per-sink-node output queues: sink_node_id -> queue for WebRTC / get_from_sink()
    sink_queues_by_node: dict[str, queue.Queue]
    # When sink_mode is ndi/spout/syphon: separate queue for Spout/NDI/Syphon threads so
    # they do not race WebRTC on the same queue (each frame would otherwise be consumed once).
    sink_hardware_queues_by_node: dict[str, queue.Queue]
    # Per-sink-node feeder processors: sink_node_id -> PipelineProcessor that feeds it
    sink_processors_by_node: dict[str, PipelineProcessor]
    # All pipeline processors (for start/stop/update_parameters)
    processors: list[PipelineProcessor]
    # Pipeline IDs in graph order (for logging/events)
    pipeline_ids: list[str]
    # Node id of the sink (for clarity)
    sink_node_id: str | None
    # Node ids of output/sink nodes (e.g. "output") for preview mapping
    output_node_ids: list[str]
    # Per-record-node output queues: record_node_id -> output queue
    record_queues_by_node: dict[str, queue.Queue]


def build_graph(
    graph: GraphConfig,
    pipeline_manager: PipelineManager,
    initial_parameters: dict[str, Any],
    session_id: str | None = None,
    user_id: str | None = None,
    connection_id: str | None = None,
    connection_info: dict | None = None,
    tempo_sync: Any | None = None,
    modulation_engine: Any | None = None,
) -> GraphRun:
    """Build executable graph: create queues and processors, wire edges.

    Args:
        graph: Parsed graph config (nodes + edges).
        pipeline_manager: Manager to resolve pipeline_id -> instance.
        initial_parameters: Parameters passed to all pipelines.
        session_id, user_id, connection_id, connection_info: For processors.

    Returns:
        GraphRun with source_queues, sink_processor, processors, pipeline_ids.

    Raises:
        ValueError: If the graph structure is invalid.
    """
    errors = graph.validate_structure()
    if errors:
        raise ValueError(f"Invalid graph: {'; '.join(errors)}")

    # Validate edge ports against pipeline input/output declarations
    _validate_edge_ports(graph, pipeline_manager)

    # Sink and record nodes get dedicated queues in steps 4–5, so we
    # exclude them here to avoid false-positive duplicate-edge errors
    # when multiple pipelines fan-in to the same sink.
    _sink_record_ids = {n.id for n in graph.nodes if n.type in ("sink", "record")}

    # 1) Create one queue per edge (all edges are stream; frame-by-frame).
    # Node→node edges use maxsize=1 so the DAG executes one cycle at a
    # time and large tensors don't pile up in memory; pipeline edges use
    # the larger default so video chunks can accumulate.
    _node_ids = {n.id for n in graph.nodes if n.type == "node"}
    stream_queues: dict[tuple[str, str], queue.Queue] = {}
    for e in graph.edges:
        if e.kind == "stream":
            if e.to_node in _sink_record_ids:
                continue
            key = (e.to_node, e.to_port)
            if key in stream_queues:
                raise ValueError(
                    f"Duplicate stream edges to the same input port: "
                    f"node={e.to_node!r}, port={e.to_port!r}. "
                    f"Fan-in to a single port is not supported."
                )
            both_nodes = e.from_node in _node_ids and e.to_node in _node_ids
            size = 1 if both_nodes else DEFAULT_INPUT_QUEUE_MAXSIZE
            stream_queues[key] = queue.Queue(maxsize=size)

    # 2) Create a processor per pipeline/custom node and wire input_queues
    node_processors: dict[str, Any] = {}  # PipelineProcessor | NodeProcessor
    pipeline_ids: list[str] = []

    # Per-pipeline tempo: if any node explicitly opts in via tempo_sync=True,
    # only those nodes get beat injection. Otherwise fall back to global
    # behaviour (all get tempo) for backwards compatibility with perform mode
    # and older saved workflows.
    any_node_has_tempo = any(n.tempo_sync for n in graph.nodes if n.type == "pipeline")

    for node in graph.nodes:
        if node.type == "pipeline" and node.pipeline_id is not None:
            node_gets_tempo = node.tempo_sync or not any_node_has_tempo
            pipeline = pipeline_manager.get_pipeline_by_id(node.id)
            processor = PipelineProcessor(
                pipeline=pipeline,
                pipeline_id=node.pipeline_id,
                initial_parameters=initial_parameters.copy(),
                session_id=session_id,
                user_id=user_id,
                connection_id=connection_id,
                connection_info=connection_info,
                tempo_sync=tempo_sync if node_gets_tempo else None,
                modulation_engine=modulation_engine if node_gets_tempo else None,
                node_id=node.id,
            )
            node_processors[node.id] = processor
            pipeline_ids.append(node.pipeline_id)
        elif node.type == "node" and node.node_type_id is not None:
            from scope.core.nodes.processor import NodeProcessor

            node_cls = NodeRegistry.get(node.node_type_id)
            if node_cls is None:
                raise ValueError(
                    f"Unknown node type '{node.node_type_id}' for node '{node.id}'"
                )
            node_instance = node_cls(node_id=node.id)
            # Merge per-node params (from workflow) with global initial params.
            # Per-node values take precedence (e.g. "steps": 8 on DiffusionConfig).
            node_params = {**initial_parameters}
            if node.params:
                node_params.update(node.params)
            processor = NodeProcessor(
                node=node_instance,
                node_id=node.id,
                initial_parameters=node_params,
            )
            node_processors[node.id] = processor
            pipeline_ids.append(f"node:{node.node_type_id}")
        else:
            continue

        for e in graph.edges_to(node.id):
            if e.kind != "stream":
                continue
            q = stream_queues.get((node.id, e.to_port))
            if q is not None:
                processor.input_queues[e.to_port] = q
                # Mark audio ports so the processor consumes them differently
                if e.to_port == "audio":
                    processor.audio_input_ports.add(e.to_port)

    # 3) Set each producer's output_queues per port
    for node in graph.nodes:
        if node.type not in ("pipeline", "node") or node.id not in node_processors:
            continue
        proc = node_processors[node.id]
        out_by_port: dict[str, list[queue.Queue]] = {}
        for e in graph.edges_from(node.id):
            if e.kind != "stream":
                continue
            q = stream_queues.get((e.to_node, e.to_port))
            if q is not None and q not in out_by_port.get(e.from_port, []):
                out_by_port.setdefault(e.from_port, []).append(q)
        for port, qlist in out_by_port.items():
            proc.output_queues[port] = qlist

    # 3c) Populate output_consumers so processors can update downstream
    # input queue references at runtime when output queues are resized.
    for e in graph.edges:
        if e.kind != "stream":
            continue
        producer = node_processors.get(e.from_node)
        consumer = node_processors.get(e.to_node)
        if producer is not None and consumer is not None:
            producer.output_consumers.setdefault(e.from_port, []).append(
                (consumer, e.to_port)
            )

    # 3d) Resize inter-pipeline queues based on downstream pipeline requirements
    _resize_graph_queues(graph, node_processors)

    # 3d2) Propagate "dynamic" marking from continuous source nodes forward.
    # A non-continuous NodeProcessor only re-fires once every dynamic input
    # port has caught up for the current upstream cycle; static ports latch
    # from cache. Without this, latch-fallback fires the node once per
    # fresh-port arrival, which can double or triple GPU work per cycle.
    _mark_dynamic_input_ports(graph, node_processors)

    # 3e) Derive source_queues from processor input_queues (post-resize).
    # Also build per-source-node queue mappings for multi-source support.
    source_queues: list[queue.Queue] = []
    source_queues_by_node: dict[str, list[queue.Queue]] = {}
    for node_id in graph.get_source_node_ids():
        per_node: list[queue.Queue] = []
        for e in graph.stream_edges_from(node_id):
            consumer = node_processors.get(e.to_node)
            if consumer is not None:
                q = consumer.input_queues.get(e.to_port)
                if q is not None:
                    if q not in source_queues:
                        source_queues.append(q)
                    if q not in per_node:
                        per_node.append(q)
        if per_node:
            source_queues_by_node[node_id] = per_node

    # 4) Identify sinks: find the pipeline nodes that feed into sink nodes.
    # Each sink gets a **dedicated** output queue appended to its feeder
    # processor's output_queues list. The processor's fan-out loop
    # (pipeline_processor.py line ~550) puts frames into every queue in the
    # list, so each sink receives its own independent copy of the frame.
    node_by_id = {n.id: n for n in graph.nodes}
    sink_queues_by_node: dict[str, queue.Queue] = {}
    sink_hardware_queues_by_node: dict[str, queue.Queue] = {}
    sink_processors_by_node: dict[str, PipelineProcessor] = {}
    for sink_id in graph.get_sink_node_ids():
        for e in graph.edges_to(sink_id):
            if e.kind == "stream":
                feeder_proc = node_processors.get(e.from_node)
                if feeder_proc is not None:
                    sink_processors_by_node[sink_id] = feeder_proc
                    # Audio edges to sinks are served via audio_output_queue,
                    # not dedicated sink queues — skip queue allocation so
                    # the feeder isn't blocked on a queue nobody drains.
                    if e.from_port == "audio" or e.to_port == "audio":
                        # Mark the feeder's audio output port as sink-bound
                        # so its _route_outputs pushes into audio_output_queue
                        # (NodeProcessor only — PipelineProcessor has its own
                        # always-on audio path via put_nowait).
                        sink_ports = getattr(feeder_proc, "audio_sink_ports", None)
                        if sink_ports is not None:
                            sink_ports.add(e.from_port)
                        break
                    sink_node = node_by_id[sink_id]
                    sink_mode = sink_node.sink_mode
                    # WebRTC preview reads sink_queues_by_node; NDI/Spout/Syphon threads
                    # must use a separate queue or they steal frames from get_from_sink().
                    sink_q = queue.Queue(maxsize=DEFAULT_INPUT_QUEUE_MAXSIZE)
                    sink_queues_by_node[sink_id] = sink_q
                    queues_to_add = [sink_q]
                    if sink_mode in ("ndi", "spout", "syphon"):
                        sink_q_hw = queue.Queue(maxsize=DEFAULT_INPUT_QUEUE_MAXSIZE)
                        sink_hardware_queues_by_node[sink_id] = sink_q_hw
                        queues_to_add.append(sink_q_hw)

                    port = e.from_port
                    feeder_proc.output_queues.setdefault(port, []).extend(queues_to_add)
                    # Register external refs so _resize_output_queue keeps
                    # SinkManager's cached queue references in sync.
                    feeder_proc.external_queue_refs.append(
                        (sink_queues_by_node, sink_id)
                    )
                    if sink_id in sink_hardware_queues_by_node:
                        feeder_proc.external_queue_refs.append(
                            (sink_hardware_queues_by_node, sink_id)
                        )
                    logger.info(
                        "Sink %s: dedicated queue(s) on %s port '%s' "
                        "(webrtc + hardware=%s, total on port: %d)",
                        sink_id,
                        e.from_node,
                        port,
                        sink_id in sink_hardware_queues_by_node,
                        len(feeder_proc.output_queues[port]),
                    )
                break

    # Backward compat: first sink determines the primary sink_processor
    sink_node_id: str | None = None
    if sink_processors_by_node:
        first_sink_id = next(iter(sink_processors_by_node))
        sink_node_id = sink_processors_by_node[first_sink_id].node_id

    if sink_node_id is None:
        # No explicit sink node: treat last pipeline node as sink (linear)
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

    # 4b) Resize sink's default output queues (initially 8) to match
    # inter-pipeline queue size so output batches don't get dropped.
    # Skip if per-sink dedicated queues were already created above.
    if sink_processor is not None and not sink_queues_by_node:
        for port, qlist in list(sink_processor.output_queues.items()):
            if qlist and qlist[0].maxsize < DEFAULT_INPUT_QUEUE_MAXSIZE:
                sink_processor.output_queues[port] = [
                    queue.Queue(maxsize=DEFAULT_INPUT_QUEUE_MAXSIZE)
                ]

    # 5) Wire record nodes: same as sinks, create dedicated output queues
    # on the feeding pipeline processor. Edges may go pipeline→record or
    # sink→record; for the latter, reuse the pipeline output that feeds the sink.
    record_queues_by_node: dict[str, queue.Queue] = {}
    for rec_id in graph.get_record_node_ids():
        for e in graph.edges_to(rec_id):
            if e.kind != "stream":
                continue
            feeder_proc = node_processors.get(e.from_node)
            port = e.from_port
            if feeder_proc is None:
                src_node = node_by_id.get(e.from_node)
                if src_node is not None and src_node.type == "sink":
                    for se in graph.edges_to(e.from_node):
                        if se.kind == "stream":
                            feeder_proc = node_processors.get(se.from_node)
                            port = se.from_port
                            break
            if feeder_proc is not None:
                rec_q = queue.Queue(maxsize=DEFAULT_INPUT_QUEUE_MAXSIZE)
                record_queues_by_node[rec_id] = rec_q
                feeder_proc.output_queues.setdefault(port, []).append(rec_q)
                feeder_proc.external_queue_refs.append((record_queues_by_node, rec_id))
                logger.info(
                    "Record %s: dedicated queue on pipeline %s port '%s' (total queues: %d)",
                    rec_id,
                    feeder_proc.node_id,
                    port,
                    len(feeder_proc.output_queues[port]),
                )
                break

    # Collect output/sink node IDs for preview mapping
    output_node_ids = [n.id for n in graph.nodes if n.type == "sink"]

    return GraphRun(
        source_queues=source_queues,
        source_queues_by_node=source_queues_by_node,
        sink_processor=sink_processor,
        sink_queues_by_node=sink_queues_by_node,
        sink_hardware_queues_by_node=sink_hardware_queues_by_node,
        sink_processors_by_node=sink_processors_by_node,
        processors=list(node_processors.values()),
        pipeline_ids=pipeline_ids,
        sink_node_id=sink_node_id,
        output_node_ids=output_node_ids,
        record_queues_by_node=record_queues_by_node,
    )


def _validate_edge_ports(
    graph: GraphConfig,
    pipeline_manager: PipelineManager,
) -> None:
    """Validate that edge ports match pipeline input/output declarations.

    Raises:
        ValueError: If any edge references an undeclared port.
    """
    # Build a map of node_id -> (declared_inputs, declared_outputs)
    port_map: dict[str, tuple[set[str], set[str]]] = {}
    for node in graph.nodes:
        if node.type == "pipeline" and node.pipeline_id is not None:
            pipeline = pipeline_manager.get_pipeline_by_id(node.id)
            config_class = pipeline.get_config_class()
            port_map[node.id] = (
                set(config_class.inputs),
                set(config_class.outputs),
            )
        elif node.type == "node" and node.node_type_id is not None:
            node_cls = NodeRegistry.get(node.node_type_id)
            if node_cls is not None:
                defn = node_cls.get_definition()
                port_map[node.id] = (
                    {p.name for p in defn.inputs},
                    {p.name for p in defn.outputs},
                )

    errors: list[str] = []
    for e in graph.edges:
        if e.kind != "stream":
            continue
        # Check output port on the producing pipeline
        if e.from_node in port_map:
            _, declared_outputs = port_map[e.from_node]
            if e.from_port not in declared_outputs:
                errors.append(
                    f"Node {e.from_node!r} has no output port {e.from_port!r} "
                    f"(declared: {sorted(declared_outputs)})"
                )
        # Check input port on the consuming pipeline
        if e.to_node in port_map:
            declared_inputs, _ = port_map[e.to_node]
            if e.to_port not in declared_inputs:
                errors.append(
                    f"Node {e.to_node!r} has no input port {e.to_port!r} "
                    f"(declared: {sorted(declared_inputs)})"
                )
    if errors:
        raise ValueError(f"Invalid graph ports: {'; '.join(errors)}")


def _mark_dynamic_input_ports(
    graph: GraphConfig,
    node_processors: dict[str, Any],
) -> None:
    """Populate ``dynamic_input_ports`` on each NodeProcessor.

    A node is "dynamic" if it is ``continuous=True`` (streaming source)
    or reachable from one through stream edges. An input port on a
    non-continuous node is "dynamic" when its upstream producer is
    dynamic — meaning fresh values are expected once per upstream cycle.
    Ports whose upstream is a one-shot static node (e.g. ``LoadModel``,
    ``DiffusionConfig``) stay empty and fall back to latch-from-cache.

    PipelineProcessor has its own tick loop and does not use this
    attribute; we guard on ``hasattr`` rather than isinstance to avoid
    dragging a type import into graph_executor.
    """
    dynamic_nodes: set[str] = set()
    for node_id, proc in node_processors.items():
        node = getattr(proc, "node", None)
        if node is None:
            continue
        try:
            if node.get_definition().continuous:
                dynamic_nodes.add(node_id)
        except Exception:  # pragma: no cover — defensive
            continue

    # Fixed-point forward propagation over stream edges.
    changed = True
    while changed:
        changed = False
        for e in graph.edges:
            if e.kind != "stream":
                continue
            if e.from_node in dynamic_nodes and e.to_node not in dynamic_nodes:
                if e.to_node in node_processors:
                    dynamic_nodes.add(e.to_node)
                    changed = True

    for node_id, proc in node_processors.items():
        if not hasattr(proc, "dynamic_input_ports"):
            continue
        dynamic_ports: set[str] = set()
        for e in graph.edges:
            if e.kind != "stream" or e.to_node != node_id:
                continue
            if e.from_node in dynamic_nodes:
                dynamic_ports.add(e.to_port)
        proc.dynamic_input_ports = dynamic_ports
        if dynamic_ports:
            logger.debug(
                "Node %s: dynamic input ports = %s",
                node_id,
                sorted(dynamic_ports),
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
            logger.warning(
                "Failed to call prepare() on pipeline %s, skipping queue resize",
                node.id,
                exc_info=True,
            )
            continue
        if requirements is None:
            continue

        target_size = max(
            DEFAULT_INPUT_QUEUE_MAXSIZE,
            requirements.input_size * OUTPUT_QUEUE_MAX_SIZE_FACTOR,
        )

        with proc.input_queue_lock:
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
