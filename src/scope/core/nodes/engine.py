"""Graph engine — tick-loop execution of data-node graphs.

The engine evaluates all non-stream nodes in topological order at a
configurable tick rate (default 60 Hz).  Stream nodes (pipeline bridges,
source, sink) are skipped during tick evaluation — they only receive
parameter updates when upstream data nodes change.

The tick loop runs in a dedicated daemon thread and reads
``BeatState`` from the active ``TempoSource`` each tick for
clock-accurate animated patterns.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable
from threading import Event, RLock, Thread
from typing import TYPE_CHECKING, Any

from .schema import EdgeInstance, GraphDefinition, NodeInstance

if TYPE_CHECKING:
    from scope.server.tempo_sync import BeatState, TempoSync

    from .base import BaseNode

logger = logging.getLogger(__name__)


class GraphEngine:
    """Core execution engine with a tick loop for data-node evaluation."""

    def __init__(
        self,
        session_id: str,
        tempo_sync: TempoSync | None = None,
    ) -> None:
        self._session_id = session_id
        self._nodes: dict[str, BaseNode] = {}
        self._edges: list[EdgeInstance] = []
        self._eval_order: list[str] = []  # Topological sort (param edges only)
        self._computed: dict[str, dict[str, Any]] = {}  # node_id → {port: value}
        self._tick_thread: Thread | None = None
        self._shutdown = Event()
        self._lock = RLock()
        self._tick_rate: float = 60.0  # Hz
        self._tempo_sync = tempo_sync
        self._start_time: float = 0.0

        # Callbacks
        self._value_change_callback: (
            Callable[[dict[str, dict[str, Any]]], None] | None
        ) = None

        # Pipeline bridge support
        self._pipeline_param_callback: Callable[[str, dict[str, Any]], None] | None = (
            None
        )

    # ------------------------------------------------------------------
    # Graph lifecycle
    # ------------------------------------------------------------------

    def load_graph(self, graph_def: GraphDefinition) -> None:
        """Instantiate nodes, compute topological order, wire edges."""
        from .registry import NodeRegistry

        with self._lock:
            self._nodes.clear()
            self._edges = list(graph_def.edges)
            self._computed.clear()

            for node_inst in graph_def.nodes:
                node_cls = NodeRegistry.get(node_inst.type)
                if node_cls is None:
                    logger.warning(
                        "Unknown node type %r for node %r — skipping",
                        node_inst.type,
                        node_inst.id,
                    )
                    continue
                node = node_cls(node_inst.id, dict(node_inst.config))
                # Stash inner graph data for subgraph nodes
                if node_inst.inner_nodes is not None:
                    node.config["_inner_nodes"] = node_inst.inner_nodes
                    node.config["_inner_edges"] = node_inst.inner_edges or []
                if node_inst.pipeline_id is not None:
                    node.config["_pipeline_id"] = node_inst.pipeline_id
                self._nodes[node_inst.id] = node

            self._eval_order = self._topological_sort()
            logger.info(
                "Graph loaded: %d nodes, %d edges, eval order: %s",
                len(self._nodes),
                len(self._edges),
                self._eval_order,
            )

    def start(self) -> None:
        """Start the tick-loop thread."""
        if self._tick_thread is not None and self._tick_thread.is_alive():
            return
        self._shutdown.clear()
        self._start_time = time.monotonic()
        self._tick_thread = Thread(
            target=self._tick_loop,
            name=f"graph-engine-{self._session_id}",
            daemon=True,
        )
        self._tick_thread.start()
        logger.info("Graph engine started for session %s", self._session_id)

    def stop(self) -> None:
        """Stop the tick loop and clean up."""
        self._shutdown.set()
        if self._tick_thread is not None:
            self._tick_thread.join(timeout=2.0)
            self._tick_thread = None
        logger.info("Graph engine stopped for session %s", self._session_id)

    # ------------------------------------------------------------------
    # Tick loop
    # ------------------------------------------------------------------

    def _tick_loop(self) -> None:
        """Main loop running at ``_tick_rate`` Hz."""
        interval = 1.0 / self._tick_rate
        while not self._shutdown.is_set():
            tick_start = time.monotonic()
            tick_time = tick_start - self._start_time

            beat_state: BeatState | None = None
            if self._tempo_sync is not None:
                beat_state = self._tempo_sync.get_beat_state()

            changed = self._evaluate_tick(tick_time, beat_state)

            # Notify frontend of changed values
            if changed and self._value_change_callback:
                try:
                    self._value_change_callback(changed)
                except Exception:
                    logger.exception("Error in value change callback")

            # Forward pipeline bridge parameters
            self._forward_pipeline_params()

            # Sleep to maintain tick rate
            elapsed = time.monotonic() - tick_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                self._shutdown.wait(sleep_time)

    def _evaluate_tick(
        self, tick_time: float, beat_state: BeatState | None
    ) -> dict[str, dict[str, Any]]:
        """Evaluate all data nodes in topological order.

        Returns only nodes whose outputs changed since the last tick.
        """
        changed: dict[str, dict[str, Any]] = {}

        with self._lock:
            for node_id in self._eval_order:
                node = self._nodes.get(node_id)
                if node is None:
                    continue

                defn = node.get_definition()
                # Skip stream nodes — they're handled by the pipeline system
                if defn.is_stream_node:
                    continue

                # Gather inputs from upstream computed values
                inputs = self._gather_inputs(node_id)

                # Handle subgraph nodes specially
                if node.node_type_id == "subgraph":
                    outputs = self._evaluate_subgraph(
                        node, inputs, tick_time, beat_state
                    )
                else:
                    try:
                        outputs = node.execute(inputs, tick_time, beat_state)
                    except Exception:
                        logger.exception(
                            "Error executing node %s (%s)", node_id, node.node_type_id
                        )
                        outputs = {}

                # Detect changes
                prev = self._computed.get(node_id, {})
                if outputs != prev:
                    changed[node_id] = outputs
                self._computed[node_id] = outputs

        return changed

    def _gather_inputs(self, node_id: str) -> dict[str, Any]:
        """Collect input values for a node from its incoming param edges."""
        inputs: dict[str, Any] = {}
        for edge in self._edges:
            if edge.target == node_id and edge.kind == "param":
                source_outputs = self._computed.get(edge.source, {})
                value = source_outputs.get(edge.source_port)
                inputs[edge.target_port] = value
        return inputs

    def _forward_pipeline_params(self) -> None:
        """Forward pending parameters from pipeline bridge nodes."""
        if self._pipeline_param_callback is None:
            return

        with self._lock:
            for _node_id, node in self._nodes.items():
                if node.node_type_id == "pipeline":
                    from .builtin.pipeline_bridge_node import PipelineBridgeNode

                    if isinstance(node, PipelineBridgeNode):
                        params = node.get_pending_params()
                        if params:
                            pipeline_id = node.config.get("_pipeline_id")
                            if pipeline_id:
                                try:
                                    self._pipeline_param_callback(pipeline_id, params)
                                except Exception:
                                    logger.exception(
                                        "Error forwarding params to pipeline %s",
                                        pipeline_id,
                                    )

    # ------------------------------------------------------------------
    # Subgraph evaluation
    # ------------------------------------------------------------------

    def _evaluate_subgraph(
        self,
        node: BaseNode,
        inputs: dict[str, Any],
        tick_time: float,
        beat_state: BeatState | None,
    ) -> dict[str, Any]:
        """Recursively evaluate a subgraph node's inner graph."""
        from .builtin.subgraph_io_nodes import SubgraphInputNode, SubgraphOutputNode
        from .builtin.subgraph_node import SubgraphNode
        from .registry import NodeRegistry

        inner_node_defs: list[NodeInstance] = node.config.get("_inner_nodes", [])
        inner_edge_defs: list[EdgeInstance] = node.config.get("_inner_edges", [])

        if not inner_node_defs:
            return {}

        # Parse inner edge defs if they're plain dicts
        inner_edges: list[EdgeInstance] = []
        for e in inner_edge_defs:
            if isinstance(e, EdgeInstance):
                inner_edges.append(e)
            elif isinstance(e, dict):
                inner_edges.append(EdgeInstance.model_validate(e))

        # Instantiate inner nodes
        inner_nodes: dict[str, BaseNode] = {}
        for ni in inner_node_defs:
            if isinstance(ni, dict):
                ni = NodeInstance.model_validate(ni)
            cls = NodeRegistry.get(ni.type)
            if cls is None:
                continue
            inner_nodes[ni.id] = cls(ni.id, dict(ni.config))

        # Seed boundary inputs
        for nid, n in inner_nodes.items():
            if isinstance(n, SubgraphInputNode):
                # Find which external input maps to this boundary node
                port_name = n.config.get("portName", nid)
                if port_name in inputs:
                    n.set_boundary_value(port_name, inputs[port_name])

        # Topological sort of inner graph (param edges only)
        eval_order = _topological_sort_edges(
            list(inner_nodes.keys()),
            [e for e in inner_edges if e.kind == "param"],
        )

        # Evaluate
        computed: dict[str, dict[str, Any]] = {}
        for nid in eval_order:
            n = inner_nodes.get(nid)
            if n is None:
                continue
            defn = n.get_definition()
            if defn.is_stream_node:
                continue

            # Gather inputs from inner edges
            inp: dict[str, Any] = {}
            for e in inner_edges:
                if e.target == nid and e.kind == "param":
                    src_out = computed.get(e.source, {})
                    inp[e.target_port] = src_out.get(e.source_port)

            try:
                out = n.execute(inp, tick_time, beat_state)
            except Exception:
                logger.exception("Error in subgraph node %s", nid)
                out = {}
            computed[nid] = out

        # Collect boundary outputs
        outputs: dict[str, Any] = {}
        for nid, n in inner_nodes.items():
            if isinstance(n, SubgraphOutputNode):
                port_name = n.config.get("portName", nid)
                val = computed.get(nid, {}).get("value")
                outputs[port_name] = val

        # Store on the subgraph node
        if isinstance(node, SubgraphNode):
            node.set_inner_outputs(outputs)

        return outputs

    # ------------------------------------------------------------------
    # Topological sort
    # ------------------------------------------------------------------

    def _topological_sort(self) -> list[str]:
        """Kahn's algorithm over param edges only."""
        return _topological_sort_edges(
            list(self._nodes.keys()),
            [e for e in self._edges if e.kind == "param"],
        )

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def handle_event(
        self, node_id: str, event_type: str, payload: dict[str, Any]
    ) -> None:
        """Route a frontend interaction to the target node."""
        with self._lock:
            node = self._nodes.get(node_id)
            if node is not None:
                node.on_event(event_type, payload)

    def get_all_values(self) -> dict[str, dict[str, Any]]:
        """Full snapshot of all computed node outputs (for initial sync)."""
        with self._lock:
            return dict(self._computed)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _topological_sort_edges(
    node_ids: list[str], edges: list[EdgeInstance]
) -> list[str]:
    """Kahn's algorithm for topological sort."""
    in_degree: dict[str, int] = defaultdict(int)
    adjacency: dict[str, list[str]] = defaultdict(list)

    node_set = set(node_ids)
    for nid in node_ids:
        in_degree.setdefault(nid, 0)

    for edge in edges:
        if edge.source in node_set and edge.target in node_set:
            adjacency[edge.source].append(edge.target)
            in_degree[edge.target] += 1

    queue: deque[str] = deque()
    for nid in node_ids:
        if in_degree[nid] == 0:
            queue.append(nid)

    result: list[str] = []
    while queue:
        nid = queue.popleft()
        result.append(nid)
        for neighbor in adjacency[nid]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(node_set):
        logger.warning(
            "Cycle detected in graph — %d nodes in cycle",
            len(node_set) - len(result),
        )
        # Add remaining nodes at the end (best-effort)
        for nid in node_ids:
            if nid not in result:
                result.append(nid)

    return result
