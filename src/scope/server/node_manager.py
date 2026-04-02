"""Node manager for backend node lifecycle and value routing.

The ``NodeManager`` is the execution engine for backend nodes. It:
- Creates and tears down node instances from a ``GraphConfig``.
- Builds a routing table from graph edges so that when a node emits an output,
  the manager forwards the value to connected backend nodes (in-process) or to
  pipeline processors (via ``FrameProcessor.update_parameters``).
- Optionally broadcasts output state to WebSocket clients for frontend
  observation; execution does **not** depend on any frontend being connected.
"""

from __future__ import annotations

import logging
import queue as stdlib_queue
import threading
from typing import TYPE_CHECKING, Any

from scope.core.nodes import NodeRegistry

if TYPE_CHECKING:
    from scope.server.frame_processor import FrameProcessor
    from scope.server.graph_schema import GraphConfig

logger = logging.getLogger(__name__)


class _EdgeTarget:
    """A resolved target for an output edge."""

    __slots__ = ("target_type", "instance_id", "port_name")

    def __init__(
        self,
        target_type: str,
        instance_id: str,
        port_name: str,
    ) -> None:
        self.target_type = target_type  # "node" or "pipeline"
        self.instance_id = instance_id
        self.port_name = port_name


class NodeManager:
    """Manages backend node instances and routes values between them."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._instances: dict[str, Any] = {}  # instance_id -> BaseNode
        self._routing_table: dict[str, list[_EdgeTarget]] = {}
        self._frame_processor: FrameProcessor | None = None
        self._ws_clients: list[stdlib_queue.Queue] = []
        self._node_states: dict[str, dict[str, Any]] = {}

    def set_frame_processor(self, fp: FrameProcessor) -> None:
        """Wire the frame processor so node outputs can reach pipelines."""
        self._frame_processor = fp

    # ------------------------------------------------------------------
    # Graph lifecycle
    # ------------------------------------------------------------------

    def load_graph(self, graph_config: GraphConfig) -> None:
        """Atomically create node instances and build the routing table.

        Call this from the pipeline load flow so that one API call boots
        both pipelines (``PipelineManager``) and nodes (``NodeManager``).
        """
        with self._lock:
            self._teardown_all()

            node_nodes = [n for n in graph_config.nodes if n.type == "node"]
            if not node_nodes:
                return

            for gnode in node_nodes:
                node_type_id = gnode.node_type_id
                if not node_type_id:
                    logger.warning(
                        f"Node '{gnode.id}' has type 'node' but no node_type_id"
                    )
                    continue

                node_cls = NodeRegistry.get(node_type_id)
                if node_cls is None:
                    logger.warning(
                        f"Unknown node type '{node_type_id}' for node '{gnode.id}'"
                    )
                    continue

                instance = node_cls()
                self._instances[gnode.id] = instance
                self._node_states[gnode.id] = {}

            self._build_routing_table(graph_config)

            for instance_id, instance in self._instances.items():
                emitter = self._make_emitter(instance_id)
                try:
                    instance.setup(emitter)
                except Exception:
                    logger.exception(f"Failed to setup node instance '{instance_id}'")

            logger.info(f"NodeManager loaded {len(self._instances)} node instance(s)")

        self._broadcast_full_state()

    def unload_graph(self) -> None:
        """Tear down all node instances."""
        with self._lock:
            self._teardown_all()
        self._broadcast_full_state()

    def remove_instance(self, instance_id: str) -> bool:
        """Destroy a single node instance and clean up its routing/state."""
        with self._lock:
            instance = self._instances.pop(instance_id, None)
            if instance is None:
                return False
            try:
                instance.teardown()
            except Exception:
                logger.exception(f"Error tearing down node '{instance_id}'")
            self._node_states.pop(instance_id, None)
            keys_to_remove = [
                k for k in self._routing_table if k.startswith(f"{instance_id}:")
            ]
            for k in keys_to_remove:
                del self._routing_table[k]
        self._broadcast_full_state()
        return True

    def stop_all_nodes(self) -> None:
        """Stop all node instances (e.g. when the stream/pipeline stops).

        Calls ``on_stream_stop()`` on every live instance. Instances are NOT
        destroyed — they can be restarted if the stream resumes.
        """
        with self._lock:
            instances = list(self._instances.values())
        for instance in instances:
            try:
                instance.on_stream_stop()
            except Exception:
                logger.exception("Error stopping node instance")
        self._broadcast_full_state()

    def _teardown_all(self) -> None:
        for instance_id, instance in list(self._instances.items()):
            try:
                instance.teardown()
            except Exception:
                logger.exception(f"Error tearing down node '{instance_id}'")
        self._instances.clear()
        self._routing_table.clear()
        self._node_states.clear()

    # ------------------------------------------------------------------
    # Routing table
    # ------------------------------------------------------------------

    def _build_routing_table(self, graph_config: GraphConfig) -> None:
        """Build a lookup from (source_instance, source_port) -> targets."""
        self._routing_table.clear()
        node_ids = {n.id for n in graph_config.nodes if n.type == "node"}
        pipeline_ids = {n.id for n in graph_config.nodes if n.type == "pipeline"}

        for edge in graph_config.edges:
            if edge.from_node not in node_ids:
                continue

            key = f"{edge.from_node}:{edge.from_port}"
            target: _EdgeTarget
            if edge.to_node in node_ids:
                target = _EdgeTarget("node", edge.to_node, edge.to_port)
            elif edge.to_node in pipeline_ids:
                target = _EdgeTarget("pipeline", edge.to_node, edge.to_port)
            else:
                continue

            self._routing_table.setdefault(key, []).append(target)

    # ------------------------------------------------------------------
    # Output routing (the core loop)
    # ------------------------------------------------------------------

    def _make_emitter(self, source_id: str):
        """Create the ``emit_output`` callback for a node instance."""

        def emit_output(port_name: str, value: Any) -> None:
            self._route_output(source_id, port_name, value)

        return emit_output

    def _route_output(self, source_id: str, port_name: str, value: Any) -> None:
        key = f"{source_id}:{port_name}"

        with self._lock:
            self._node_states.setdefault(source_id, {})[port_name] = value

        targets = self._routing_table.get(key, [])
        for tgt in targets:
            try:
                if tgt.target_type == "node":
                    node = self._instances.get(tgt.instance_id)
                    if node is not None:
                        node.update_input(tgt.port_name, value)
                elif tgt.target_type == "pipeline":
                    if self._frame_processor is not None:
                        self._frame_processor.update_parameters(
                            {"node_id": tgt.instance_id, tgt.port_name: value}
                        )
            except Exception:
                logger.exception(
                    f"Error routing {key} -> {tgt.target_type}:{tgt.instance_id}:{tgt.port_name}"
                )

        self._broadcast_state(source_id, port_name, value)

    # ------------------------------------------------------------------
    # External input API
    # ------------------------------------------------------------------

    def update_input(self, instance_id: str, name: str, value: Any) -> bool:
        """Update an input on a node instance.

        Can be called from REST, CLI, OSC, or any other source.

        Returns:
            ``True`` if the instance was found and updated.
        """
        with self._lock:
            instance = self._instances.get(instance_id)
        if instance is None:
            return False
        try:
            instance.update_input(name, value)
        except Exception:
            logger.exception(f"Error updating input '{name}' on node '{instance_id}'")
        return True

    def update_config(self, instance_id: str, config: dict[str, Any]) -> bool:
        """Update runtime configuration on a node instance.

        Returns:
            ``True`` if the instance was found and updated.
        """
        with self._lock:
            instance = self._instances.get(instance_id)
        if instance is None:
            return False
        try:
            instance.update_config(config)
            state = instance.get_state()
            for port_name, value in state.items():
                self._node_states.setdefault(instance_id, {})[port_name] = value
                self._broadcast_state(instance_id, port_name, value)
        except Exception:
            logger.exception(f"Error updating config on node '{instance_id}'")
            return False
        return True

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_instance_state(self, instance_id: str) -> dict[str, Any] | None:
        """Return the current observable state of a node instance."""
        with self._lock:
            instance = self._instances.get(instance_id)
        if instance is None:
            return None
        try:
            return instance.get_state()
        except Exception:
            logger.exception(f"Error getting state for node '{instance_id}'")
            return {}

    def get_instance_ports(self, instance_id: str) -> dict[str, list] | None:
        """Return the current ports for a dynamic-ports node instance."""
        with self._lock:
            instance = self._instances.get(instance_id)
        if instance is None:
            return None
        try:
            return instance.get_current_ports()
        except Exception:
            logger.exception(f"Error getting ports for node '{instance_id}'")
            return None

    def list_instances(self) -> list[dict[str, str]]:
        """Return metadata for all active node instances."""
        with self._lock:
            result = []
            for inst_id, inst in self._instances.items():
                cfg = inst.get_config_class()
                result.append(
                    {
                        "instance_id": inst_id,
                        "node_type_id": cfg.node_type_id,
                        "node_name": cfg.node_name,
                    }
                )
            return result

    # ------------------------------------------------------------------
    # WebSocket broadcast (observation only)
    # ------------------------------------------------------------------

    def register_ws_client(self, q: stdlib_queue.Queue) -> None:
        """Register a WebSocket client queue for state broadcasts."""
        self._ws_clients.append(q)

    def unregister_ws_client(self, q: stdlib_queue.Queue) -> None:
        """Unregister a WebSocket client queue."""
        try:
            self._ws_clients.remove(q)
        except ValueError:
            pass

    def _broadcast_state(self, instance_id: str, port_name: str, value: Any) -> None:
        if not self._ws_clients:
            return
        msg = {
            "type": "node_output",
            "instance_id": instance_id,
            "port": port_name,
            "value": _serialise_value(value),
        }
        dead: list[stdlib_queue.Queue] = []
        for q in self._ws_clients:
            try:
                q.put_nowait(msg)
            except stdlib_queue.Full:
                pass
            except Exception:
                dead.append(q)
        for q in dead:
            try:
                self._ws_clients.remove(q)
            except ValueError:
                pass

    def _broadcast_full_state(self) -> None:
        """Push a full_state snapshot to all connected WebSocket clients.

        Called after ``load_graph`` / ``unload_graph`` so the frontend
        immediately sees the fresh (possibly empty) state.
        """
        if not self._ws_clients:
            return
        states = self.get_all_states()
        msg = {"type": "full_state", "states": states}
        dead: list[stdlib_queue.Queue] = []
        for q in self._ws_clients:
            try:
                q.put_nowait(msg)
            except stdlib_queue.Full:
                pass
            except Exception:
                dead.append(q)
        for q in dead:
            try:
                self._ws_clients.remove(q)
            except ValueError:
                pass

    def get_all_states(self) -> dict[str, dict[str, Any]]:
        """Return the latest output values for all instances."""
        with self._lock:
            return {
                k: {pk: _serialise_value(pv) for pk, pv in v.items()}
                for k, v in self._node_states.items()
            }


def _serialise_value(value: Any) -> Any:
    """Best-effort JSON-safe conversion for broadcast values."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialise_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialise_value(v) for k, v in value.items()}
    return str(value)
