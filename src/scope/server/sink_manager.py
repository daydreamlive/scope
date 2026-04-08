"""SinkManager — manages all output concerns: sinks and recording.

Extracted from FrameProcessor to separate output lifecycle management
from frame processing logic. Also owns per-node sink queue routing
and recording coordination so FrameProcessor doesn't need to know
about specific output types.
"""

import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .recording_coordinator import RecordingCoordinator

if TYPE_CHECKING:
    from scope.core.outputs import OutputSink

    from .pipeline_processor import PipelineProcessor

logger = logging.getLogger(__name__)


class SinkManager:
    """Manages all output concerns for FrameProcessor.

    Owns:
    - Per-node sink queue mappings (WebRTC + hardware)
    - Per-sink feeder processor references (for FPS)
    - Generic hardware output sinks (NDI/Spout/Syphon)
    - Per-node hardware output sinks for graph sink nodes
    - Recording coordination (per-record-node queues and managers)
    """

    def __init__(self):
        self._running = False

        # Per-node sink queues from graph executor
        self._sink_queues_by_node: dict[str, queue.Queue] = {}
        # NDI/Spout/Syphon: separate fan-out queue (see graph_executor.GraphRun)
        self._sink_hardware_queues_by_node: dict[str, queue.Queue] = {}
        # Per-sink-node feeder processors for per-sink FPS
        self._sink_processors_by_node: dict[str, PipelineProcessor] = {}

        # Generic output sinks: sink_type -> {sink, queue, thread, name}
        self._sinks: dict[str, dict] = {}

        # Per-node output sinks: node_id -> {sink, thread, type, name}
        self._sinks_by_node: dict[str, dict] = {}

        # Recording coordination (per-record-node queues and managers)
        self._recording = RecordingCoordinator()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Mark the manager as running."""
        self._running = True

    # ------------------------------------------------------------------
    # Graph queue setup
    # ------------------------------------------------------------------

    def setup_graph_queues(
        self,
        sink_queues_by_node: dict[str, queue.Queue],
        sink_hardware_queues_by_node: dict[str, queue.Queue],
        sink_processors_by_node: dict[str, "PipelineProcessor"],
        record_queues_by_node: dict[str, queue.Queue],
    ) -> None:
        """Store sink and record queue mappings from graph executor."""
        self._sink_queues_by_node = sink_queues_by_node
        self._sink_hardware_queues_by_node = sink_hardware_queues_by_node
        self._sink_processors_by_node = sink_processors_by_node
        self._recording.setup_queues(record_queues_by_node)

    # ------------------------------------------------------------------
    # Sink queue routing
    # ------------------------------------------------------------------

    def get_from_sink(self, sink_node_id: str) -> torch.Tensor | None:
        """Read a frame from a specific sink node's output queue (multi-sink)."""
        sink_q = self._sink_queues_by_node.get(sink_node_id)
        if sink_q is None:
            return None

        try:
            frame = sink_q.get_nowait()
            frame = frame.squeeze(0)
            if frame.is_cuda:
                frame = frame.cpu()
            return frame
        except queue.Empty:
            return None

    def get_sink_node_ids(self) -> list[str]:
        """Return the list of sink node IDs available for reading."""
        return list(self._sink_queues_by_node.keys())

    def get_unhandled_sink_node_ids(self) -> list[str]:
        """Return sink node IDs without their own output sink thread.

        These sinks need external draining (e.g. by the headless consumer)
        to prevent their queues from filling up and stalling the pipeline.
        """
        return [
            sid for sid in self._sink_queues_by_node if sid not in self._sinks_by_node
        ]

    def get_fps_for_sink(self, sink_node_id: str) -> float | None:
        """Get FPS for a specific sink node from its feeder processor.

        Returns None if no feeder processor is registered for this sink,
        so the caller can fall back to the default FPS.
        """
        proc = self._sink_processors_by_node.get(sink_node_id)
        if proc is not None:
            return proc.get_fps()
        return None

    # ------------------------------------------------------------------
    # Generic output sinks info
    # ------------------------------------------------------------------

    def get_info(self) -> dict:
        """Return info about generic sinks for stats reporting."""
        return {k: {"name": v["name"]} for k, v in self._sinks.items()}

    @property
    def has_generic_sinks(self) -> bool:
        return bool(self._sinks)

    # ------------------------------------------------------------------
    # Generic output sinks
    # ------------------------------------------------------------------

    def update_config(self, config: dict, dimensions: tuple[int, int]) -> None:
        """Handle the generic output_sinks config dict.

        Args:
            config: Format: {"spout": {"enabled": True, "name": "ScopeOut"}, ...}
            dimensions: (width, height) for creating sinks.
        """
        from scope.core.outputs import get_output_sink_classes

        sink_classes = get_output_sink_classes()
        for sink_type, sink_config in config.items():
            enabled = sink_config.get("enabled", False)
            name = sink_config.get("name", "")
            sink_cls = sink_classes.get(sink_type)
            if sink_cls is None:
                if enabled:
                    logger.warning(f"Output sink '{sink_type}' not available")
                continue
            self._update_sink(
                sink_type=sink_type,
                enabled=enabled,
                sink_name=name,
                dimensions=dimensions,
                sink_class=sink_cls,
            )

    def _update_sink(
        self,
        sink_type: str,
        enabled: bool,
        sink_name: str,
        dimensions: tuple[int, int],
        sink_class: "type[OutputSink]",
    ) -> None:
        """Create, update, or destroy a single generic output sink entry."""
        width, height = dimensions
        existing = self._sinks.get(sink_type)

        logger.info(
            f"Output sink config: type={sink_type}, enabled={enabled}, "
            f"name={sink_name}, size={width}x{height}"
        )

        if enabled and existing is None:
            # Create new sink
            try:
                sink = sink_class()
                if sink.create(sink_name, width, height):
                    q: queue.Queue = queue.Queue(maxsize=30)
                    t = threading.Thread(
                        target=self._generic_sink_loop,
                        args=(sink_type,),
                        daemon=True,
                    )
                    self._sinks[sink_type] = {
                        "sink": sink,
                        "queue": q,
                        "thread": t,
                        "name": sink_name,
                    }
                    t.start()
                    logger.info(f"Output sink enabled: {sink_type} '{sink_name}'")
                else:
                    logger.error(f"Failed to create output sink: {sink_type}")
            except Exception as e:
                logger.error(f"Error creating output sink '{sink_type}': {e}")

        elif not enabled and existing is not None:
            # Destroy existing sink
            self._close_sink(sink_type)
            logger.info(f"Output sink disabled: {sink_type}")

        elif enabled and existing is not None:
            # Recreate if name or dimensions changed
            old_sink = existing["sink"]
            needs_recreate = sink_name != existing["name"]
            if hasattr(old_sink, "width") and hasattr(old_sink, "height"):
                if old_sink.width != width or old_sink.height != height:
                    needs_recreate = True

            if needs_recreate:
                self._close_sink(sink_type)
                self._update_sink(
                    sink_type=sink_type,
                    enabled=True,
                    sink_name=sink_name,
                    dimensions=dimensions,
                    sink_class=sink_class,
                )

    def _close_sink(self, sink_type: str) -> None:
        """Stop and remove a single generic output sink."""
        entry = self._sinks.pop(sink_type, None)
        if entry is None:
            return

        q = entry["queue"]
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
        q.put_nowait(None)

        thread = entry.get("thread")
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
            if thread.is_alive():
                logger.warning(
                    f"Output sink thread '{sink_type}' did not stop within 2s"
                )
        try:
            entry["sink"].close()
        except Exception as e:
            logger.error(f"Error closing output sink '{sink_type}': {e}")

    def _generic_sink_loop(self, sink_type: str) -> None:
        """Background thread that sends frames for a single generic output sink."""
        logger.info(f"Output sink thread started: {sink_type}")
        frame_count = 0

        while self._running and sink_type in self._sinks:
            entry = self._sinks.get(sink_type)
            if entry is None:
                break
            try:
                try:
                    frame_np = entry["queue"].get(timeout=0.1)
                    if frame_np is None:
                        break
                except queue.Empty:
                    continue

                success = entry["sink"].send_frame(frame_np)
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info(
                        f"Output sink '{sink_type}' sent frame {frame_count}, "
                        f"shape={frame_np.shape}, success={success}"
                    )

            except Exception as e:
                logger.error(f"Error in output sink loop '{sink_type}': {e}")
                time.sleep(0.01)

        logger.info(f"Output sink thread stopped: {sink_type} ({frame_count} frames)")

    def fan_out_frame(self, frame_np: np.ndarray) -> None:
        """Send a frame to all generic output sinks."""
        for _sink_type, entry in self._sinks.items():
            try:
                entry["queue"].put_nowait(frame_np)
            except queue.Full:
                pass

    # ------------------------------------------------------------------
    # Per-node output sinks (multi-sink graph mode)
    # ------------------------------------------------------------------

    def setup_multi_sinks(self, graph: Any, dimensions: tuple[int, int]) -> None:
        """Set up per-sink-node output sinks for non-WebRTC graph sinks.

        For sink nodes with sink_mode in (spout, ndi, syphon), creates a
        separate OutputSink + sender thread for each one.
        """
        from .graph_schema import GraphConfig

        if not isinstance(graph, GraphConfig):
            return

        from scope.core.outputs import get_output_sink_classes

        sink_classes = get_output_sink_classes()

        for node in graph.nodes:
            if node.type != "sink":
                continue
            if node.sink_mode not in ("spout", "ndi", "syphon"):
                continue
            sink_name = node.sink_name or ""
            node_id = node.id

            if node_id not in self._sink_queues_by_node:
                continue

            sink_class = sink_classes.get(node.sink_mode)
            if sink_class is None:
                logger.warning(
                    f"Output sink '{node.sink_mode}' not available for node {node_id}"
                )
                continue

            try:
                width, height = dimensions
                sink = sink_class()
                if sink.create(sink_name, width, height):
                    thread = threading.Thread(
                        target=self._per_node_sink_loop,
                        args=(node_id, node.sink_mode),
                        daemon=True,
                    )
                    self._sinks_by_node[node_id] = {
                        "sink": sink,
                        "thread": thread,
                        "type": node.sink_mode,
                        "name": sink_name,
                    }
                    thread.start()
                    logger.info(
                        f"Multi-sink: started {node.sink_mode} '{sink_name}' "
                        f"for node {node_id}"
                    )
                else:
                    logger.error(
                        f"Failed to create output sink {node.sink_mode} for node {node_id}"
                    )
                    sink.close()
            except Exception as e:
                logger.error(
                    f"Error creating output sink '{node.sink_mode}' for node {node_id}: {e}"
                )

    def _per_node_sink_loop(self, node_id: str, sink_type: str) -> None:
        """Background thread that sends frames for a specific sink node."""
        entry = self._sinks_by_node.get(node_id)
        if entry is None:
            return

        sink = entry["sink"]
        sink_q = self._sink_hardware_queues_by_node.get(node_id)
        if sink_q is None:
            sink_q = self._sink_queues_by_node.get(node_id)
        if sink_q is None:
            logger.error(f"No sink queue for node {node_id}")
            return

        frame_count = 0
        logger.info(f"Multi-sink output thread started: {sink_type} node {node_id}")

        while self._running and node_id in self._sinks_by_node:
            try:
                try:
                    frame = sink_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Convert tensor to numpy for the output sink
                frame_squeezed = frame.squeeze(0)
                if frame_squeezed.is_cuda:
                    frame_squeezed = frame_squeezed.cpu()
                frame_np = frame_squeezed.numpy()

                sink.send_frame(frame_np)
                frame_count += 1

                if frame_count % 300 == 0:
                    logger.debug(
                        f"Multi-sink ({sink_type}) node {node_id}: "
                        f"{frame_count} frames sent"
                    )

            except Exception as e:
                logger.error(f"Error in multi-sink output loop node {node_id}: {e}")
                time.sleep(0.01)

        logger.info(
            f"Multi-sink output thread stopped ({sink_type}) node {node_id} "
            f"after {frame_count} frames"
        )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def get_from_record(self, record_node_id: str):
        """Read a frame from a record node's output queue."""
        return self._recording.get(record_node_id)

    def put_to_record(self, node_id: str, frame) -> None:
        """Convert a VideoFrame to tensor and put it into a record node's queue."""
        import torch

        rec_q = self._recording._record_queues.get(node_id)
        if rec_q is None:
            return
        try:
            frame_np = frame.to_ndarray(format="rgb24")
            t = torch.as_tensor(frame_np, dtype=torch.uint8).unsqueeze(0)
            try:
                rec_q.put_nowait(t)
            except queue.Full:
                try:
                    rec_q.get_nowait()
                    rec_q.put_nowait(t)
                except queue.Empty:
                    pass
        except Exception as e:
            logger.error(f"Error in put_to_record for node {node_id}: {e}")

    @property
    def recording(self) -> RecordingCoordinator:
        """Access the recording coordinator for record-node operations."""
        return self._recording

    def setup_cloud_graph(self, graph: Any) -> None:
        """Set up record queues from a graph config (cloud mode).

        In local mode, record queues are wired via setup_graph_queues()
        from the graph executor. In cloud mode there is no graph executor
        run, so the sink manager creates the queues directly from the
        graph config.
        """
        from .graph_schema import GraphConfig

        if not isinstance(graph, GraphConfig):
            return
        record_node_ids = graph.get_record_node_ids()
        if record_node_ids:
            self._recording.setup_queues(record_node_ids)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Stop and clean up all output sinks and recording."""
        self._running = False

        # Generic output sinks
        for sink_type in list(self._sinks):
            self._close_sink(sink_type)

        # Per-node output sinks
        for node_id, entry in list(self._sinks_by_node.items()):
            thread = entry.get("thread")
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
                if thread.is_alive():
                    logger.warning(
                        f"Output sink thread for node '{node_id}' "
                        f"did not stop within 2s"
                    )
            try:
                entry["sink"].close()
            except Exception as e:
                logger.error(f"Error closing output sink for node {node_id}: {e}")
        self._sinks_by_node.clear()

        # Recording
        self._recording.cleanup()
