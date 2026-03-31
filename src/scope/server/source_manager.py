"""InputSourceManager — manages generic and per-node hardware input sources.

Extracted from FrameProcessor to separate input source lifecycle management
from frame processing logic. Also owns the per-node source queue routing
from the graph executor so FrameProcessor doesn't need to know about it.
"""

import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    import numpy as np

    from scope.core.inputs import InputSource

    from .cloud_connection import CloudConnectionManager

logger = logging.getLogger(__name__)


class InputSourceManager:
    """Manages input sources and source-queue routing for FrameProcessor.

    Owns:
    - Graph source queues (from graph executor) for frame fan-out
    - Per-node source queue mappings for multi-source routing
    - Generic hardware input source (NDI/Spout/Syphon/video_file)
    - Per-node hardware input sources for graph source nodes
    """

    def __init__(
        self,
        *,
        frame_to_tensor: Callable[["np.ndarray"], torch.Tensor],
        is_running: Callable[[], bool],
        cloud_manager: "CloudConnectionManager | None" = None,
        is_video_mode: Callable[[], bool] = lambda: False,
    ):
        self._frame_to_tensor = frame_to_tensor
        self._is_running = is_running
        self._cloud_manager = cloud_manager
        self._cloud_mode = cloud_manager is not None
        self._is_video_mode = is_video_mode

        # Graph source queues for generic source fan-out
        self._graph_source_queues: list[queue.Queue] = []
        # Per-node source queues: source_node_id -> list of queues
        self._source_queues_by_node: dict[str, list[queue.Queue]] = {}

        # Generic input source
        self._source: InputSource | None = None
        self._source_enabled = False
        self._source_type = ""
        self._source_thread: threading.Thread | None = None

        # Per-node input sources: node_id -> {source, thread, type}
        self._sources_by_node: dict[str, dict] = {}

        # Counter for frames sent to cloud from hardware sources
        self.frames_to_cloud = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._source_enabled

    @property
    def source_type(self) -> str:
        return self._source_type

    @property
    def has_per_node_sources(self) -> bool:
        return bool(self._sources_by_node)

    @property
    def has_source_queues(self) -> bool:
        """True when per-node source queues are active (graph mode)."""
        return bool(self._source_queues_by_node)

    @property
    def single_source_node_id(self) -> str | None:
        """If exactly one source node, return its ID (for put() shortcut)."""
        if len(self._source_queues_by_node) == 1:
            return next(iter(self._source_queues_by_node))
        return None

    def get_source_node_ids(self) -> list[str]:
        """Return the list of source node IDs."""
        return list(self._source_queues_by_node.keys())

    # ------------------------------------------------------------------
    # Graph queue setup
    # ------------------------------------------------------------------

    def setup_graph_queues(
        self,
        source_queues: list[queue.Queue],
        source_queues_by_node: dict[str, list[queue.Queue]],
    ) -> None:
        """Store source queue mappings from graph executor."""
        self._graph_source_queues = source_queues
        self._source_queues_by_node = source_queues_by_node

    # ------------------------------------------------------------------
    # Frame routing
    # ------------------------------------------------------------------

    def route_frame_to_source(
        self, frame_tensor: torch.Tensor, source_node_id: str
    ) -> bool:
        """Route a pre-converted tensor frame to a source node's queues."""
        queues = self._source_queues_by_node.get(source_node_id)
        if not queues:
            return False

        for sq in queues:
            try:
                sq.put_nowait(frame_tensor)
            except queue.Full:
                logger.debug(
                    "Source node %s queue full, dropping frame", source_node_id
                )

        return True

    # ------------------------------------------------------------------
    # Generic input source
    # ------------------------------------------------------------------

    def update_config(self, config: dict) -> None:
        """Update generic input source configuration."""
        enabled = config.get("enabled", False)
        source_type = config.get("source_type", "")
        source_name = config.get("source_name", "")

        logger.info(
            f"Input source config: enabled={enabled}, "
            f"type={source_type}, name={source_name}"
        )

        if enabled and not self._source_enabled:
            self._create_and_connect(source_type, source_name)

        elif not enabled and self._source_enabled:
            self._source_enabled = False
            if self._source is not None:
                self._source.close()
                self._source = None
            logger.info("Input source disabled")

        elif enabled and (
            source_type != self._source_type or config.get("reconnect", False)
        ):
            self._source_enabled = False
            if self._source is not None:
                self._source.close()
                self._source = None
            self._create_and_connect(source_type, source_name)

    def _create_and_connect(self, source_type: str, source_name: str) -> None:
        """Create an input source instance and connect to the given source."""
        from scope.core.inputs import get_input_source_classes

        input_source_classes = get_input_source_classes()
        source_class = input_source_classes.get(source_type)

        if source_class is None:
            logger.error(
                f"Unknown input source type '{source_type}'. "
                f"Available: {list(input_source_classes.keys())}"
            )
            return

        if not source_class.is_available():
            logger.error(
                f"Input source '{source_type}' is not available on this platform"
            )
            return

        try:
            self._source = source_class()
            if self._source.connect(source_name):
                self._source_enabled = True
                self._source_type = source_type
                self._source_thread = threading.Thread(
                    target=self._generic_receiver_loop, daemon=True
                )
                self._source_thread.start()
                logger.info(f"Input source enabled: {source_type} -> '{source_name}'")
            else:
                logger.error(
                    f"Failed to connect to input source: "
                    f"{source_type} -> '{source_name}'"
                )
                self._source.close()
                self._source = None
        except Exception as e:
            logger.error(f"Error creating input source '{source_type}': {e}")
            if self._source is not None:
                try:
                    self._source.close()
                except Exception:
                    pass
            self._source = None

    def _generic_receiver_loop(self) -> None:
        """Background thread that receives frames from a generic input source.

        Receives frames as fast as the source provides them, without throttling
        based on pipeline FPS. Backpressure is handled by the downstream queues
        (put_nowait drops frames when full). This matches the behavior of the
        WebRTC camera input path and avoids a feedback loop where FPS-based
        throttling + receive latency causes a downward FPS spiral for sources
        with non-trivial receive overhead (NDI, Syphon).
        """
        logger.info(f"Input source thread started ({self._source_type})")

        frame_count = 0

        while self._is_running() and self._source_enabled and self._source is not None:
            try:
                rgb_frame = self._source.receive_frame(timeout_ms=100)
                if rgb_frame is not None:
                    if self._cloud_mode:
                        if self._is_video_mode() and self._cloud_manager:
                            if self._cloud_manager.send_frame(rgb_frame):
                                self.frames_to_cloud += 1
                    elif self._graph_source_queues:
                        frame_tensor = self._frame_to_tensor(rgb_frame)

                        for sq in self._graph_source_queues:
                            try:
                                sq.put_nowait(frame_tensor)
                            except queue.Full:
                                logger.debug(
                                    f"Graph source queue full, "
                                    f"dropping {self._source_type} frame"
                                )

                    frame_count += 1
                    if frame_count % 100 == 0:
                        logger.debug(
                            f"Input source ({self._source_type}) "
                            f"received {frame_count} frames"
                        )
                else:
                    time.sleep(0.001)  # Small sleep when no frame available

            except Exception as e:
                logger.error(f"Error in input source loop: {e}")
                time.sleep(0.01)

        logger.info(
            f"Input source thread stopped ({self._source_type}) "
            f"after {frame_count} frames"
        )

    # ------------------------------------------------------------------
    # Per-node input sources (multi-source graph mode)
    # ------------------------------------------------------------------

    def setup_multi_sources(self, graph: Any) -> None:
        """Set up per-source-node input sources for non-WebRTC graph sources.

        For source nodes with source_mode in (spout, ndi, syphon, video_file),
        creates a separate InputSource + receiver thread for each one.
        """
        from .graph_schema import GraphConfig

        if not isinstance(graph, GraphConfig):
            return

        for node in graph.nodes:
            if node.type != "source":
                continue
            source_mode = getattr(node, "source_mode", None)
            if source_mode not in ("spout", "ndi", "syphon", "video_file"):
                continue
            source_name = getattr(node, "source_name", "") or ""
            node_id = node.id

            # In cloud mode we don't need local source queues — frames are
            # forwarded to the cloud via send_frame_to_track. In local mode
            # the node must have queues registered by the graph executor.
            if not self._cloud_mode and node_id not in self._source_queues_by_node:
                continue

            from scope.core.inputs import get_input_source_classes

            input_source_classes = get_input_source_classes()
            source_class = input_source_classes.get(source_mode)
            if source_class is None or not source_class.is_available():
                logger.warning(
                    f"Input source '{source_mode}' not available for node {node_id}"
                )
                continue

            try:
                source = source_class()
                if source.connect(source_name):
                    thread = threading.Thread(
                        target=self._per_node_receiver_loop,
                        args=(node_id, source_mode),
                        daemon=True,
                    )
                    self._sources_by_node[node_id] = {
                        "source": source,
                        "thread": thread,
                        "type": source_mode,
                    }
                    thread.start()
                    logger.info(
                        f"Multi-source: started {source_mode} for node {node_id}"
                    )
                else:
                    logger.error(
                        f"Failed to connect input source {source_mode} "
                        f"for node {node_id}"
                    )
                    source.close()
            except Exception as e:
                logger.error(
                    f"Error creating input source '{source_mode}' "
                    f"for node {node_id}: {e}"
                )

    def _per_node_receiver_loop(self, node_id: str, source_type: str) -> None:
        """Background thread that receives frames for a specific source node.

        Receives as fast as the source provides frames, without throttling to
        pipeline output FPS. Throttling on measured get_fps() created a feedback
        loop: slower output -> lower get_fps() -> longer sleeps -> starved inputs ->
        even slower output. Backpressure is queue full + drop, same as
        :meth:`_generic_receiver_loop`.
        """
        entry = self._sources_by_node.get(node_id)
        if entry is None:
            return

        source = entry["source"]
        frame_count = 0

        while self._is_running() and node_id in self._sources_by_node:
            try:
                rgb_frame = source.receive_frame(timeout_ms=100)
                if rgb_frame is not None:
                    if self._cloud_mode:
                        # Cloud mode: forward to the correct cloud input track
                        if self._is_video_mode() and self._cloud_manager:
                            track_idx = self._cloud_manager.get_source_track_index(
                                node_id
                            )
                            if track_idx is not None:
                                sent = self._cloud_manager.send_frame_to_track(
                                    rgb_frame, track_idx
                                )
                            else:
                                sent = self._cloud_manager.send_frame(rgb_frame)
                            if sent:
                                self.frames_to_cloud += 1
                    else:
                        queues = self._source_queues_by_node.get(node_id)
                        if queues:
                            frame_tensor = self._frame_to_tensor(rgb_frame)
                            for sq in queues:
                                try:
                                    sq.put_nowait(frame_tensor)
                                except queue.Full:
                                    pass

                    frame_count += 1
                    if frame_count % 100 == 0:
                        logger.debug(
                            f"Multi-source ({source_type}) node {node_id}: "
                            f"{frame_count} frames"
                        )
                else:
                    time.sleep(0.001)

            except Exception as e:
                logger.error(f"Error in multi-source loop node {node_id}: {e}")
                time.sleep(0.01)

        logger.info(
            f"Multi-source thread stopped ({source_type}) node {node_id} "
            f"after {frame_count} frames"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Stop and clean up all input sources."""
        # Generic input source
        self._source_enabled = False
        if self._source is not None:
            try:
                self._source.close()
            except Exception as e:
                logger.error(f"Error closing input source: {e}")
            self._source = None

        # Per-node input sources: join threads first to avoid closing the
        # source while the thread is still inside receive_frame() (causes
        # segfault with PyAV/FFmpeg).
        for node_id, entry in list(self._sources_by_node.items()):
            thread = entry.get("thread")
            if thread and thread.is_alive():
                thread.join(timeout=3.0)
                if thread.is_alive():
                    logger.warning(
                        f"Input source thread for node '{node_id}' "
                        f"did not stop within 3s"
                    )
            try:
                entry["source"].close()
            except Exception as e:
                logger.error(f"Error closing input source for node {node_id}: {e}")
        self._sources_by_node.clear()
