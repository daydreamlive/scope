"""SourceManager — manages generic and per-node hardware input sources.

Extracted from FrameProcessor to separate input source lifecycle management
from frame processing logic. Also owns the per-node source queue routing
from the graph executor so FrameProcessor doesn't need to know about it.

The manager does not know about tensors, cloud forwarding, or GPU uploads.
Receiver loops emit raw numpy frames via a callback; the caller (FrameProcessor)
decides how to route them (tensor conversion, cloud forwarding, etc.).
"""

import logging
import queue
import threading
import time
from collections.abc import Callable
from fractions import Fraction
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from scope.core.inputs import InputSource

logger = logging.getLogger(__name__)

# Type alias for the frame callback:
# (source_node_id | None, numpy_frame, pts, time_base) -> None
# source_node_id is None for the generic (non-graph) input source.
FrameCallback = Callable[[str | None, "np.ndarray", int | None, Fraction | None], None]


class SourceManager:
    """Manages input sources and source-queue routing for FrameProcessor.

    Owns:
    - Graph source queues (from graph executor) for frame fan-out
    - Per-node source queue mappings for multi-source routing
    - Generic hardware input source (NDI/Spout/Syphon/video_file)
    - Per-node hardware input sources for graph source nodes
    """

    def __init__(self):
        self._running = False

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

        # Callback invoked with (source_node_id, numpy_frame) for each received frame.
        # Set by the caller before starting receiver threads.
        self._on_frame: FrameCallback | None = None

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
    # Configuration
    # ------------------------------------------------------------------

    def set_on_frame(self, callback: FrameCallback) -> None:
        """Set the callback invoked for each received hardware-source frame."""
        self._on_frame = callback

    def start(self) -> None:
        """Mark the manager as running (call before starting receiver threads)."""
        self._running = True

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

    def route_frame_to_source(self, frame_tensor, source_node_id: str) -> bool:
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

    def route_frame_to_all_sources(self, frame_tensor) -> bool:
        """Route a pre-converted tensor frame to all generic source queues."""
        if not self._graph_source_queues:
            return False

        for sq in self._graph_source_queues:
            try:
                sq.put_nowait(frame_tensor)
            except queue.Full:
                logger.debug(
                    "Graph source queue full, dropping %s frame", self._source_type
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
            self._stop_primary_source()
            logger.info("Input source disabled")

        elif enabled and (
            source_type != self._source_type or config.get("reconnect", False)
        ):
            self._stop_primary_source()
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
                    target=self._receiver_loop,
                    args=(
                        self._source,
                        source_type,
                        None,
                        lambda: self._source_enabled and self._source is not None,
                    ),
                    daemon=True,
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
            self._source_thread = None

    def _stop_primary_source(self) -> None:
        """Stop and clean up the primary input source safely.

        Join the receiver thread before closing the source. If we close
        PyAV/FFmpeg state while another thread is still inside
        ``receive_frame()``/decode, teardown can block or wedge in native
        FFmpeg cleanup/waits.

        "Primary" here means the single non-per-node source stored on
        ``self._source`` / ``self._source_thread`` (as opposed to entries in
        ``self._sources_by_node`` for multi-source graph mode).
        """
        self._source_enabled = False

        if self._source_thread and self._source_thread.is_alive():
            self._source_thread.join(timeout=3.0)
            if self._source_thread.is_alive():
                logger.warning("Generic input source thread did not stop within 3s")
        self._source_thread = None

        if self._source is not None:
            try:
                self._source.close()
            except Exception as e:
                logger.error(f"Error closing input source: {e}")
            finally:
                self._source = None
        self._source_type = ""

    def _receiver_loop(
        self,
        source: "InputSource",
        source_type: str,
        node_id: str | None,
        is_active: Callable[[], bool],
    ) -> None:
        """Background thread that receives frames from an input source.

        Receives frames as fast as the source provides them, without throttling
        based on pipeline FPS. Backpressure is handled by the downstream queues
        (put_nowait drops frames when full). This matches the behavior of the
        WebRTC camera input path and avoids a feedback loop where FPS-based
        throttling + receive latency causes a downward FPS spiral for sources
        with non-trivial receive overhead (NDI, Syphon).

        Args:
            source: The input source to receive from.
            source_type: Source type label for logging.
            node_id: Graph source node ID, or None for the generic source.
            is_active: Callable returning False when the loop should stop.
        """
        label = (
            f"multi-source ({source_type}) node {node_id}"
            if node_id
            else f"input source ({source_type})"
        )
        logger.info(f"{label}: thread started")

        frame_count = 0

        while self._running and is_active():
            try:
                rgb_frame = source.receive_frame(timeout_ms=100)
                if rgb_frame is not None:
                    if self._on_frame is not None:
                        pts: int | None = None
                        time_base: Fraction | None = None
                        if isinstance(rgb_frame, tuple) and len(rgb_frame) == 3:
                            rgb_frame, pts, time_base = rgb_frame
                            if time_base is not None and not isinstance(
                                time_base, Fraction
                            ):
                                try:
                                    time_base = Fraction(time_base)
                                except Exception:
                                    time_base = None
                        self._on_frame(node_id, rgb_frame, pts, time_base)

                    frame_count += 1
                    if frame_count % 100 == 0:
                        logger.debug(f"{label}: received {frame_count} frames")
                else:
                    time.sleep(0.001)  # Small sleep when no frame available

            except Exception as e:
                logger.error(f"{label}: error in receive loop: {e}")
                time.sleep(0.01)

        logger.info(f"{label}: thread stopped after {frame_count} frames")

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

        from scope.core.inputs import get_input_source_classes

        input_source_classes = get_input_source_classes()

        for node in graph.nodes:
            if node.type != "source":
                continue
            if node.source_mode not in (
                "spout",
                "ndi",
                "syphon",
                "video_file",
                "youtube",
            ):
                continue
            source_name = node.source_name or ""
            node_id = node.id

            # Skip nodes without registered queues (unless handled by callback)
            if node_id not in self._source_queues_by_node and self._on_frame is None:
                continue
            source_class = input_source_classes.get(node.source_mode)
            if source_class is None or not source_class.is_available():
                logger.warning(
                    f"Input source '{node.source_mode}' not available for node {node_id}"
                )
                continue

            try:
                source = source_class()
                if source.connect(source_name):
                    thread = threading.Thread(
                        target=self._receiver_loop,
                        args=(
                            source,
                            node.source_mode,
                            node_id,
                            lambda nid=node_id: nid in self._sources_by_node,
                        ),
                        daemon=True,
                    )
                    self._sources_by_node[node_id] = {
                        "source": source,
                        "thread": thread,
                        "type": node.source_mode,
                    }
                    thread.start()
                    logger.info(
                        f"Multi-source: started {node.source_mode} for node {node_id}"
                    )
                else:
                    logger.error(
                        f"Failed to connect input source {node.source_mode} "
                        f"for node {node_id}"
                    )
                    source.close()
            except Exception as e:
                logger.error(
                    f"Error creating input source '{node.source_mode}' "
                    f"for node {node_id}: {e}"
                )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Stop and clean up all input sources."""
        self._running = False

        # Generic input source
        self._stop_primary_source()

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
