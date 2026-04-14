"""Node processor — wraps a BaseNode for execution in the pipeline graph.

Adapts the node interface (typed I/O ports) to the pipeline processor
interface (input/output queues, worker thread).
"""

import logging
import queue
import threading
from typing import Any

import torch

from .base import BaseNode

logger = logging.getLogger(__name__)

SLEEP_TIME = 0.01


class NodeProcessor:
    """Runs a BaseNode in a dedicated thread. Input queues feed the node,
    output queues fan out its results to downstream nodes.

    Source nodes (no inputs) execute once by default; nodes marked
    ``continuous=True`` in their definition re-execute on every tick, which
    is how streaming sources (audio) and sinks (audio loop) stay alive.
    """

    def __init__(
        self,
        node: BaseNode,
        node_id: str,
        initial_parameters: dict | None = None,
    ):
        self.node = node
        self.node_id = node_id
        self.pipeline_id = f"node:{node.node_type_id}"
        self.parameters = initial_parameters or {}

        # Port-based queues wired by the graph executor
        self.input_queues: dict[str, queue.Queue] = {}
        self.output_queues: dict[str, list[queue.Queue]] = {}
        self.input_queue_lock = threading.Lock()
        self.external_queue_refs: list[tuple[dict, str]] = []

        definition = node.get_definition()
        self.audio_input_ports: set[str] = {
            p.name for p in definition.inputs if p.port_type == "audio"
        }
        self.audio_output_ports: set[str] = {
            p.name for p in definition.outputs if p.port_type == "audio"
        }

        # Audio output queue consumed by FrameProcessor.get_audio() on the sink
        self.audio_output_queue: queue.Queue[tuple[torch.Tensor, int]] = queue.Queue(
            maxsize=10
        )

        self.worker_thread: threading.Thread | None = None
        self.shutdown_event = threading.Event()
        self.running = False

        # Source-node execution tracking
        self._source_executed = False
        self._continuous = definition.continuous

        # Output cache: stores last outputs so unchanged nodes skip re-execution
        self._cached_outputs: dict[str, Any] | None = None
        self._last_change_key: Any = None

        # PipelineProcessor interface compatibility
        self.output_consumers: dict[str, list] = {}
        self.is_prepared = False
        self.native_fps: float | None = None
        self.paused = False

    @property
    def output_queue(self) -> queue.Queue | None:
        qs = self.output_queues.get("video")
        return qs[0] if qs else None

    @property
    def pipeline(self):
        return self.node

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.shutdown_event.clear()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        self.shutdown_event.set()
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=5.0)
        logger.info("NodeProcessor stopped: %s", self.node_id)

    def update_parameters(self, parameters: dict[str, Any]) -> None:
        self.parameters.update(parameters)

    def set_beat_cache_reset_rate(self, rate):  # PipelineProcessor compat
        pass

    def get_fps(self) -> float:
        return self.native_fps or 30.0

    def _worker_loop(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                self._process_once()
            except Exception:
                logger.exception("Error in node processor %s", self.node_id)
                with self.input_queue_lock:
                    is_source = not self.input_queues
                if is_source:
                    # Avoid infinite retry on failing source nodes
                    self._source_executed = True
                    self._continuous = False
                self.shutdown_event.wait(SLEEP_TIME)

    def _process_once(self) -> None:
        if self.paused:
            self.shutdown_event.wait(SLEEP_TIME)
            return

        with self.input_queue_lock:
            all_queues = dict(self.input_queues)

        is_source_node = not all_queues

        # Source nodes execute once; continuous=True nodes re-execute every
        # tick (for streaming I/O like AudioSource chunking).
        if is_source_node and self._source_executed and not self._continuous:
            self.shutdown_event.wait(1.0)
            return

        # Gather inputs. Continuous nodes consume whatever's available
        # (empty inputs stay absent). Non-continuous nodes wait until every
        # input queue has data, so they execute with a complete input set.
        inputs: dict[str, Any] = {}
        if all_queues:
            if self._continuous:
                for port_name, q in all_queues.items():
                    try:
                        inputs[port_name] = q.get_nowait()
                    except queue.Empty:
                        pass
            else:
                if not all(not q.empty() for q in all_queues.values()):
                    self.shutdown_event.wait(SLEEP_TIME)
                    return
                for port_name, q in all_queues.items():
                    try:
                        inputs[port_name] = q.get_nowait()
                    except queue.Empty:
                        self.shutdown_event.wait(SLEEP_TIME)
                        return

        # Check IS_CHANGED before executing — if cache is valid, reuse it
        change_key = self.node.IS_CHANGED(**self.parameters)
        if (
            self._cached_outputs is not None
            and change_key == self._last_change_key
            and not inputs  # only skip if no new inputs arrived
            and not self._continuous
        ):
            self.shutdown_event.wait(SLEEP_TIME)
            return

        outputs = self.node.execute(inputs, **self.parameters)

        if is_source_node:
            self._source_executed = True

        if not outputs:
            self.shutdown_event.wait(SLEEP_TIME)
            return

        self._cached_outputs = outputs
        self._last_change_key = change_key
        self._route_outputs(outputs)

    def _route_outputs(self, outputs: dict[str, Any]) -> None:
        for port_name, value in outputs.items():
            if value is None:
                continue

            # Audio outputs also feed the FrameProcessor's audio path
            if port_name in self.audio_output_ports:
                self._route_audio(value)

            # Fan out to all downstream queues on this port. Block briefly
            # when queues are full so producers throttle to consumer pace
            # and GPU tensors don't pile up in memory.
            out_queues = self.output_queues.get(port_name)
            if out_queues:
                for oq in out_queues:
                    while not self.shutdown_event.is_set():
                        try:
                            oq.put(value, timeout=0.1)
                            break
                        except queue.Full:
                            continue

    def _route_audio(self, value: Any) -> None:
        """Extract audio tensor and push to audio_output_queue for WebRTC."""
        if isinstance(value, tuple) and len(value) == 2:
            audio_tensor, audio_sr = value
        else:
            audio_tensor = getattr(value, "waveform", None)
            audio_sr = getattr(value, "sample_rate", 48000)
        if audio_tensor is None:
            return
        if hasattr(audio_tensor, "is_cuda") and audio_tensor.is_cuda:
            audio_tensor = audio_tensor.detach().cpu()
        try:
            self.audio_output_queue.put_nowait((audio_tensor, audio_sr))
        except queue.Full:
            pass
