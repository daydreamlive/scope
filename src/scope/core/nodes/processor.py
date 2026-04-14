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

        # Execution state
        self._source_executed = False
        self._has_executed = False
        self._continuous = definition.continuous
        # Cached inputs from the last successful run, replayed when
        # parameters change so non-continuous nodes can refresh their
        # output without requiring upstream to re-emit.
        self._last_inputs: dict[str, Any] = {}
        self._needs_rerun = False

        # PipelineProcessor interface compatibility: graph_executor populates
        # this for every processor; kept as an empty dict so that write is safe.
        self.output_consumers: dict[str, list] = {}
        self.paused = False

    @property
    def output_queue(self) -> queue.Queue | None:
        qs = self.output_queues.get("video")
        return qs[0] if qs else None

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.shutdown_event.clear()
        self.worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name=f"NodeProcessor[{self.node_id}]"
        )
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
        # Ask the worker to re-run with the updated params so non-continuous
        # nodes (which would otherwise sit idle until new inputs arrived)
        # reflect the change in their next output.
        self._needs_rerun = True

    def set_beat_cache_reset_rate(self, rate):  # PipelineProcessor compat
        pass

    def get_fps(self) -> float:
        return 30.0

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
            elif not any(q.empty() for q in all_queues.values()):
                inputs = {name: q.get_nowait() for name, q in all_queues.items()}

        # Decide whether to actually run. The worker re-executes when:
        #   - the node is continuous (every tick), OR
        #   - it's a source node that hasn't run yet, OR
        #   - new inputs arrived on every port (non-continuous), OR
        #   - update_parameters() flagged _needs_rerun — replay the last
        #     inputs with the new parameters so live tweaks take effect.
        consume_rerun = False
        if not self._continuous:
            if is_source_node:
                if self._source_executed and not self._needs_rerun:
                    self.shutdown_event.wait(1.0)
                    return
            else:
                if not inputs:
                    if self._has_executed and self._needs_rerun:
                        inputs = self._last_inputs
                        consume_rerun = True
                    else:
                        self.shutdown_event.wait(SLEEP_TIME)
                        return

        outputs = self.node.execute(inputs, **self.parameters)

        if is_source_node:
            self._source_executed = True
        if consume_rerun or self._needs_rerun:
            self._needs_rerun = False

        if not outputs:
            self.shutdown_event.wait(SLEEP_TIME)
            return

        self._has_executed = True
        self._last_inputs = inputs
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
        # VAE decoders (e.g. ACEStep) return (1, C, T); the audio track
        # expects (C, T). Drop a leading singleton batch dim so the
        # channel/interleave path in AudioProcessingTrack doesn't misread
        # the layout and produce slowed-down / garbled playback.
        if hasattr(audio_tensor, "dim") and audio_tensor.dim() == 3:
            if audio_tensor.shape[0] == 1:
                audio_tensor = audio_tensor.squeeze(0)
        try:
            self.audio_output_queue.put_nowait((audio_tensor, audio_sr))
        except queue.Full:
            pass
