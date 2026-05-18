"""Node processor — wraps a BaseNode for execution in the pipeline graph.

Adapts the node interface (typed I/O ports) to the pipeline processor
interface (input/output queues, worker thread).
"""

import logging
import queue
import threading
from typing import Any

from .base import BaseNode

logger = logging.getLogger(__name__)

SLEEP_TIME = 0.01


class NodeProcessor:
    """Runs a BaseNode in a dedicated thread. Input queues feed the node,
    output queues fan out its results to downstream nodes.

    Source nodes (no inputs) execute once by default; nodes marked
    ``continuous=True`` in their definition re-execute on every tick so
    streaming sources and sinks stay alive.
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

        # Output ports wired straight to a sink; populated by graph_executor.
        # Values on these ports are routed to ``audio_output_queue`` for
        # FrameProcessor.get_audio_packet() to drain.
        self.audio_sink_ports: set[str] = set()
        # Parameter names this node declares — used to ignore broadcast
        # updates aimed at other nodes.
        self._declared_param_names: set[str] = {p.name for p in definition.params}

        # Consumed by FrameProcessor.get_audio_packet() on the sink feeder.
        # maxsize=1 + blocking put (see _route_audio) gives backpressure so
        # batch decoders can't outrun real-time playback.
        self.audio_output_queue: queue.Queue = queue.Queue(maxsize=1)

        self.worker_thread: threading.Thread | None = None
        self.shutdown_event = threading.Event()
        self.running = False

        # Execution state
        self._source_executed = False
        self._has_executed = False
        self._continuous = definition.continuous
        # Latch of last-seen inputs per port, so static upstreams (one-shot
        # model/vae/clip handles) survive across param-triggered re-runs.
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
        try:
            self.node.shutdown()
        except Exception:
            logger.exception("Error shutting down node %s", self.node_id)
        logger.info("NodeProcessor stopped: %s", self.node_id)

    def update_parameters(self, parameters: dict[str, Any]) -> None:
        # FrameProcessor broadcasts node-less updates to every processor;
        # only mark ourselves dirty when a value we actually declare changes.
        changed = any(
            key in self._declared_param_names and self.parameters.get(key) != value
            for key, value in parameters.items()
        )
        self.parameters.update(parameters)
        if changed:
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

        # Source nodes execute once; continuous=True nodes re-execute every
        # tick (for streaming I/O). A pending parameter change also re-wakes.
        if (
            is_source_node
            and self._source_executed
            and not self._continuous
            and not self._needs_rerun
        ):
            self.shutdown_event.wait(1.0)
            return

        # Drain fresh values into the latch cache; ports whose upstream has
        # already gone quiet (e.g. one-shot model handles) replay from cache.
        fresh: dict[str, Any] = {}
        inputs: dict[str, Any] = {}
        if all_queues:
            for port_name, q in all_queues.items():
                try:
                    fresh[port_name] = q.get_nowait()
                except queue.Empty:
                    pass
            self._last_inputs.update(fresh)
            inputs = dict(self._last_inputs)
            # First run: wait until every port has been seen at least once.
            if not self._has_executed and set(all_queues.keys()) - inputs.keys():
                self.shutdown_event.wait(SLEEP_TIME)
                return

        # Non-continuous nodes skip when nothing changed since last run.
        if (
            self._has_executed
            and not self._continuous
            and not fresh
            and not self._needs_rerun
        ):
            self.shutdown_event.wait(SLEEP_TIME)
            return

        outputs = self.node.execute(inputs, **self.parameters)

        if is_source_node:
            self._source_executed = True
        self._needs_rerun = False

        if not outputs:
            self.shutdown_event.wait(SLEEP_TIME)
            return

        self._has_executed = True
        self._route_outputs(outputs)

    def _route_outputs(self, outputs: dict[str, Any]) -> None:
        for port_name, value in outputs.items():
            if value is None:
                continue

            # Sink-bound audio also goes to audio_output_queue for WebRTC.
            if port_name in self.audio_sink_ports:
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
        # Lazy imports keep ``scope.core`` from reaching back into
        # ``scope.server`` at module load (disallowed by the project layout).
        from fractions import Fraction

        import torch

        from scope.server.media_packets import AudioPacket, MediaTimestamp

        start_sample: int | None = None
        if isinstance(value, tuple) and len(value) == 2:
            audio_tensor, audio_sr = value
        else:
            audio_tensor = getattr(value, "waveform", None)
            audio_sr = getattr(value, "sample_rate", 48000)
            # ACEStep StreamVAEDecode tags each window with start_sample so
            # AudioProcessingTrack can trim overlapping windows downstream.
            start_sample = getattr(value, "start_sample", None)
        if audio_tensor is None:
            return
        if isinstance(audio_tensor, torch.Tensor):
            if audio_tensor.is_cuda:
                audio_tensor = audio_tensor.detach().cpu()
            # VAE decoders return (1, C, T); the audio track expects (C, T).
            if audio_tensor.dim() == 3 and audio_tensor.shape[0] == 1:
                audio_tensor = audio_tensor.squeeze(0)
            if audio_tensor.dtype in (torch.bfloat16, torch.float16):
                audio_tensor = audio_tensor.float()

        timestamp = (
            MediaTimestamp(pts=int(start_sample), time_base=Fraction(1, int(audio_sr)))
            if start_sample is not None and audio_sr
            else MediaTimestamp()
        )
        packet = AudioPacket(
            audio=audio_tensor, sample_rate=int(audio_sr), timestamp=timestamp
        )
        # Blocking put with retry: stalls the worker when the audio track
        # hasn't drained the previous chunk — this is the backpressure.
        while not self.shutdown_event.is_set():
            try:
                self.audio_output_queue.put(packet, timeout=0.1)
                break
            except queue.Full:
                continue
