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

        # Output ports wired to a sink node via graph_executor. Only these
        # route through ``audio_output_queue`` → FrameProcessor.get_audio().
        # A node whose audio output feeds another node (e.g. AudioSource →
        # VAEEncodeAudio) must NOT push to audio_output_queue: with
        # maxsize=1 + blocking put, nothing would ever drain it and the
        # worker would deadlock after the second emission.
        self.audio_sink_ports: set[str] = set()
        # Names of parameters this node declares. Global param updates
        # (no node_id) are broadcast to every processor; this filter
        # keeps a graph-level tweak from spuriously marking every custom
        # node for re-execution.
        self._declared_param_names: set[str] = {p.name for p in definition.params}

        # Audio output queue consumed by FrameProcessor.get_audio() on the
        # sink. Size 1 + blocking producer (see _route_audio) gives us
        # backpressure: the audio decoder stalls until AudioProcessingTrack
        # has served enough of the current chunk to pull a new one.
        self.audio_output_queue: queue.Queue = queue.Queue(maxsize=1)

        self.worker_thread: threading.Thread | None = None
        self.shutdown_event = threading.Event()
        self.running = False

        # Execution state
        self._source_executed = False
        self._has_executed = False
        self._continuous = definition.continuous
        # Cached inputs from the last successful run, replayed when
        # parameters change or when a static upstream (one-shot model
        # handle) never re-emits.
        self._last_inputs: dict[str, Any] = {}
        self._needs_rerun = False
        # Fresh values drained from input queues while waiting for every
        # port to catch up. Draining unblocks upstream producers without
        # committing to a run yet.
        self._pending_inputs: dict[str, Any] = {}

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
        # Only mark the node dirty when the update actually touches a
        # parameter this node declares AND the value differs from the
        # current one. FrameProcessor.update_parameters broadcasts
        # global updates (no node_id) to every processor, so without
        # this guard a stream-level tweak would fire _needs_rerun on
        # every custom node and cascade through the DAG.
        changed = False
        for key, value in parameters.items():
            if key not in self._declared_param_names:
                continue
            if self.parameters.get(key) != value:
                changed = True
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
        inputs: dict[str, Any] = {}

        if is_source_node:
            # Source nodes run once, then re-run only when params change
            # (or every tick when continuous).
            if self._source_executed and not self._continuous and not self._needs_rerun:
                self.shutdown_event.wait(1.0)
                return
        elif self._continuous:
            # Continuous nodes: pick up whatever's currently in the queues
            # and run every tick. Latched values from prior ticks are
            # merged in so static upstreams (model/vae/clip handles)
            # persist across ticks.
            for port_name, q in all_queues.items():
                try:
                    inputs[port_name] = q.get_nowait()
                except queue.Empty:
                    pass
            # Wait until every port has been seen at least once.
            if not self._has_executed and (
                set(all_queues.keys()) - inputs.keys() - self._last_inputs.keys()
            ):
                # Cache what arrived so we don't lose it while waiting.
                self._last_inputs.update(inputs)
                self.shutdown_event.wait(SLEEP_TIME)
                return
            # Merge with latched: fresh values override cache.
            merged = {**self._last_inputs, **inputs}
            inputs = merged
        else:
            # Non-continuous node with inputs:
            # - First run waits until every port has received data at least
            #   once so the latch cache (_last_inputs) is populated.
            # - Subsequent runs drain fresh values into _pending_inputs (to
            #   unblock upstream producers) and fire when at least one
            #   port has fresh data, OR when params change.
            # - Static ports — upstreams that never re-emit, e.g.
            #   MODEL/VAE handles — are latched from _last_inputs.
            if not self._has_executed:
                if any(q.empty() for q in all_queues.values()):
                    self.shutdown_event.wait(SLEEP_TIME)
                    return
                inputs = {name: q.get_nowait() for name, q in all_queues.items()}
            else:
                for port_name, q in all_queues.items():
                    try:
                        self._pending_inputs[port_name] = q.get_nowait()
                    except queue.Empty:
                        pass
                fire = self._needs_rerun or bool(self._pending_inputs)
                if not fire:
                    self.shutdown_event.wait(SLEEP_TIME)
                    return
                inputs = {}
                for port_name in all_queues:
                    if port_name in self._pending_inputs:
                        inputs[port_name] = self._pending_inputs[port_name]
                    elif port_name in self._last_inputs:
                        inputs[port_name] = self._last_inputs[port_name]
                self._pending_inputs.clear()

        outputs = self.node.execute(inputs, **self.parameters)

        if is_source_node:
            self._source_executed = True
        self._needs_rerun = False

        if not outputs:
            self.shutdown_event.wait(SLEEP_TIME)
            return

        self._has_executed = True
        if inputs:
            self._last_inputs.update(inputs)
        self._route_outputs(outputs)

    def _route_outputs(self, outputs: dict[str, Any]) -> None:
        for port_name, value in outputs.items():
            if value is None:
                continue

            # Audio outputs only push to audio_output_queue for ports
            # graph_executor wired to a sink — an intermediate audio port
            # (e.g. AudioSource → VAEEncodeAudio) has no drainer, and the
            # maxsize=1 blocking put would deadlock after the second emit.
            # The fan-out below still runs so non-sink consumers receive
            # the value via the normal output_queues path; sink consumers
            # are not registered in output_queues, so the fan-out is a
            # no-op for them.
            if port_name in self.audio_sink_ports:
                self._route_audio(value)

            # Fan out to downstream node queues. Block briefly when a queue
            # is full so producers throttle to consumer pace and GPU tensors
            # don't accumulate in memory.
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
        # Lazy imports: pulling torch/MediaTimestamp at module scope would
        # also pull ``scope.server.media_packets`` into ``scope.core``,
        # which the project layout disallows.
        from fractions import Fraction

        import torch

        from scope.server.media_packets import AudioPacket, MediaTimestamp

        start_sample: int | None = None
        if isinstance(value, tuple) and len(value) == 2:
            audio_tensor, audio_sr = value
        else:
            audio_tensor = getattr(value, "waveform", None)
            audio_sr = getattr(value, "sample_rate", 48000)
            # ACEStep StreamVAEDecode tags each window with a ``start_sample``
            # offset (window t_start * sample_rate). Without this, overlapping
            # windows produced by ``follow_playhead`` are appended back-to-back
            # in AudioProcessingTrack and the overlapped region plays twice —
            # heard as "parts repeated". Forward the offset as a PTS so the
            # sink can detect and trim duplicates.
            start_sample = getattr(value, "start_sample", None)
        if audio_tensor is None:
            return
        if isinstance(audio_tensor, torch.Tensor):
            if audio_tensor.is_cuda:
                audio_tensor = audio_tensor.detach().cpu()
            # VAE decoders (e.g. ACEStep) return (1, C, T); the audio
            # track expects (C, T). Drop a leading singleton batch dim.
            if audio_tensor.dim() == 3 and audio_tensor.shape[0] == 1:
                audio_tensor = audio_tensor.squeeze(0)
            # Upcast bfloat16/float16 to float32 before numpy conversion
            # (AudioProcessingTrack calls .numpy() on the tensor).
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
        # Blocking-with-retry put. Stalls the worker thread when the audio
        # track hasn't finished serving the previous chunk — this is the
        # backpressure that rate-limits batch generators to real-time
        # playback instead of silently dropping audio.
        while not self.shutdown_event.is_set():
            try:
                self.audio_output_queue.put(packet, timeout=0.1)
                break
            except queue.Full:
                continue
