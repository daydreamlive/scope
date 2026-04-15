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
        # Output ports wired to a sink node via graph_executor. Only these
        # route through ``audio_output_queue`` → FrameProcessor.get_audio().
        # A node whose audio output feeds another node (e.g. AudioSource →
        # VAEEncodeAudio) must NOT push to audio_output_queue: with
        # maxsize=1 + blocking put, nothing would ever drain it and the
        # worker would deadlock after the second emission.
        self.audio_sink_ports: set[str] = set()
        # Names of parameters this node actually declares. Global param
        # updates (no node_id) are broadcast to every processor; without
        # this filter a graph-level tweak (quantize_mode, modulations…)
        # would spuriously flag every custom node for re-execution.
        self._declared_param_names: set[str] = {p.name for p in definition.params}

        # Audio output queue consumed by FrameProcessor.get_audio() on the
        # sink. Size 1 + blocking producer (see _route_audio) gives us
        # backpressure: audio_decode stalls until AudioProcessingTrack has
        # served enough of the current chunk to pull a new one, which
        # cascades upstream through node-to-node edge queues and
        # rate-limits batch generators to real-time playback.
        self.audio_output_queue: queue.Queue[tuple[torch.Tensor, int]] = queue.Queue(
            maxsize=1
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

        # Input ports whose upstream is transitively reachable from a
        # ``continuous=True`` node. Populated by graph_executor at build
        # time. When non-empty, ``_process_once`` waits until every
        # dynamic port has received fresh data since the last run before
        # firing, so a single upstream cycle triggers exactly one run
        # (not one run per fresh-port arrival). Static ports — typically
        # one-shot handles like MODEL/VAE/CONFIG — always latch from
        # ``_last_inputs``. Empty set falls back to the legacy "fire on
        # any fresh input" behaviour.
        self.dynamic_input_ports: set[str] = set()
        # Fresh values drained from input queues while waiting for every
        # dynamic port to catch up. Draining unblocks upstream producers
        # without committing to a run yet.
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
        logger.info("NodeProcessor stopped: %s", self.node_id)

    def update_parameters(self, parameters: dict[str, Any]) -> None:
        # Only mark the node dirty when the update actually touches a
        # parameter this node declares AND the value differs from the
        # current one. FrameProcessor.update_parameters broadcasts
        # global updates (no node_id) to every processor, so without
        # this guard a stream-level tweak like quantize_mode would fire
        # _needs_rerun on every custom node and cascade through the DAG.
        changed = False
        for key, value in parameters.items():
            if key not in self._declared_param_names:
                continue
            if self.parameters.get(key) != value:
                changed = True
                break
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
            # and run every tick.
            for port_name, q in all_queues.items():
                try:
                    inputs[port_name] = q.get_nowait()
                except queue.Empty:
                    pass
        else:
            # Non-continuous node with inputs:
            # - First run waits until every port has received data at least
            #   once so the latch cache (_last_inputs) is populated.
            # - Subsequent runs drain fresh values into ``_pending_inputs``
            #   (so upstream producers unblock immediately) and fire once
            #   every dynamic input port has caught up. Static ports —
            #   upstreams that never re-emit, e.g. MODEL/VAE handles —
            #   are latched from ``_last_inputs``. Parameter changes on
            #   this node bypass the gate via ``_needs_rerun``.
            # - When ``dynamic_input_ports`` is empty (graph_executor did
            #   not classify this node, e.g. a static-only subgraph), we
            #   fall back to the legacy "fire on any fresh input" gate.
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

                if self._needs_rerun:
                    fire = True
                elif self.dynamic_input_ports:
                    fire = self.dynamic_input_ports <= self._pending_inputs.keys()
                else:
                    fire = bool(self._pending_inputs)

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
        self._last_inputs = inputs
        self._route_outputs(outputs)

    def _route_outputs(self, outputs: dict[str, Any]) -> None:
        for port_name, value in outputs.items():
            if value is None:
                continue

            # Audio outputs also feed the FrameProcessor's audio path
            # — but only for ports that graph_executor wired to a sink,
            # not every port whose type is "audio". Otherwise an
            # intermediate audio-producing node (AudioSource → encoder)
            # would also push to its own audio_output_queue, which
            # nothing drains → worker deadlocks on the blocking put
            # after the second emission.
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
        # Blocking-with-retry put. Stalls the worker thread when the audio
        # track hasn't finished serving the previous chunk, which is the
        # backpressure mechanism that rate-limits batch generators to
        # real-time playback instead of silently dropping audio.
        while not self.shutdown_event.is_set():
            try:
                self.audio_output_queue.put((audio_tensor, audio_sr), timeout=0.1)
                break
            except queue.Full:
                continue
