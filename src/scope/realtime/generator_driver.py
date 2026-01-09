"""Generator driver - tick loop that owns the pipeline and applies control events.

The GeneratorDriver is the core loop that:
- Owns the pipeline and PipelineAdapter
- Applies control events at chunk boundaries in deterministic order
- Produces GenerationResult with frames and state snapshots
- Supports pause, resume, step, and snapshot/restore
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

from scope.realtime.control_bus import ControlBus, EventType
from scope.realtime.control_state import ControlState, GenerationMode
from scope.realtime.pipeline_adapter import PipelineAdapter, PipelineProtocol


class DriverState(Enum):
    """State of the GeneratorDriver."""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    STEPPING = "stepping"


@dataclass
class GenerationResult:
    """Result of generating one chunk of frames."""

    frames: Any  # Tensor or numpy array
    chunk_index: int
    control_state_snapshot: dict
    timing_ms: float


class GeneratorDriver:
    """Tick loop that owns the pipeline and applies control events.

    The driver maintains:
    - ControlState: current control surface
    - ControlBus: queue of pending events
    - PipelineAdapter: maps control to pipeline kwargs

    Events are applied at chunk boundaries in deterministic order.
    """

    def __init__(
        self,
        pipeline: Optional[PipelineProtocol] = None,
        on_chunk: Optional[Callable[[GenerationResult], None]] = None,
        on_state_change: Optional[Callable[[DriverState], None]] = None,
    ):
        """Initialize the driver.

        Args:
            pipeline: The Scope/KREA pipeline instance (can be None for testing)
            on_chunk: Callback for each generated chunk
            on_state_change: Callback for driver state changes
        """
        self.pipeline = pipeline
        self.adapter = PipelineAdapter(pipeline)
        self.control_bus = ControlBus()
        self.on_chunk = on_chunk or (lambda _: None)
        self.on_state_change = on_state_change or (lambda _: None)

        self.state = DriverState.STOPPED
        self.control_state = ControlState()
        self.chunk_index = 0

        self._run_task: Optional[asyncio.Task] = None
        self._is_prepared: bool = False  # Controls init_cache on first call / resets

    def _apply_control_events(self) -> bool:
        """Drain and apply queued ControlBus events at a chunk boundary.

        Returns:
            True if generation should continue, False if stopped
        """
        events = self.control_bus.drain_pending(
            is_paused=(self.state == DriverState.PAUSED),
            chunk_index=self.chunk_index,
        )

        for event in events:
            if event.type == EventType.STOP:
                self.stop()
                return False

            elif event.type == EventType.PAUSE:
                self.state = DriverState.PAUSED
                self.on_state_change(self.state)

            elif event.type == EventType.RESUME:
                if self.state == DriverState.PAUSED:
                    self.state = DriverState.RUNNING
                    self.on_state_change(self.state)

            elif event.type == EventType.SET_PROMPT:
                # Direct override: payload may contain prompts and/or transition
                for key, value in event.payload.items():
                    if hasattr(self.control_state, key):
                        setattr(self.control_state, key, value)

            elif event.type == EventType.SET_LORA_SCALES:
                self.control_state.lora_scales = event.payload.get("lora_scales", [])

            elif event.type == EventType.SET_DENOISE_STEPS:
                self.control_state.denoising_step_list = event.payload.get(
                    "denoising_step_list", self.control_state.denoising_step_list
                )

            elif event.type == EventType.SET_SEED:
                if "base_seed" in event.payload:
                    self.control_state.base_seed = int(event.payload["base_seed"])
                if "branch_seed_offset" in event.payload:
                    self.control_state.branch_seed_offset = int(
                        event.payload["branch_seed_offset"]
                    )

            elif event.type == EventType.RESTORE_SNAPSHOT:
                snapshot = event.payload.get("snapshot")
                if snapshot:
                    self.restore(snapshot)

        return True

    async def run(self):
        """Main generation loop."""
        self.state = DriverState.RUNNING
        self.on_state_change(self.state)

        while self.state == DriverState.RUNNING:
            await self._generate_chunk()
            await asyncio.sleep(0)  # Yield to event loop

    async def step(self) -> Optional[GenerationResult]:
        """Generate exactly one chunk (for Dev Console).

        Returns:
            GenerationResult if successful, None if stopped/paused
        """
        self.state = DriverState.STEPPING
        self.on_state_change(self.state)

        result = await self._generate_chunk()

        if self.state == DriverState.STEPPING:
            self.state = DriverState.PAUSED
            self.on_state_change(self.state)

        return result

    async def _generate_chunk(self) -> Optional[GenerationResult]:
        """Generate one chunk of frames."""
        # Apply control events at chunk boundary
        should_continue = self._apply_control_events()
        if not should_continue:
            return None
        if self.state not in (DriverState.RUNNING, DriverState.STEPPING):
            return None

        start_time = time.perf_counter()

        # Call pipeline (if available)
        output = None
        if self.pipeline is not None:
            output = self.pipeline(
                **self.adapter.kwargs_for_call(
                    self.control_state,
                    init_cache=(not self._is_prepared),
                )
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        self._is_prepared = True

        # Mirror pipeline-owned start frame into control_state for UI/snapshots
        if self.pipeline is not None and hasattr(self.pipeline, "state"):
            frame = self.pipeline.state.get("current_start_frame")
            if frame is not None:
                self.control_state.current_start_frame = frame

        self.chunk_index += 1

        result = GenerationResult(
            frames=output,
            chunk_index=self.chunk_index,
            control_state_snapshot=self._snapshot_control_state(),
            timing_ms=elapsed_ms,
        )

        self.on_chunk(result)
        return result

    def pause(self):
        """Pause generation."""
        self.state = DriverState.PAUSED
        self.on_state_change(self.state)

    def resume(self):
        """Resume generation. Guards against spawning multiple loops."""
        if self.state != DriverState.PAUSED:
            return  # Can only resume from paused
        if self._run_task and not self._run_task.done():
            return  # Already have an active loop
        self._run_task = asyncio.create_task(self.run())

    def stop(self):
        """Stop generation and cancel any running task."""
        self.state = DriverState.STOPPED
        self.on_state_change(self.state)
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()
        self._run_task = None

    def snapshot(self) -> dict:
        """Create a restorable snapshot of current state.

        Includes generator continuity buffers needed for seamless continuation.
        """
        return {
            "control_state": self._snapshot_control_state(),
            "chunk_index": self.chunk_index,
            "generator_continuity": self.adapter.capture_continuity(),
        }

    def restore(self, snapshot: dict):
        """Restore from a snapshot.

        If generator_continuity is present and valid, this produces seamless
        continuation. Otherwise, it's a hard cut (acceptable for branching).
        """
        # Restore control state
        ctrl_data = snapshot.get("control_state", {})
        for key, value in ctrl_data.items():
            if hasattr(self.control_state, key):
                if key == "mode":
                    if isinstance(value, str):
                        value = GenerationMode(value)
                setattr(self.control_state, key, value)

        # Restore chunk index
        self.chunk_index = snapshot.get("chunk_index", 0)

        # Restore generator continuity (if available)
        if "generator_continuity" in snapshot:
            self.adapter.restore_continuity(snapshot["generator_continuity"])

        # Ensure next generate does not wipe continuity by forcing init_cache
        self._is_prepared = True

    def _snapshot_control_state(self) -> dict:
        """Create a serializable snapshot of ControlState."""
        return {
            "prompts": self.control_state.prompts,
            "negative_prompt": self.control_state.negative_prompt,
            "lora_scales": self.control_state.lora_scales,
            "base_seed": self.control_state.base_seed,
            "branch_seed_offset": self.control_state.branch_seed_offset,
            "current_start_frame": self.control_state.current_start_frame,
            "denoising_step_list": self.control_state.denoising_step_list,
            "mode": self.control_state.mode.value,
            "kv_cache_attention_bias": self.control_state.kv_cache_attention_bias,
        }
