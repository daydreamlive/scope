"""Control bus - event queue with chunk-boundary semantics.

All control is chunk-transactional:
- Generator state is mutated only at chunk boundaries (between pipeline calls)
- Events are applied in deterministic order and recorded with chunk index
- "Immediate" events only apply immediately when paused (safe without generating)

Deterministic application order at each boundary:
1. Lifecycle (STOP, PAUSE, RESUME, STEP)
2. Snapshot/restore (RESTORE_SNAPSHOT, SNAPSHOT_REQUEST)
3. Style (SET_STYLE_MANIFEST) - rebind compiler
4. World (SET_WORLD_STATE) - then recompile if compiler active
5. Prompt/transition (SET_PROMPT) - direct override; may include transition
6. Runtime params (SET_DENOISE_STEPS, SET_SEED, SET_LORA_SCALES, ...)
"""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EventType(Enum):
    """Types of control events."""

    # Prompt and style
    SET_PROMPT = "set_prompt"
    SET_WORLD_STATE = "set_world_state"
    SET_STYLE_MANIFEST = "set_style_manifest"
    SET_LORA_SCALES = "set_lora_scales"

    # Generation parameters
    SET_DENOISE_STEPS = "set_denoise_steps"
    SET_SEED = "set_seed"

    # Lifecycle
    PAUSE = "pause"
    RESUME = "resume"
    STEP = "step"
    STOP = "stop"

    # Branching
    SNAPSHOT_REQUEST = "snapshot_request"
    FORK_REQUEST = "fork_request"
    ROLLOUT_REQUEST = "rollout_request"
    SELECT_BRANCH = "select_branch"
    RESTORE_SNAPSHOT = "restore_snapshot"


class ApplyMode(Enum):
    """When to apply an event."""

    NEXT_BOUNDARY = "next_boundary"  # Apply at start of next chunk
    IMMEDIATE_IF_PAUSED = "immediate"  # Apply now if paused, else next boundary


# Deterministic ordering for event types at chunk boundaries
# Lower number = applied first
EVENT_TYPE_ORDER: dict[EventType, int] = {
    # Lifecycle first
    EventType.STOP: 0,
    EventType.PAUSE: 1,
    EventType.RESUME: 2,
    EventType.STEP: 3,
    # Snapshot/restore second
    EventType.RESTORE_SNAPSHOT: 10,
    EventType.SNAPSHOT_REQUEST: 11,
    # Style third
    EventType.SET_STYLE_MANIFEST: 20,
    # World fourth
    EventType.SET_WORLD_STATE: 30,
    # Prompt/transition fifth
    EventType.SET_PROMPT: 40,
    # Runtime params last
    EventType.SET_DENOISE_STEPS: 50,
    EventType.SET_SEED: 51,
    EventType.SET_LORA_SCALES: 52,
    # Branching requests (processed after state updates)
    EventType.FORK_REQUEST: 60,
    EventType.ROLLOUT_REQUEST: 61,
    EventType.SELECT_BRANCH: 62,
}


@dataclass
class ControlEvent:
    """A single control event with timing and application semantics."""

    type: EventType
    payload: dict = field(default_factory=dict)
    apply_mode: ApplyMode = ApplyMode.NEXT_BOUNDARY
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(time.time_ns()))

    # For debugging/replay
    source: str = "api"  # "api", "vlm", "timeline", "dev_console"

    # Set when event is applied (for history tracking)
    applied_chunk_index: Optional[int] = None


@dataclass
class ControlBus:
    """Timestamped event queue with chunk-boundary semantics.

    Events are queued immediately but applied at chunk boundaries,
    ensuring the generator always sees consistent state.
    """

    pending: deque[ControlEvent] = field(default_factory=deque)
    history: list[ControlEvent] = field(default_factory=list)
    max_history: int = 1000

    def enqueue(
        self,
        event_type: EventType,
        payload: Optional[dict] = None,
        apply_mode: ApplyMode = ApplyMode.NEXT_BOUNDARY,
        source: str = "api",
    ) -> ControlEvent:
        """Add an event to the queue."""
        event = ControlEvent(
            type=event_type,
            payload=payload or {},
            apply_mode=apply_mode,
            source=source,
        )
        self.pending.append(event)
        return event

    def drain_pending(
        self, is_paused: bool = False, chunk_index: Optional[int] = None
    ) -> list[ControlEvent]:
        """Get all events that should be applied now, in deterministic order.

        Called at chunk boundaries (or immediately if checking for pause-mode events).

        Args:
            is_paused: Whether the driver is currently paused
            chunk_index: Current chunk index (for history tracking)

        Returns:
            Events to apply, sorted by deterministic order
        """
        to_apply = []
        remaining = deque()

        for event in self.pending:
            should_apply = event.apply_mode == ApplyMode.NEXT_BOUNDARY or (
                event.apply_mode == ApplyMode.IMMEDIATE_IF_PAUSED and is_paused
            )

            if should_apply:
                # Record when this event was applied
                event.applied_chunk_index = chunk_index
                to_apply.append(event)
                self._add_to_history(event)
            else:
                remaining.append(event)

        self.pending = remaining

        # Sort by deterministic order: type order, then timestamp, then event_id
        to_apply.sort(
            key=lambda e: (
                EVENT_TYPE_ORDER.get(e.type, 999),
                e.timestamp,
                e.event_id,
            )
        )

        return to_apply

    def _add_to_history(self, event: ControlEvent):
        """Store event in history for debugging/replay."""
        self.history.append(event)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def get_history(
        self,
        since_timestamp: float = 0,
        event_types: Optional[list[EventType]] = None,
    ) -> list[ControlEvent]:
        """Query event history for debugging or replay."""
        filtered = [e for e in self.history if e.timestamp >= since_timestamp]
        if event_types:
            filtered = [e for e in filtered if e.type in event_types]
        return filtered

    def clear_pending(self):
        """Clear all pending events (e.g., on stop)."""
        self.pending.clear()


# Convenience functions for common event patterns


def prompt_event(
    prompts: list[dict],
    transition: Optional[dict] = None,
    source: str = "api",
) -> ControlEvent:
    """Create a prompt update event."""
    payload = {"prompts": prompts}
    if transition is not None:
        payload["transition"] = transition
    return ControlEvent(
        type=EventType.SET_PROMPT,
        payload=payload,
        source=source,
    )


def world_state_event(updates: dict, source: str = "api") -> ControlEvent:
    """Create a world state update event."""
    return ControlEvent(
        type=EventType.SET_WORLD_STATE,
        payload=updates,
        source=source,
    )


def pause_event(source: str = "api") -> ControlEvent:
    """Create a pause event (applies at next chunk boundary)."""
    return ControlEvent(
        type=EventType.PAUSE,
        apply_mode=ApplyMode.NEXT_BOUNDARY,
        source=source,
    )
