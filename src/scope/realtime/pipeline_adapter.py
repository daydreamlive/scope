"""Pipeline adapter - maps ControlState to pipeline kwargs and handles continuity.

The Scope/KREA pipeline stores both control inputs and continuity buffers inside
`pipeline.state` (a PipelineState key-value store). This adapter provides:

1. Convert ControlState → exact kwargs for KreaRealtimeVideoPipeline.__call__()
2. Extract and restore continuity buffers from pipeline.state using known keys
3. Edge-trigger runtime changes with side effects (notably lora_scales)
"""

from typing import Any, Optional, Protocol

from scope.realtime.control_state import ControlState


class PipelineStateProtocol(Protocol):
    """Protocol for pipeline.state access."""

    def get(self, key: str) -> Any: ...
    def set(self, key: str, value: Any) -> None: ...


class PipelineProtocol(Protocol):
    """Protocol for the pipeline object."""

    state: PipelineStateProtocol

    def __call__(self, **kwargs: Any) -> Any: ...


class PipelineAdapter:
    """Adapter between ControlState and the Scope/KREA pipeline.

    Responsibilities:
    - Convert ControlState to pipeline kwargs
    - Edge-trigger lora_scales (only include when changed to avoid cache resets)
    - Capture/restore continuity buffers from pipeline.state
    """

    # Keys in pipeline.state that hold continuity buffers
    # These are produced/consumed by pipeline blocks and needed for seamless continuation
    CONTINUITY_KEYS = [
        "current_start_frame",
        "first_context_frame",
        "context_frame_buffer",
        "decoded_frame_buffer",
        "context_frame_buffer_max_size",
        "decoded_frame_buffer_max_size",
    ]

    def __init__(self, pipeline: Optional[PipelineProtocol] = None):
        """Initialize the adapter.

        Args:
            pipeline: The Scope/KREA pipeline instance. Can be None for testing.
        """
        self.pipeline = pipeline
        self._last_lora_scales: Optional[list[dict]] = None

    def kwargs_for_call(self, control: ControlState, *, init_cache: bool) -> dict:
        """Convert ControlState to pipeline kwargs.

        Args:
            control: The current ControlState
            init_cache: Whether to initialize/reset the cache

        Returns:
            Dict of kwargs for pipeline.__call__()

        Note:
            - lora_scales is edge-triggered: only included when it changes
            - negative_prompt is NOT included (not consumed by Scope/KREA)
            - init_cache is always explicit (driver decides)
        """
        kwargs = control.to_pipeline_kwargs()
        kwargs["init_cache"] = init_cache

        # Edge-trigger: only include lora_scales when it changes
        # In Scope/KREA, providing lora_scales may force init_cache=True
        # when manage_cache is enabled
        current_scales = control.lora_scales
        last_scales = self._last_lora_scales or []

        if current_scales != last_scales:
            if current_scales:
                kwargs["lora_scales"] = current_scales
            self._last_lora_scales = list(current_scales) if current_scales else []
        else:
            # Ensure lora_scales is not in kwargs when unchanged
            kwargs.pop("lora_scales", None)

        return kwargs

    def capture_continuity(self) -> dict:
        """Capture continuity buffers from pipeline.state.

        Returns:
            Dict of continuity buffers keyed by state key name.
            Only includes keys that have non-None values.
        """
        if self.pipeline is None:
            return {}

        st = self.pipeline.state
        return {k: st.get(k) for k in self.CONTINUITY_KEYS if st.get(k) is not None}

    def restore_continuity(self, continuity: dict):
        """Restore continuity buffers to pipeline.state.

        Args:
            continuity: Dict of continuity buffers from capture_continuity()
        """
        if self.pipeline is None:
            return

        st = self.pipeline.state
        for k, v in continuity.items():
            st.set(k, v)

    def reset_lora_tracking(self):
        """Reset lora_scales tracking.

        Call this after a full pipeline reset to ensure the next call
        includes lora_scales even if it matches the previous value.
        """
        self._last_lora_scales = None
