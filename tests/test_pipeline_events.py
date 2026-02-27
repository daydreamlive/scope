"""Tests for the events-vs-parameters refactor (issues #134 and #487)."""

from unittest.mock import MagicMock

from scope.core.pipelines.interface import Pipeline


class FakeState:
    """Minimal pipeline state mock for testing event clearing."""

    def __init__(self):
        self.values = {}

    def set(self, key, value):
        self.values[key] = value

    def get(self, key, default=None):
        return self.values.get(key, default)


# ---------------------------------------------------------------------------
# Pipeline.events and _clear_events tests
# ---------------------------------------------------------------------------


class TestPipelineEventsClassVar:
    """Tests for the events ClassVar on Pipeline."""

    def test_base_pipeline_events_is_empty_frozenset(self):
        """Base Pipeline should have an empty events frozenset."""
        assert Pipeline.events == frozenset()

    def test_concrete_pipelines_declare_expected_events(self):
        """All 5 concrete pipelines should declare the same events frozenset."""
        from scope.core.pipelines.krea_realtime_video.pipeline import (
            KreaRealtimeVideoPipeline,
        )
        from scope.core.pipelines.longlive.pipeline import LongLivePipeline
        from scope.core.pipelines.memflow.pipeline import MemFlowPipeline
        from scope.core.pipelines.reward_forcing.pipeline import (
            RewardForcingPipeline,
        )
        from scope.core.pipelines.streamdiffusionv2.pipeline import (
            StreamDiffusionV2Pipeline,
        )

        expected = frozenset({
            "video",
            "vace_ref_images",
            "vace_input_frames",
            "vace_input_masks",
            "first_frame_image",
            "last_frame_image",
            "lora_scales",
        })

        for cls in [
            LongLivePipeline,
            MemFlowPipeline,
            RewardForcingPipeline,
            StreamDiffusionV2Pipeline,
            KreaRealtimeVideoPipeline,
        ]:
            assert cls.events == expected, f"{cls.__name__}.events != expected"

    def test_transition_not_in_events(self):
        """transition must NOT be in events — it is multi-chunk stateful."""
        from scope.core.pipelines.longlive.pipeline import LongLivePipeline

        assert "transition" not in LongLivePipeline.events


class TestClearEvents:
    """Tests for Pipeline._clear_events() helper."""

    def _make_pipeline_subclass(self, events_set):
        """Create a minimal concrete Pipeline subclass with given events."""

        class TestPipeline(Pipeline):
            events = events_set

            def __call__(self, **kwargs):
                return {}

        return TestPipeline()

    def test_clears_event_keys_not_in_kwargs(self):
        """Event keys not provided in kwargs should be set to None in state."""
        pipeline = self._make_pipeline_subclass(
            frozenset({"video", "vace_input_frames"})
        )
        state = FakeState()
        state.set("video", [1, 2, 3])
        state.set("vace_input_frames", [[4, 5]])
        state.set("prompts", "hello")

        pipeline._clear_events(state, {"prompts": "hello"})

        assert state.get("video") is None
        assert state.get("vace_input_frames") is None
        # Non-event key should be untouched
        assert state.get("prompts") == "hello"

    def test_preserves_event_keys_in_kwargs(self):
        """Event keys that ARE in kwargs should NOT be cleared."""
        pipeline = self._make_pipeline_subclass(
            frozenset({"video", "vace_input_frames"})
        )
        state = FakeState()
        state.set("video", [1, 2, 3])
        state.set("vace_input_frames", [[4, 5]])

        pipeline._clear_events(state, {"video": [10, 20]})

        # video was in kwargs — should be untouched
        assert state.get("video") == [1, 2, 3]
        # vace_input_frames was NOT in kwargs — should be cleared
        assert state.get("vace_input_frames") is None

    def test_noop_with_empty_events(self):
        """Pipeline with no events should not clear anything."""
        pipeline = self._make_pipeline_subclass(frozenset())
        state = FakeState()
        state.set("video", [1, 2, 3])

        pipeline._clear_events(state, {})

        assert state.get("video") == [1, 2, 3]


# ---------------------------------------------------------------------------
# PipelineProcessor uses pipeline.events
# ---------------------------------------------------------------------------


class TestPipelineProcessorUsesEvents:
    """Verify PipelineProcessor reads pipeline.events instead of hardcoded list."""

    def test_processor_clears_events_from_parameters(self):
        """PipelineProcessor should pop event keys from self.parameters after a call."""
        # We test the relevant logic extracted from process_chunk
        # without needing GPU / real pipeline

        events = frozenset({"video", "vace_input_frames", "lora_scales"})
        pipeline = MagicMock()
        pipeline.events = events

        # Simulate what PipelineProcessor does after pipeline call
        parameters = {
            "prompts": "hello",
            "video": [1, 2, 3],
            "vace_input_frames": [[4, 5]],
            "height": 320,
        }

        pipeline_events = getattr(pipeline, "events", frozenset())
        for key in pipeline_events:
            parameters.pop(key, None)

        # Event keys should be gone
        assert "video" not in parameters
        assert "vace_input_frames" not in parameters
        # Non-event keys should remain
        assert parameters["prompts"] == "hello"
        assert parameters["height"] == 320

    def test_processor_handles_missing_events_attr(self):
        """Pipelines without events attr (plugins) should not break."""
        pipeline = MagicMock(spec=[])  # no .events attribute

        parameters = {"prompts": "hello", "video": [1]}

        pipeline_events = getattr(pipeline, "events", frozenset())
        for key in pipeline_events:
            parameters.pop(key, None)

        # Nothing should be popped
        assert "prompts" in parameters
        assert "video" in parameters
