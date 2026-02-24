"""Tests for timeline schema and workflow integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from scope.core.workflows.schema import (
    ScopeWorkflow,
    WorkflowPrompt,
    WorkflowTimeline,
    WorkflowTimelineEntry,
)

from .workflow_helpers import make_workflow, mock_plugin_manager


class TestTimelineSchema:
    """Timeline model serialization."""

    def test_prompt_round_trip(self):
        p = WorkflowPrompt(text="a forest", weight=0.8)
        data = p.model_dump()
        assert data == {"text": "a forest", "weight": 0.8}
        assert WorkflowPrompt.model_validate(data) == p

    def test_entry_round_trip(self):
        entry = WorkflowTimelineEntry(
            start_time=0.0,
            end_time=13.1,
            prompts=[WorkflowPrompt(text="hello", weight=1.0)],
            transition_steps=5,
            temporal_interpolation_method="slerp",
        )
        data = entry.model_dump()
        restored = WorkflowTimelineEntry.model_validate(data)
        assert restored.start_time == 0.0
        assert restored.end_time == 13.1
        assert restored.transition_steps == 5
        assert restored.temporal_interpolation_method == "slerp"
        assert len(restored.prompts) == 1

    def test_timeline_round_trip(self):
        timeline = WorkflowTimeline(
            entries=[
                WorkflowTimelineEntry(start_time=0, end_time=10, prompts=[]),
                WorkflowTimelineEntry(start_time=10, end_time=20, prompts=[]),
            ]
        )
        data = timeline.model_dump()
        restored = WorkflowTimeline.model_validate(data)
        assert len(restored.entries) == 2

    def test_timeline_unknown_fields_ignored(self):
        data = {
            "entries": [],
            "future_field": "should be dropped",
        }
        tl = WorkflowTimeline.model_validate(data)
        assert tl.entries == []
        assert not hasattr(tl, "future_field")

    def test_entry_unknown_fields_ignored(self):
        data = {
            "start_time": 0,
            "end_time": 5,
            "prompts": [],
            "new_thing": 42,
        }
        entry = WorkflowTimelineEntry.model_validate(data)
        assert entry.start_time == 0


class TestWorkflowWithTimeline:
    """ScopeWorkflow with optional timeline."""

    def test_workflow_without_timeline(self):
        wf = make_workflow()
        assert wf.timeline is None
        data = wf.model_dump()
        assert data["timeline"] is None
        restored = ScopeWorkflow.model_validate(data)
        assert restored.timeline is None

    def test_workflow_with_timeline_round_trip(self):
        timeline = WorkflowTimeline(
            entries=[
                WorkflowTimelineEntry(
                    start_time=0,
                    end_time=13.1,
                    prompts=[WorkflowPrompt(text="a forest", weight=1.0)],
                    transition_steps=5,
                    temporal_interpolation_method="slerp",
                ),
            ]
        )
        wf = make_workflow(timeline=timeline)
        data = wf.model_dump()
        restored = ScopeWorkflow.model_validate(data)
        assert restored.timeline is not None
        assert len(restored.timeline.entries) == 1
        assert restored.timeline.entries[0].prompts[0].text == "a forest"

    def test_workflow_json_round_trip_with_timeline(self):
        timeline = WorkflowTimeline(
            entries=[
                WorkflowTimelineEntry(start_time=0, end_time=10, prompts=[]),
            ]
        )
        wf = make_workflow(timeline=timeline)
        json_str = wf.model_dump_json()
        restored = ScopeWorkflow.model_validate_json(json_str)
        assert restored.timeline is not None
        assert len(restored.timeline.entries) == 1


class TestMinScopeVersion:
    """ScopeWorkflow.min_scope_version field."""

    def test_default_is_none(self):
        wf = make_workflow()
        assert wf.min_scope_version is None

    def test_round_trip(self):
        wf = make_workflow(min_scope_version="0.5.0")
        data = wf.model_dump()
        restored = ScopeWorkflow.model_validate(data)
        assert restored.min_scope_version == "0.5.0"

    def test_resolve_warns_when_current_is_older(self):
        """min_scope_version check produces a warning on resolve."""
        from scope.core.workflows.resolve import resolve_workflow

        wf = make_workflow(min_scope_version="99.0.0")

        pm = mock_plugin_manager()

        with patch("scope.core.workflows.resolve.PipelineRegistry") as mock_reg:
            mock_reg.is_registered.return_value = True
            mock_reg.get_config_class.return_value = None

            plan = resolve_workflow(wf, pm, MagicMock())

        assert any("99.0.0" in w for w in plan.settings_warnings)

    def test_resolve_no_warning_when_version_ok(self):
        from scope.core.workflows.resolve import resolve_workflow

        wf = make_workflow(min_scope_version="0.0.1")

        pm = mock_plugin_manager()

        with patch("scope.core.workflows.resolve.PipelineRegistry") as mock_reg:
            mock_reg.is_registered.return_value = True
            mock_reg.get_config_class.return_value = None

            plan = resolve_workflow(wf, pm, MagicMock())

        # No warnings about min_scope_version (0.0.1 is very old)
        version_warnings = [
            w
            for w in plan.settings_warnings
            if "min_scope_version" in w.lower() or "Scope >=" in w
        ]
        assert len(version_warnings) == 0


class TestBuildWorkflowWithTimeline:
    """build_workflow passes timeline through."""

    def test_timeline_included_in_export(self):
        from scope.core.workflows.export import build_workflow

        timeline = WorkflowTimeline(
            entries=[
                WorkflowTimelineEntry(start_time=0, end_time=5, prompts=[]),
            ]
        )

        pm = MagicMock()
        pm.get_load_snapshot.return_value = {}
        plm = MagicMock()
        plm.list_plugins_sync.return_value = []

        with patch("scope.core.workflows.export.importlib.metadata") as mock_meta:
            mock_meta.version.return_value = "0.1.0"
            wf = build_workflow(
                name="test",
                description="",
                author="",
                pipeline_manager=pm,
                plugin_manager=plm,
                lora_dir=MagicMock(),
                timeline=timeline,
            )

        assert wf.timeline is not None
        assert len(wf.timeline.entries) == 1

    def test_no_timeline_by_default(self):
        from scope.core.workflows.export import build_workflow

        pm = MagicMock()
        pm.get_load_snapshot.return_value = {}
        plm = MagicMock()
        plm.list_plugins_sync.return_value = []

        with patch("scope.core.workflows.export.importlib.metadata") as mock_meta:
            mock_meta.version.return_value = "0.1.0"
            wf = build_workflow(
                name="test",
                description="",
                author="",
                pipeline_manager=pm,
                plugin_manager=plm,
                lora_dir=MagicMock(),
            )

        assert wf.timeline is None
