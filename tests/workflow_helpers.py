"""Shared test helpers for workflow tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from scope.core.workflows.resolve import (
    ResolutionItem,
    WorkflowPipeline,
    WorkflowPipelineSource,
    WorkflowRequest,
    WorkflowResolutionPlan,
)


def make_workflow(**overrides) -> WorkflowRequest:
    """Build a minimal valid :class:`WorkflowRequest` for tests."""
    defaults = {
        "pipelines": [
            WorkflowPipeline(
                pipeline_id="test_pipe",
                source=WorkflowPipelineSource(type="builtin"),
            )
        ],
    }
    defaults.update(overrides)
    return WorkflowRequest(**defaults)


def mock_plugin_manager(plugins: list[dict] | None = None) -> MagicMock:
    pm = MagicMock()
    pm.list_plugins_sync.return_value = plugins or []
    pm.install_plugin_async = AsyncMock(return_value={"success": True})
    return pm


def mock_pipeline_manager(success: bool = True) -> MagicMock:
    pm = MagicMock()
    pm.load_pipelines = AsyncMock(return_value=success)
    return pm


def ok_plan() -> WorkflowResolutionPlan:
    return WorkflowResolutionPlan(
        can_apply=True,
        items=[
            ResolutionItem(kind="pipeline", name="test_pipe", status="ok"),
        ],
    )


def blocked_plan(plugin_name: str = "scope-deeplivecam") -> WorkflowResolutionPlan:
    return WorkflowResolutionPlan(
        can_apply=False,
        items=[
            ResolutionItem(
                kind="plugin",
                name=plugin_name,
                status="missing",
                can_auto_resolve=True,
            ),
        ],
    )
