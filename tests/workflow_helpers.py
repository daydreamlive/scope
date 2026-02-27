"""Shared test helpers for workflow tests."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from pydantic import BaseModel, Field

from scope.core.pipelines.base_schema import ui_field_config
from scope.core.workflows.resolve import ResolutionItem, WorkflowResolutionPlan
from scope.core.workflows.schema import (
    ScopeWorkflow,
    WorkflowMetadata,
    WorkflowPipeline,
    WorkflowPipelineSource,
)


def make_workflow(**overrides) -> ScopeWorkflow:
    """Build a minimal valid :class:`ScopeWorkflow` for tests."""
    defaults = {
        "metadata": WorkflowMetadata(
            name="test",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
            scope_version="0.1.0",
        ),
        "pipelines": [
            WorkflowPipeline(
                pipeline_id="test_pipe",
                pipeline_version="1.0.0",
                source=WorkflowPipelineSource(type="builtin"),
                params={"height": 480, "width": 640},
            )
        ],
    }
    defaults.update(overrides)
    return ScopeWorkflow(**defaults)


class FakeConfig(BaseModel):
    """Minimal pipeline config with load and runtime params for testing."""

    height: int = Field(
        default=480,
        json_schema_extra=ui_field_config(is_load_param=True),
    )
    width: int = Field(
        default=640,
        json_schema_extra=ui_field_config(is_load_param=True),
    )
    noise_scale: float = Field(
        default=0.7,
        json_schema_extra=ui_field_config(is_load_param=False),
    )


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
