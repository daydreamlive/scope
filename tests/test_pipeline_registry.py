"""Unit tests for PipelineRegistry unregister and is_registered methods."""

from typing import ClassVar

from scope.core.pipelines.interface import Pipeline
from scope.core.pipelines.schema import BasePipelineConfig


def _make_pipeline_class(registration_id: str) -> type[Pipeline]:
    """Build a minimal Pipeline subclass whose config carries the given id."""

    class _TestConfig(BasePipelineConfig):
        pipeline_id: ClassVar[str] = registration_id

    class _TestPipeline(Pipeline):
        @classmethod
        def get_config_class(cls) -> type[BasePipelineConfig]:
            return _TestConfig

        def __call__(self, **kwargs) -> dict:
            return {}

    return _TestPipeline


class TestPipelineRegistryUnregister:
    """Tests for PipelineRegistry.unregister method."""

    def test_unregister_removes_pipeline(self):
        """Should remove a registered pipeline from the registry."""
        from scope.core.pipelines.registry import PipelineRegistry

        pipeline_cls = _make_pipeline_class("test-unregister-1")

        PipelineRegistry.register("test-unregister-1", pipeline_cls)
        assert "test-unregister-1" in PipelineRegistry.list_pipelines()

        result = PipelineRegistry.unregister("test-unregister-1")

        assert result is True
        assert "test-unregister-1" not in PipelineRegistry.list_pipelines()

    def test_unregister_returns_true_when_found(self):
        """Should return True when pipeline was found and removed."""
        from scope.core.pipelines.registry import PipelineRegistry

        pipeline_cls = _make_pipeline_class("test-unregister-2")

        PipelineRegistry.register("test-unregister-2", pipeline_cls)
        result = PipelineRegistry.unregister("test-unregister-2")

        assert result is True

    def test_unregister_returns_false_when_not_found(self):
        """Should return False when pipeline was not found."""
        from scope.core.pipelines.registry import PipelineRegistry

        result = PipelineRegistry.unregister("nonexistent-pipeline-xyz")

        assert result is False


class TestPipelineRegistryIsRegistered:
    """Tests for PipelineRegistry.is_registered method."""

    def test_is_registered_returns_true(self):
        """Should return True for registered pipeline."""
        from scope.core.pipelines.registry import PipelineRegistry

        pipeline_cls = _make_pipeline_class("test-is-registered-1")

        PipelineRegistry.register("test-is-registered-1", pipeline_cls)

        assert PipelineRegistry.is_registered("test-is-registered-1") is True

        PipelineRegistry.unregister("test-is-registered-1")

    def test_is_registered_returns_false(self):
        """Should return False for unknown pipeline."""
        from scope.core.pipelines.registry import PipelineRegistry

        assert PipelineRegistry.is_registered("unknown-pipeline-xyz") is False
