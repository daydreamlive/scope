"""Unit tests for PipelineRegistry unregister and is_registered methods."""

from unittest.mock import MagicMock


class TestPipelineRegistryUnregister:
    """Tests for PipelineRegistry.unregister method."""

    def test_unregister_removes_pipeline(self):
        """Should remove a registered pipeline from the registry."""
        # Import fresh to avoid state from other tests
        from scope.core.pipelines.registry import PipelineRegistry

        # Create a mock pipeline class
        mock_pipeline = MagicMock()
        mock_config = MagicMock()
        mock_config.pipeline_id = "test-pipeline"
        mock_pipeline.get_config_class.return_value = mock_config

        # Register the pipeline
        PipelineRegistry.register("test-unregister-1", mock_pipeline)
        assert "test-unregister-1" in PipelineRegistry.list_pipelines()

        # Unregister it
        result = PipelineRegistry.unregister("test-unregister-1")

        # Verify it was removed
        assert result is True
        assert "test-unregister-1" not in PipelineRegistry.list_pipelines()

    def test_unregister_returns_true_when_found(self):
        """Should return True when pipeline was found and removed."""
        from scope.core.pipelines.registry import PipelineRegistry

        mock_pipeline = MagicMock()

        # Register then unregister
        PipelineRegistry.register("test-unregister-2", mock_pipeline)
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

        mock_pipeline = MagicMock()

        # Register the pipeline
        PipelineRegistry.register("test-is-registered-1", mock_pipeline)

        assert PipelineRegistry.is_registered("test-is-registered-1") is True

        # Cleanup
        PipelineRegistry.unregister("test-is-registered-1")

    def test_is_registered_returns_false(self):
        """Should return False for unknown pipeline."""
        from scope.core.pipelines.registry import PipelineRegistry

        assert PipelineRegistry.is_registered("unknown-pipeline-xyz") is False
