"""Tests for workflow application logic."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from scope.core.workflows.apply import apply_workflow
from scope.core.workflows.schema import (
    WorkflowLoRA,
    WorkflowPipeline,
    WorkflowPipelineSource,
)

from .workflow_helpers import (
    FakeConfig,
    blocked_plan,
    make_workflow,
    mock_pipeline_manager,
    mock_plugin_manager,
    ok_plan,
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestApplyBasic:
    @patch("scope.core.workflows.apply.PipelineRegistry")
    def test_successful_apply(self, mock_registry, tmp_path):
        mock_registry.get_config_class.return_value = FakeConfig

        wf = make_workflow()
        result = asyncio.run(
            apply_workflow(
                wf, ok_plan(), mock_pipeline_manager(), mock_plugin_manager(), tmp_path
            )
        )

        assert result.applied is True
        assert result.pipeline_ids == ["test_pipe"]

    @patch("scope.core.workflows.apply.PipelineRegistry")
    def test_classifies_params(self, mock_registry, tmp_path):
        mock_registry.get_config_class.return_value = FakeConfig

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test_pipe",
                    pipeline_version="1.0.0",
                    source=WorkflowPipelineSource(type="builtin"),
                    params={"height": 480, "width": 640, "noise_scale": 0.5},
                )
            ]
        )
        pm = mock_pipeline_manager()
        result = asyncio.run(
            apply_workflow(wf, ok_plan(), pm, mock_plugin_manager(), tmp_path)
        )

        call_args = pm.load_pipelines.call_args
        load_params = call_args[0][1]
        assert "height" in load_params
        assert "width" in load_params
        assert "noise_scale" not in load_params

        assert "noise_scale" in result.runtime_params

    def test_blocked_without_install(self, tmp_path):
        result = asyncio.run(
            apply_workflow(
                make_workflow(),
                blocked_plan(),
                mock_pipeline_manager(),
                mock_plugin_manager(),
                tmp_path,
            )
        )
        assert result.applied is False
        assert "missing" in result.message.lower() or "Cannot" in result.message


class TestApplyLoRAs:
    @patch("scope.core.workflows.apply.PipelineRegistry")
    def test_skip_missing_lora(self, mock_registry, tmp_path):
        mock_registry.get_config_class.return_value = FakeConfig
        (tmp_path / "lora").mkdir()

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test_pipe",
                    pipeline_version="1.0.0",
                    source=WorkflowPipelineSource(type="builtin"),
                    loras=[WorkflowLoRA(filename="missing.safetensors")],
                )
            ]
        )

        result = asyncio.run(
            apply_workflow(
                wf,
                ok_plan(),
                mock_pipeline_manager(),
                mock_plugin_manager(),
                tmp_path,
                skip_missing_loras=True,
            )
        )

        assert result.applied is True
        assert "missing.safetensors" in result.skipped_loras

    @patch("scope.core.workflows.apply.PipelineRegistry")
    def test_fail_on_missing_lora(self, mock_registry, tmp_path):
        mock_registry.get_config_class.return_value = FakeConfig
        (tmp_path / "lora").mkdir()

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test_pipe",
                    pipeline_version="1.0.0",
                    source=WorkflowPipelineSource(type="builtin"),
                    loras=[WorkflowLoRA(filename="missing.safetensors")],
                )
            ]
        )

        result = asyncio.run(
            apply_workflow(
                wf,
                ok_plan(),
                mock_pipeline_manager(),
                mock_plugin_manager(),
                tmp_path,
                skip_missing_loras=False,
            )
        )

        assert result.applied is False
        assert "missing.safetensors" in result.message

    @patch("scope.core.workflows.apply.PipelineRegistry")
    def test_lora_present_included(self, mock_registry, tmp_path):
        mock_registry.get_config_class.return_value = FakeConfig

        lora_dir = tmp_path / "lora"
        lora_dir.mkdir()
        (lora_dir / "test.safetensors").write_bytes(b"fake")

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test_pipe",
                    pipeline_version="1.0.0",
                    source=WorkflowPipelineSource(type="builtin"),
                    loras=[WorkflowLoRA(filename="test.safetensors", weight=0.8)],
                )
            ]
        )

        pm = mock_pipeline_manager()
        result = asyncio.run(
            apply_workflow(wf, ok_plan(), pm, mock_plugin_manager(), tmp_path)
        )

        assert result.applied is True
        call_args = pm.load_pipelines.call_args
        load_params = call_args[0][1]
        assert "loras" in load_params
        assert load_params["loras"][0]["weight"] == 0.8


class TestApplyPluginInstall:
    @patch("scope.core.workflows.apply.PipelineRegistry")
    def test_install_plugin_restart(self, mock_registry, tmp_path):
        mock_registry.get_config_class.return_value = FakeConfig

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="face-swap",
                    pipeline_version="0.1.0",
                    source=WorkflowPipelineSource(
                        type="pypi",
                        plugin_name="scope-deeplivecam",
                    ),
                )
            ]
        )

        plm = mock_plugin_manager()
        result = asyncio.run(
            apply_workflow(
                wf,
                blocked_plan(),
                mock_pipeline_manager(),
                plm,
                tmp_path,
                install_missing_plugins=True,
            )
        )

        plm.install_plugin_async.assert_called_once_with("scope-deeplivecam")
        assert result.restart_required is True
        assert result.applied is False


class TestApplyMultiPipeline:
    @patch("scope.core.workflows.apply.PipelineRegistry")
    def test_params_from_all_pipelines_merged(self, mock_registry, tmp_path):
        """Params from preprocessor and main pipeline are both included."""
        mock_registry.get_config_class.return_value = FakeConfig

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="preprocessor",
                    pipeline_version="1.0.0",
                    source=WorkflowPipelineSource(type="builtin"),
                    params={"height": 256},
                ),
                WorkflowPipeline(
                    pipeline_id="test_pipe",
                    pipeline_version="1.0.0",
                    source=WorkflowPipelineSource(type="builtin"),
                    params={"height": 480, "width": 640},
                ),
            ]
        )

        pm = mock_pipeline_manager()
        result = asyncio.run(
            apply_workflow(wf, ok_plan(), pm, mock_plugin_manager(), tmp_path)
        )

        assert result.applied is True
        call_args = pm.load_pipelines.call_args
        load_params = call_args[0][1]
        # Primary pipeline (last) wins on conflict
        assert load_params["height"] == 480
        assert load_params["width"] == 640
