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


# ---------------------------------------------------------------------------
# SHA256 verification
# ---------------------------------------------------------------------------


class TestApplyLoRASha256:
    @patch("scope.core.workflows.apply.PipelineRegistry")
    def test_sha256_match_no_warning(self, mock_registry, tmp_path):
        """LoRA with matching sha256 is loaded without warnings."""
        mock_registry.get_config_class.return_value = FakeConfig

        lora_dir = tmp_path / "lora"
        lora_dir.mkdir()
        (lora_dir / "test.safetensors").write_bytes(b"fake-lora-data")

        # Compute actual hash of the file
        from scope.core.lora.manifest import compute_sha256

        expected_hash = compute_sha256(lora_dir / "test.safetensors")

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test_pipe",
                    source=WorkflowPipelineSource(type="builtin"),
                    loras=[
                        WorkflowLoRA(
                            filename="test.safetensors",
                            sha256=expected_hash,
                        )
                    ],
                )
            ]
        )

        pm = mock_pipeline_manager()
        result = asyncio.run(
            apply_workflow(wf, ok_plan(), pm, mock_plugin_manager(), tmp_path)
        )

        assert result.applied is True
        load_params = pm.load_pipelines.call_args[0][1]
        assert len(load_params["loras"]) == 1

    @patch("scope.core.workflows.apply.PipelineRegistry")
    def test_sha256_mismatch_warns_but_still_loads(self, mock_registry, tmp_path):
        """LoRA with mismatched sha256 logs a warning but is still loaded."""
        mock_registry.get_config_class.return_value = FakeConfig

        lora_dir = tmp_path / "lora"
        lora_dir.mkdir()
        (lora_dir / "test.safetensors").write_bytes(b"fake-lora-data")

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test_pipe",
                    source=WorkflowPipelineSource(type="builtin"),
                    loras=[
                        WorkflowLoRA(
                            filename="test.safetensors",
                            sha256="0000000000000000000000000000000000000000000000000000000000000000",
                        )
                    ],
                )
            ]
        )

        pm = mock_pipeline_manager()
        with patch("scope.core.workflows.apply.logger") as mock_logger:
            result = asyncio.run(
                apply_workflow(wf, ok_plan(), pm, mock_plugin_manager(), tmp_path)
            )
            mock_logger.warning.assert_called_once()
            assert "SHA256 mismatch" in mock_logger.warning.call_args[0][0]

        assert result.applied is True
        load_params = pm.load_pipelines.call_args[0][1]
        assert len(load_params["loras"]) == 1

    @patch("scope.core.workflows.apply.PipelineRegistry")
    def test_sha256_none_skips_verification(self, mock_registry, tmp_path):
        """LoRA with sha256=None does not call compute_sha256."""
        mock_registry.get_config_class.return_value = FakeConfig

        lora_dir = tmp_path / "lora"
        lora_dir.mkdir()
        (lora_dir / "test.safetensors").write_bytes(b"fake")

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test_pipe",
                    source=WorkflowPipelineSource(type="builtin"),
                    loras=[WorkflowLoRA(filename="test.safetensors")],
                )
            ]
        )

        pm = mock_pipeline_manager()
        with patch("scope.core.lora.manifest.compute_sha256") as mock_hash:
            result = asyncio.run(
                apply_workflow(wf, ok_plan(), pm, mock_plugin_manager(), tmp_path)
            )
            mock_hash.assert_not_called()

        assert result.applied is True


# ---------------------------------------------------------------------------
# Pipeline loading failures
# ---------------------------------------------------------------------------


class TestApplyPipelineLoadFailures:
    @patch("scope.core.workflows.apply.PipelineRegistry")
    def test_load_pipelines_raises_exception(self, mock_registry, tmp_path):
        """Exception from load_pipelines is caught and returns failure."""
        mock_registry.get_config_class.return_value = FakeConfig

        from unittest.mock import AsyncMock

        pm = mock_pipeline_manager()
        pm.load_pipelines = AsyncMock(side_effect=RuntimeError("GPU OOM"))

        wf = make_workflow()
        result = asyncio.run(
            apply_workflow(wf, ok_plan(), pm, mock_plugin_manager(), tmp_path)
        )

        assert result.applied is False
        assert "Failed to load pipelines" in result.message

    @patch("scope.core.workflows.apply.PipelineRegistry")
    def test_load_pipelines_returns_false(self, mock_registry, tmp_path):
        """load_pipelines returning False produces a failure result."""
        mock_registry.get_config_class.return_value = FakeConfig

        pm = mock_pipeline_manager(success=False)
        wf = make_workflow()
        result = asyncio.run(
            apply_workflow(wf, ok_plan(), pm, mock_plugin_manager(), tmp_path)
        )

        assert result.applied is False
        assert "Pipeline loading returned failure" in result.message


# ---------------------------------------------------------------------------
# Plugin install edge cases
# ---------------------------------------------------------------------------


class TestApplyPluginInstallEdgeCases:
    @patch("scope.core.workflows.apply.PipelineRegistry")
    def test_install_plugin_failure(self, mock_registry, tmp_path):
        """Plugin install exception is caught and returns failure."""
        mock_registry.get_config_class.return_value = FakeConfig

        from unittest.mock import AsyncMock

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="face-swap",
                    source=WorkflowPipelineSource(
                        type="pypi",
                        plugin_name="scope-deeplivecam",
                    ),
                )
            ]
        )

        plm = mock_plugin_manager()
        plm.install_plugin_async = AsyncMock(side_effect=RuntimeError("network error"))

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

        assert result.applied is False
        assert "Failed to install plugin" in result.message

    @patch("scope.core.workflows.apply.PipelineRegistry")
    def test_install_plugin_git_source(self, mock_registry, tmp_path):
        """Git source plugin install uses package_spec instead of plugin name."""
        mock_registry.get_config_class.return_value = FakeConfig

        git_url = "git+https://github.com/example/scope-deeplivecam.git"
        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="face-swap",
                    source=WorkflowPipelineSource(
                        type="git",
                        plugin_name="scope-deeplivecam",
                        package_spec=git_url,
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

        plm.install_plugin_async.assert_called_once_with(git_url)
        assert result.restart_required is True


# ---------------------------------------------------------------------------
# LoRA merge mode conflicts and cross-pipeline accumulation
# ---------------------------------------------------------------------------


class TestApplyLoRAMergeModes:
    @patch("scope.core.workflows.apply.PipelineRegistry")
    def test_conflicting_merge_modes_warns(self, mock_registry, tmp_path):
        """Multiple LoRAs with different merge modes logs a warning."""
        mock_registry.get_config_class.return_value = FakeConfig

        lora_dir = tmp_path / "lora"
        lora_dir.mkdir()
        (lora_dir / "a.safetensors").write_bytes(b"a")
        (lora_dir / "b.safetensors").write_bytes(b"b")

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test_pipe",
                    source=WorkflowPipelineSource(type="builtin"),
                    loras=[
                        WorkflowLoRA(
                            filename="a.safetensors",
                            merge_mode="permanent_merge",
                        ),
                        WorkflowLoRA(
                            filename="b.safetensors",
                            merge_mode="on_the_fly",
                        ),
                    ],
                )
            ]
        )

        pm = mock_pipeline_manager()
        with patch("scope.core.workflows.apply.logger") as mock_logger:
            result = asyncio.run(
                apply_workflow(wf, ok_plan(), pm, mock_plugin_manager(), tmp_path)
            )
            mock_logger.warning.assert_called_once()
            assert "conflicting merge modes" in mock_logger.warning.call_args[0][0]

        assert result.applied is True
        load_params = pm.load_pipelines.call_args[0][1]
        assert load_params["lora_merge_mode"] == "permanent_merge"

    @patch("scope.core.workflows.apply.PipelineRegistry")
    def test_loras_accumulate_across_pipelines(self, mock_registry, tmp_path):
        """LoRAs from multiple pipelines are merged into one list."""
        mock_registry.get_config_class.return_value = FakeConfig

        lora_dir = tmp_path / "lora"
        lora_dir.mkdir()
        (lora_dir / "a.safetensors").write_bytes(b"a")
        (lora_dir / "b.safetensors").write_bytes(b"b")

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="preprocessor",
                    source=WorkflowPipelineSource(type="builtin"),
                    loras=[WorkflowLoRA(filename="a.safetensors", weight=0.5)],
                ),
                WorkflowPipeline(
                    pipeline_id="test_pipe",
                    source=WorkflowPipelineSource(type="builtin"),
                    loras=[WorkflowLoRA(filename="b.safetensors", weight=0.8)],
                ),
            ]
        )

        pm = mock_pipeline_manager()
        result = asyncio.run(
            apply_workflow(wf, ok_plan(), pm, mock_plugin_manager(), tmp_path)
        )

        assert result.applied is True
        load_params = pm.load_pipelines.call_args[0][1]
        assert len(load_params["loras"]) == 2
        weights = [entry["weight"] for entry in load_params["loras"]]
        assert 0.5 in weights
        assert 0.8 in weights


# ---------------------------------------------------------------------------
# Config class None (all params become runtime)
# ---------------------------------------------------------------------------


class TestApplyNoConfigClass:
    @patch("scope.core.workflows.apply.PipelineRegistry")
    def test_all_params_become_runtime_when_no_config(self, mock_registry, tmp_path):
        """When get_config_class returns None, all params are runtime params."""
        mock_registry.get_config_class.return_value = None

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test_pipe",
                    source=WorkflowPipelineSource(type="builtin"),
                    params={"height": 480, "width": 640, "noise_scale": 0.5},
                )
            ]
        )

        pm = mock_pipeline_manager()
        result = asyncio.run(
            apply_workflow(wf, ok_plan(), pm, mock_plugin_manager(), tmp_path)
        )

        assert result.applied is True
        load_params = pm.load_pipelines.call_args[0][1]
        # No config class means no field can be identified as a load param
        assert "height" not in load_params
        assert "width" not in load_params
        assert "noise_scale" not in load_params
        # All go to runtime
        assert result.runtime_params == {
            "height": 480,
            "width": 640,
            "noise_scale": 0.5,
        }


# ---------------------------------------------------------------------------
# Apply message content
# ---------------------------------------------------------------------------


class TestApplyMessageContent:
    @patch("scope.core.workflows.apply.PipelineRegistry")
    def test_skip_message_includes_count(self, mock_registry, tmp_path):
        """Success message mentions skipped LoRA count when applicable."""
        mock_registry.get_config_class.return_value = FakeConfig
        (tmp_path / "lora").mkdir()

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test_pipe",
                    source=WorkflowPipelineSource(type="builtin"),
                    loras=[
                        WorkflowLoRA(filename="a.safetensors"),
                        WorkflowLoRA(filename="b.safetensors"),
                    ],
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
        assert "skipped 2 missing LoRA(s)" in result.message
