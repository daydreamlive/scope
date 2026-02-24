"""Tests for workflow dependency resolution."""

from __future__ import annotations

from unittest.mock import patch

from scope.core.workflows.resolve import is_load_param, resolve_workflow
from scope.core.workflows.schema import (
    WorkflowLoRA,
    WorkflowLoRAProvenance,
    WorkflowPipeline,
    WorkflowPipelineSource,
)

from .workflow_helpers import FakeConfig, make_workflow, mock_plugin_manager

# ---------------------------------------------------------------------------
# is_load_param tests
# ---------------------------------------------------------------------------


class TestIsLoadParam:
    def test_load_param_true(self):
        assert is_load_param(FakeConfig, "height") is True

    def test_load_param_false(self):
        assert is_load_param(FakeConfig, "noise_scale") is False

    def test_unknown_field(self):
        assert is_load_param(FakeConfig, "nonexistent") is False


# ---------------------------------------------------------------------------
# resolve_workflow tests
# ---------------------------------------------------------------------------


class TestResolveBuiltinPipeline:
    @patch("scope.core.workflows.resolve.PipelineRegistry")
    def test_all_ok(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = True
        mock_registry.get_config_class.return_value = FakeConfig

        wf = make_workflow()
        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        assert plan.can_apply is True
        pipeline_items = [i for i in plan.items if i.kind == "pipeline"]
        assert len(pipeline_items) == 1
        assert pipeline_items[0].status == "ok"

    @patch("scope.core.workflows.resolve.PipelineRegistry")
    def test_missing_builtin(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = False
        mock_registry.get_config_class.return_value = None

        wf = make_workflow()
        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        assert plan.can_apply is False
        pipeline_items = [i for i in plan.items if i.kind == "pipeline"]
        assert pipeline_items[0].status == "missing"


class TestResolvePlugin:
    @patch("scope.core.workflows.resolve.PipelineRegistry")
    def test_missing_plugin_auto_resolvable(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = False
        mock_registry.get_config_class.return_value = None

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="face-swap",
                    pipeline_version="0.1.0",
                    source=WorkflowPipelineSource(
                        type="pypi",
                        plugin_name="scope-deeplivecam",
                        plugin_version="0.1.0",
                    ),
                )
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        assert plan.can_apply is False
        plugin_items = [i for i in plan.items if i.kind == "plugin"]
        assert len(plugin_items) == 1
        assert plugin_items[0].status == "missing"
        assert plugin_items[0].can_auto_resolve is True
        assert "scope-deeplivecam" in plugin_items[0].action

    @patch("scope.core.workflows.resolve.PipelineRegistry")
    def test_plugin_installed_ok(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = True
        mock_registry.get_config_class.return_value = FakeConfig

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="face-swap",
                    pipeline_version="0.1.0",
                    source=WorkflowPipelineSource(
                        type="pypi",
                        plugin_name="scope-deeplivecam",
                        plugin_version="0.1.0",
                    ),
                )
            ]
        )
        pm = mock_plugin_manager([{"name": "scope-deeplivecam", "version": "0.2.0"}])

        plan = resolve_workflow(wf, pm, tmp_path)

        assert plan.can_apply is True
        plugin_items = [i for i in plan.items if i.kind == "plugin"]
        assert plugin_items[0].status == "ok"

    @patch("scope.core.workflows.resolve.PipelineRegistry")
    def test_plugin_version_mismatch(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = True
        mock_registry.get_config_class.return_value = FakeConfig

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="face-swap",
                    pipeline_version="0.1.0",
                    source=WorkflowPipelineSource(
                        type="pypi",
                        plugin_name="scope-deeplivecam",
                        plugin_version="2.0.0",
                    ),
                )
            ]
        )
        pm = mock_plugin_manager([{"name": "scope-deeplivecam", "version": "0.1.0"}])

        plan = resolve_workflow(wf, pm, tmp_path)

        plugin_items = [i for i in plan.items if i.kind == "plugin"]
        assert plugin_items[0].status == "version_mismatch"
        assert plugin_items[0].can_auto_resolve is True


class TestResolveLoRA:
    @patch("scope.core.workflows.resolve.PipelineRegistry")
    def test_lora_present(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = True
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
                    loras=[WorkflowLoRA(filename="test.safetensors")],
                )
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)
        lora_items = [i for i in plan.items if i.kind == "lora"]
        assert lora_items[0].status == "ok"

    @patch("scope.core.workflows.resolve.PipelineRegistry")
    def test_lora_missing_no_provenance(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = True
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

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        assert plan.can_apply is True  # missing LoRAs don't block
        lora_items = [i for i in plan.items if i.kind == "lora"]
        assert lora_items[0].status == "missing"
        assert lora_items[0].can_auto_resolve is False

    @patch("scope.core.workflows.resolve.PipelineRegistry")
    def test_lora_missing_with_provenance(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = True
        mock_registry.get_config_class.return_value = FakeConfig

        (tmp_path / "lora").mkdir()

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test_pipe",
                    pipeline_version="1.0.0",
                    source=WorkflowPipelineSource(type="builtin"),
                    loras=[
                        WorkflowLoRA(
                            filename="arcane.safetensors",
                            provenance=WorkflowLoRAProvenance(
                                source="huggingface",
                                repo_id="user/arcane-lora",
                            ),
                        )
                    ],
                )
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        lora_items = [i for i in plan.items if i.kind == "lora"]
        assert lora_items[0].status == "missing"
        assert lora_items[0].can_auto_resolve is True
        assert "HuggingFace" in lora_items[0].action


class TestSettingsValidation:
    @patch("scope.core.workflows.resolve.PipelineRegistry")
    def test_unknown_param_warns(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = True
        mock_registry.get_config_class.return_value = FakeConfig

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test_pipe",
                    pipeline_version="1.0.0",
                    source=WorkflowPipelineSource(type="builtin"),
                    params={"height": 480, "unknown_param": 42},
                )
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        assert any("unknown_param" in w for w in plan.settings_warnings)

    @patch("scope.core.workflows.resolve.PipelineRegistry")
    def test_known_params_no_warnings(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = True
        mock_registry.get_config_class.return_value = FakeConfig

        wf = make_workflow()
        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        assert plan.settings_warnings == []
