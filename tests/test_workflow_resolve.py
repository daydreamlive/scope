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


# ---------------------------------------------------------------------------
# is_load_param edge cases
# ---------------------------------------------------------------------------


class TestIsLoadParamEdgeCases:
    def test_callable_json_schema_extra(self):
        """When json_schema_extra is a callable, is_load_param returns False."""
        from pydantic import BaseModel, Field

        class CallableExtraConfig(BaseModel):
            x: int = Field(
                default=1,
                json_schema_extra=lambda schema: schema.update(
                    {"ui": {"is_load_param": True}}
                ),
            )

        assert is_load_param(CallableExtraConfig, "x") is False


# ---------------------------------------------------------------------------
# Plugin resolution edge cases
# ---------------------------------------------------------------------------


class TestResolvePluginEdgeCases:
    @patch("scope.core.workflows.resolve.PipelineRegistry")
    def test_plugin_name_none(self, mock_registry, tmp_path):
        """Non-builtin pipeline with no plugin_name is marked missing."""
        mock_registry.is_registered.return_value = False
        mock_registry.get_config_class.return_value = None

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="unknown-pipe",
                    source=WorkflowPipelineSource(type="pypi"),
                )
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        assert plan.can_apply is False
        plugin_items = [i for i in plan.items if i.kind == "plugin"]
        assert len(plugin_items) == 1
        assert plugin_items[0].status == "missing"
        assert "No plugin name" in plugin_items[0].detail

    @patch("scope.core.workflows.resolve.PipelineRegistry")
    def test_missing_plugin_git_source(self, mock_registry, tmp_path):
        """Missing plugin with git source shows git install action."""
        mock_registry.is_registered.return_value = False
        mock_registry.get_config_class.return_value = None

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="face-swap",
                    source=WorkflowPipelineSource(
                        type="git",
                        plugin_name="scope-deeplivecam",
                        package_spec="git+https://github.com/example/scope-deeplivecam.git",
                    ),
                )
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        plugin_items = [i for i in plan.items if i.kind == "plugin"]
        assert plugin_items[0].status == "missing"
        assert "Install from git:" in plugin_items[0].action
        assert "github.com" in plugin_items[0].action

    @patch("scope.core.workflows.resolve.PipelineRegistry")
    def test_invalid_version_treated_as_ok(self, mock_registry, tmp_path):
        """Unparseable version strings don't block; treated as ok with detail."""
        mock_registry.is_registered.return_value = True
        mock_registry.get_config_class.return_value = FakeConfig

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="face-swap",
                    source=WorkflowPipelineSource(
                        type="pypi",
                        plugin_name="scope-deeplivecam",
                        plugin_version="1.0.0",
                    ),
                )
            ]
        )
        pm = mock_plugin_manager(
            [{"name": "scope-deeplivecam", "version": "not-a-version"}]
        )

        plan = resolve_workflow(wf, pm, tmp_path)

        plugin_items = [i for i in plan.items if i.kind == "plugin"]
        assert plugin_items[0].status == "ok"
        assert "Could not compare versions" in plugin_items[0].detail

    @patch("scope.core.workflows.resolve.PipelineRegistry")
    def test_plugin_no_version_info(self, mock_registry, tmp_path):
        """Plugin installed with no version on either side is ok."""
        mock_registry.is_registered.return_value = True
        mock_registry.get_config_class.return_value = FakeConfig

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
        pm = mock_plugin_manager([{"name": "scope-deeplivecam"}])

        plan = resolve_workflow(wf, pm, tmp_path)

        assert plan.can_apply is True
        plugin_items = [i for i in plan.items if i.kind == "plugin"]
        assert plugin_items[0].status == "ok"
        assert plugin_items[0].detail is None


# ---------------------------------------------------------------------------
# LoRA provenance edge cases
# ---------------------------------------------------------------------------


class TestResolveLoRAProvenance:
    @patch("scope.core.workflows.resolve.PipelineRegistry")
    def test_missing_lora_civitai_provenance(self, mock_registry, tmp_path):
        """CivitAI provenance generates correct action string."""
        mock_registry.is_registered.return_value = True
        mock_registry.get_config_class.return_value = FakeConfig
        (tmp_path / "lora").mkdir()

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test_pipe",
                    source=WorkflowPipelineSource(type="builtin"),
                    loras=[
                        WorkflowLoRA(
                            filename="style.safetensors",
                            provenance=WorkflowLoRAProvenance(
                                source="civitai",
                                model_id="12345",
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
        assert "CivitAI" in lora_items[0].action
        assert "12345" in lora_items[0].action

    @patch("scope.core.workflows.resolve.PipelineRegistry")
    def test_missing_lora_url_provenance_with_url(self, mock_registry, tmp_path):
        """URL provenance with a url field uses that URL in the action."""
        mock_registry.is_registered.return_value = True
        mock_registry.get_config_class.return_value = FakeConfig
        (tmp_path / "lora").mkdir()

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test_pipe",
                    source=WorkflowPipelineSource(type="builtin"),
                    loras=[
                        WorkflowLoRA(
                            filename="custom.safetensors",
                            provenance=WorkflowLoRAProvenance(
                                source="url",
                                url="https://example.com/lora.safetensors",
                            ),
                        )
                    ],
                )
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        lora_items = [i for i in plan.items if i.kind == "lora"]
        assert lora_items[0].can_auto_resolve is True
        assert "example.com" in lora_items[0].action

    @patch("scope.core.workflows.resolve.PipelineRegistry")
    def test_missing_lora_url_provenance_without_url(self, mock_registry, tmp_path):
        """URL provenance without a url field falls back to generic action."""
        mock_registry.is_registered.return_value = True
        mock_registry.get_config_class.return_value = FakeConfig
        (tmp_path / "lora").mkdir()

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test_pipe",
                    source=WorkflowPipelineSource(type="builtin"),
                    loras=[
                        WorkflowLoRA(
                            filename="custom.safetensors",
                            provenance=WorkflowLoRAProvenance(source="url"),
                        )
                    ],
                )
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        lora_items = [i for i in plan.items if i.kind == "lora"]
        assert lora_items[0].can_auto_resolve is True
        assert lora_items[0].action == "Download from source"

    @patch("scope.core.workflows.resolve.PipelineRegistry")
    def test_missing_lora_local_provenance_not_auto_resolvable(
        self, mock_registry, tmp_path
    ):
        """Local provenance is treated as no provenance (not auto-resolvable)."""
        mock_registry.is_registered.return_value = True
        mock_registry.get_config_class.return_value = FakeConfig
        (tmp_path / "lora").mkdir()

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test_pipe",
                    source=WorkflowPipelineSource(type="builtin"),
                    loras=[
                        WorkflowLoRA(
                            filename="local.safetensors",
                            provenance=WorkflowLoRAProvenance(source="local"),
                        )
                    ],
                )
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        lora_items = [i for i in plan.items if i.kind == "lora"]
        assert lora_items[0].status == "missing"
        assert lora_items[0].can_auto_resolve is False
        assert lora_items[0].action is None


# ---------------------------------------------------------------------------
# Multi-pipeline resolve
# ---------------------------------------------------------------------------


class TestResolveMultiplePipelines:
    @patch("scope.core.workflows.resolve.PipelineRegistry")
    def test_mixed_builtin_ok_and_plugin_missing(self, mock_registry, tmp_path):
        """One ok builtin + one missing plugin: can_apply is False, both items present."""
        mock_registry.is_registered.side_effect = lambda pid: pid == "test_pipe"
        mock_registry.get_config_class.return_value = FakeConfig

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test_pipe",
                    source=WorkflowPipelineSource(type="builtin"),
                ),
                WorkflowPipeline(
                    pipeline_id="face-swap",
                    source=WorkflowPipelineSource(
                        type="pypi",
                        plugin_name="scope-deeplivecam",
                    ),
                ),
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        assert plan.can_apply is False
        pipeline_items = [i for i in plan.items if i.kind == "pipeline"]
        plugin_items = [i for i in plan.items if i.kind == "plugin"]
        assert len(pipeline_items) == 1
        assert pipeline_items[0].status == "ok"
        assert len(plugin_items) == 1
        assert plugin_items[0].status == "missing"

    @patch("scope.core.workflows.resolve.PipelineRegistry")
    def test_multiple_pipelines_all_ok(self, mock_registry, tmp_path):
        """Multiple builtin pipelines all registered: can_apply is True."""
        mock_registry.is_registered.return_value = True
        mock_registry.get_config_class.return_value = FakeConfig

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="pipe_a",
                    source=WorkflowPipelineSource(type="builtin"),
                ),
                WorkflowPipeline(
                    pipeline_id="pipe_b",
                    source=WorkflowPipelineSource(type="builtin"),
                ),
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        assert plan.can_apply is True
        pipeline_items = [i for i in plan.items if i.kind == "pipeline"]
        assert len(pipeline_items) == 2
        assert all(i.status == "ok" for i in pipeline_items)
