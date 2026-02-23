"""Tests for the workflow schema and export logic."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from scope.core.workflows.schema import (
    WORKFLOW_FORMAT_VERSION,
    ScopeWorkflow,
    WorkflowLoRA,
    WorkflowMetadata,
    WorkflowPipeline,
    WorkflowPipelineSource,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_workflow(**overrides) -> ScopeWorkflow:
    defaults = {
        "metadata": WorkflowMetadata(
            name="test",
            description="desc",
            author="me",
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


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestSchemaRoundTrip:
    def test_round_trip(self):
        wf = _make_workflow()
        data = wf.model_dump(mode="json")
        restored = ScopeWorkflow.model_validate(data)
        assert restored.metadata.name == "test"
        assert restored.pipelines[0].pipeline_id == "test_pipe"

    def test_format_field(self):
        wf = _make_workflow()
        assert wf.format == "scope-workflow"

    def test_format_version(self):
        wf = _make_workflow()
        assert wf.format_version == WORKFLOW_FORMAT_VERSION

    def test_unknown_fields_ignored(self):
        data = _make_workflow().model_dump(mode="json")
        data["some_future_field"] = "hello"
        data["metadata"]["unknown_meta"] = 42
        data["pipelines"][0]["new_pipeline_field"] = True
        wf = ScopeWorkflow.model_validate(data)
        assert wf.metadata.name == "test"
        assert not hasattr(wf, "some_future_field")


class TestWorkflowLoRA:
    def test_defaults(self):
        lora = WorkflowLoRA(filename="my_lora.safetensors")
        assert lora.weight == 1.0
        assert lora.merge_mode == "permanent_merge"
        assert lora.provenance is None
        assert lora.expected_sha256 is None
        assert lora.id is None


# ---------------------------------------------------------------------------
# Export / build_workflow tests
# ---------------------------------------------------------------------------


def _mock_pipeline_manager(
    snapshot: dict | None = None,
):
    """Create a mock PipelineManager with a get_load_snapshot() return value."""
    pm = MagicMock()
    pm.get_load_snapshot.return_value = snapshot or {
        "longlive": {"height": 480, "width": 640}
    }
    return pm


def _mock_plugin_manager(plugin_for: dict | None = None, plugin_list=None):
    pm = MagicMock()
    pm.get_plugin_for_pipeline = MagicMock(
        side_effect=lambda pid: (plugin_for or {}).get(pid)
    )
    pm.list_plugins_sync = MagicMock(return_value=plugin_list or [])
    return pm


class _FakeConfigClass:
    pipeline_version = "2.0.0"


class TestBuildWorkflow:
    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_builtin_pipeline(self, mock_manifest, mock_config, mock_ver):
        from scope.core.lora.manifest import LoRAManifest
        from scope.core.workflows.export import build_workflow

        mock_manifest.return_value = LoRAManifest()
        pm = _mock_pipeline_manager()
        plm = _mock_plugin_manager()

        wf = build_workflow(
            name="my workflow",
            description="testing",
            author="tester",
            pipeline_manager=pm,
            plugin_manager=plm,
            lora_dir=Path("/models/lora"),
        )

        assert wf.format == "scope-workflow"
        assert wf.metadata.scope_version == "0.5.0"
        assert len(wf.pipelines) == 1
        assert wf.pipelines[0].source.type == "builtin"
        assert wf.pipelines[0].pipeline_version == "2.0.0"
        assert wf.pipelines[0].params["height"] == 480

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_plugin_pipeline(self, mock_manifest, mock_config, mock_ver):
        from scope.core.lora.manifest import LoRAManifest
        from scope.core.workflows.export import build_workflow

        mock_manifest.return_value = LoRAManifest()
        pm = _mock_pipeline_manager(
            snapshot={"ext_pipe": {"height": 720}},
        )
        plm = _mock_plugin_manager(
            plugin_for={"ext_pipe": "scope-plugin-cool"},
            plugin_list=[
                {
                    "name": "scope-plugin-cool",
                    "version": "0.3.1",
                    "source": "pypi",
                    "package_spec": "scope-plugin-cool>=0.3",
                }
            ],
        )

        wf = build_workflow(
            name="ext",
            description="",
            author="",
            pipeline_manager=pm,
            plugin_manager=plm,
            lora_dir=Path("/models/lora"),
        )

        src = wf.pipelines[0].source
        assert src.type == "pypi"
        assert src.plugin_name == "scope-plugin-cool"
        assert src.plugin_version == "0.3.1"
        assert src.package_spec == "scope-plugin-cool>=0.3"

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_lora_relativization(self, mock_manifest, mock_config, mock_ver):
        from scope.core.lora.manifest import (
            LoRAManifest,
            LoRAManifestEntry,
            LoRAProvenance,
        )
        from scope.core.workflows.export import build_workflow

        manifest = LoRAManifest(
            entries={
                "my_lora.safetensors": LoRAManifestEntry(
                    filename="my_lora.safetensors",
                    provenance=LoRAProvenance(
                        source="huggingface",
                        repo_id="user/lora-repo",
                        hf_filename="my_lora.safetensors",
                    ),
                    sha256="abc123",
                    size_bytes=1024,
                    added_at=datetime(2025, 1, 1, tzinfo=UTC),
                )
            }
        )
        mock_manifest.return_value = manifest

        lora_dir = Path("/models/lora")
        pm = _mock_pipeline_manager(
            snapshot={
                "longlive": {
                    "height": 480,
                    "loras": [
                        {"path": "/models/lora/my_lora.safetensors", "weight": 0.8}
                    ],
                    "lora_merge_mode": "on_the_fly",
                }
            }
        )
        plm = _mock_plugin_manager()

        wf = build_workflow(
            name="lora test",
            description="",
            author="",
            pipeline_manager=pm,
            plugin_manager=plm,
            lora_dir=lora_dir,
        )

        lora = wf.pipelines[0].loras[0]
        assert lora.filename == "my_lora.safetensors"
        assert lora.weight == 0.8
        assert lora.merge_mode == "on_the_fly"
        assert lora.provenance is not None
        assert lora.provenance.source == "huggingface"
        assert lora.provenance.repo_id == "user/lora-repo"
        assert lora.expected_sha256 == "abc123"

        # loras and lora_merge_mode removed from params
        assert "loras" not in wf.pipelines[0].params
        assert "lora_merge_mode" not in wf.pipelines[0].params

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_frontend_params_merged(self, mock_manifest, mock_config, mock_ver):
        from scope.core.lora.manifest import LoRAManifest
        from scope.core.workflows.export import build_workflow

        mock_manifest.return_value = LoRAManifest()
        pm = _mock_pipeline_manager(
            snapshot={"longlive": {"height": 480, "width": 640}}
        )
        plm = _mock_plugin_manager()

        wf = build_workflow(
            name="fe",
            description="",
            author="",
            pipeline_manager=pm,
            plugin_manager=plm,
            lora_dir=Path("/models/lora"),
            frontend_params={"longlive": {"guidance_scale": 7.5, "num_steps": 20}},
        )

        params = wf.pipelines[0].params
        assert params["guidance_scale"] == 7.5
        assert params["num_steps"] == 20
        assert params["height"] == 480

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_frontend_params_per_pipeline(self, mock_manifest, mock_config, mock_ver):
        from scope.core.lora.manifest import LoRAManifest
        from scope.core.workflows.export import build_workflow

        mock_manifest.return_value = LoRAManifest()
        pm = _mock_pipeline_manager(
            snapshot={
                "pipe_a": {"height": 480},
                "pipe_b": {"height": 720},
            }
        )
        plm = _mock_plugin_manager()

        wf = build_workflow(
            name="multi",
            description="",
            author="",
            pipeline_manager=pm,
            plugin_manager=plm,
            lora_dir=Path("/models/lora"),
            frontend_params={
                "pipe_a": {"guidance_scale": 7.5},
                "pipe_b": {"guidance_scale": 3.0, "steps": 10},
            },
        )

        assert wf.pipelines[0].params["guidance_scale"] == 7.5
        assert "steps" not in wf.pipelines[0].params
        assert wf.pipelines[1].params["guidance_scale"] == 3.0
        assert wf.pipelines[1].params["steps"] == 10


# ---------------------------------------------------------------------------
# Forward-compatibility tests
# ---------------------------------------------------------------------------


def _full_workflow_dict() -> dict:
    """A complete workflow dict used as a baseline for mutation tests."""
    return {
        "format": "scope-workflow",
        "format_version": "1.0",
        "metadata": {
            "name": "compat test",
            "description": "",
            "author": "",
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
                        "expected_sha256": "deadbeef",
                    }
                ],
                "params": {"height": 480},
            }
        ],
    }


class TestForwardCompatibility:
    """Verify that unknown fields at every nesting level are silently dropped."""

    def test_unknown_top_level_field(self):
        data = _full_workflow_dict()
        data["new_top_level"] = {"nested": True}
        wf = ScopeWorkflow.model_validate(data)
        assert wf.metadata.name == "compat test"

    def test_unknown_metadata_field(self):
        data = _full_workflow_dict()
        data["metadata"]["tags"] = ["art", "video"]
        wf = ScopeWorkflow.model_validate(data)
        assert wf.metadata.name == "compat test"

    def test_unknown_pipeline_field(self):
        data = _full_workflow_dict()
        data["pipelines"][0]["graph"] = {"nodes": []}
        wf = ScopeWorkflow.model_validate(data)
        assert wf.pipelines[0].pipeline_id == "longlive"

    def test_unknown_source_field(self):
        data = _full_workflow_dict()
        data["pipelines"][0]["source"]["plugin_hash"] = "abc"
        wf = ScopeWorkflow.model_validate(data)
        assert wf.pipelines[0].source.type == "builtin"

    def test_unknown_lora_field(self):
        data = _full_workflow_dict()
        data["pipelines"][0]["loras"][0]["trigger_words"] = ["style"]
        wf = ScopeWorkflow.model_validate(data)
        assert wf.pipelines[0].loras[0].filename == "my.safetensors"

    def test_unknown_provenance_field(self):
        """This is the key test — LoRAProvenance from manifest.py lacks
        extra='ignore', so we use WorkflowLoRAProvenance to add it."""
        data = _full_workflow_dict()
        data["pipelines"][0]["loras"][0]["provenance"]["download_count"] = 9999
        wf = ScopeWorkflow.model_validate(data)
        assert wf.pipelines[0].loras[0].provenance.repo_id == "user/repo"


class TestMinimalDocument:
    """The smallest valid workflow — only required fields."""

    def test_minimal(self):
        data = {
            "format": "scope-workflow",
            "format_version": "1.0",
            "metadata": {
                "name": "min",
                "created_at": "2025-06-01T00:00:00Z",
                "scope_version": "0.1.0",
            },
            "pipelines": [
                {
                    "pipeline_id": "p",
                    "pipeline_version": "1.0.0",
                    "source": {"type": "builtin"},
                }
            ],
        }
        wf = ScopeWorkflow.model_validate(data)
        assert wf.pipelines[0].loras == []
        assert wf.pipelines[0].params == {}
        assert wf.metadata.description == ""
        assert wf.metadata.author == ""


class TestSerializationStability:
    """Exported JSON must contain exactly the expected top-level keys."""

    def test_top_level_keys(self):
        wf = _make_workflow()
        data = wf.model_dump(mode="json")
        assert set(data.keys()) == {
            "format",
            "format_version",
            "metadata",
            "pipelines",
        }

    def test_format_survives_round_trip(self):
        wf = _make_workflow()
        data = wf.model_dump(mode="json")
        assert data["format"] == "scope-workflow"
        assert data["format_version"] == WORKFLOW_FORMAT_VERSION
        restored = ScopeWorkflow.model_validate(data)
        assert restored.format == "scope-workflow"
        assert restored.format_version == WORKFLOW_FORMAT_VERSION

    def test_json_file_round_trip(self, tmp_path):
        """Write to disk as JSON, read back, compare."""
        wf = _make_workflow()
        path = tmp_path / "test.scope-workflow.json"
        path.write_text(wf.model_dump_json(indent=2), encoding="utf-8")
        raw = path.read_text(encoding="utf-8")
        restored = ScopeWorkflow.model_validate_json(raw)
        assert restored == wf

    def test_pipeline_source_keys_builtin(self):
        src = WorkflowPipelineSource(type="builtin")
        data = src.model_dump(mode="json")
        assert data == {
            "type": "builtin",
            "plugin_name": None,
            "plugin_version": None,
            "package_spec": None,
        }

    def test_lora_with_provenance_keys(self):
        from scope.core.workflows.schema import WorkflowLoRAProvenance

        lora = WorkflowLoRA(
            filename="test.safetensors",
            provenance=WorkflowLoRAProvenance(source="huggingface", repo_id="u/r"),
            expected_sha256="abc",
        )
        data = lora.model_dump(mode="json")
        assert data["provenance"]["source"] == "huggingface"
        assert data["provenance"]["repo_id"] == "u/r"
        assert data["expected_sha256"] == "abc"
        # id field present even when None
        assert "id" in data


class TestProvenanceSubclass:
    """WorkflowLoRAProvenance must explicitly accept unknown fields
    so the workflow schema remains forward-compatible even if the
    upstream LoRAProvenance model changes its extra policy."""

    def test_workflow_provenance_accepts_unknown(self):
        from scope.core.workflows.schema import WorkflowLoRAProvenance

        p = WorkflowLoRAProvenance.model_validate(
            {"source": "civitai", "model_id": "123", "future_field": "ok"}
        )
        assert p.source == "civitai"
        assert p.model_id == "123"

    def test_workflow_provenance_extra_is_explicit(self):
        """Guard: WorkflowLoRAProvenance must set extra='ignore' explicitly
        so it stays forward-compatible regardless of upstream changes."""
        from scope.core.workflows.schema import WorkflowLoRAProvenance

        assert WorkflowLoRAProvenance.model_config.get("extra") == "ignore"
