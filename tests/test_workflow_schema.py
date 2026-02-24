"""Tests for the workflow schema and export logic."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from scope.core.workflows.schema import (
    WORKFLOW_FORMAT_VERSION,
    ScopeWorkflow,
    WorkflowLoRA,
    WorkflowPipelineSource,
)

from .workflow_helpers import make_workflow as _make_workflow

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
        assert lora.sha256 is None
        assert lora.id is None


# ---------------------------------------------------------------------------
# Export / build_workflow tests
# ---------------------------------------------------------------------------


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
        plm = _mock_plugin_manager()

        wf = build_workflow(
            name="my workflow",
            pipelines_input=[
                {"pipeline_id": "longlive", "params": {"height": 480, "width": 640}},
            ],
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
            pipelines_input=[
                {"pipeline_id": "ext_pipe", "params": {"height": 720}},
            ],
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
        plm = _mock_plugin_manager()

        wf = build_workflow(
            name="lora test",
            pipelines_input=[
                {
                    "pipeline_id": "longlive",
                    "params": {"height": 480},
                    "loras": [
                        {"path": "/models/lora/my_lora.safetensors", "scale": 0.8}
                    ],
                },
            ],
            plugin_manager=plm,
            lora_dir=lora_dir,
            lora_merge_mode="on_the_fly",
        )

        lora = wf.pipelines[0].loras[0]
        assert lora.filename == "my_lora.safetensors"
        assert lora.weight == 0.8
        assert lora.merge_mode == "on_the_fly"
        assert lora.provenance is not None
        assert lora.provenance.source == "huggingface"
        assert lora.provenance.repo_id == "user/lora-repo"
        assert lora.sha256 == "abc123"

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_multiple_pipelines(self, mock_manifest, mock_config, mock_ver):
        from scope.core.lora.manifest import LoRAManifest
        from scope.core.workflows.export import build_workflow

        mock_manifest.return_value = LoRAManifest()
        plm = _mock_plugin_manager()

        wf = build_workflow(
            name="multi",
            pipelines_input=[
                {
                    "pipeline_id": "pipe_a",
                    "params": {"height": 480, "guidance_scale": 7.5},
                },
                {
                    "pipeline_id": "pipe_b",
                    "params": {"height": 720, "guidance_scale": 3.0, "steps": 10},
                },
            ],
            plugin_manager=plm,
            lora_dir=Path("/models/lora"),
        )

        assert len(wf.pipelines) == 2
        assert wf.pipelines[0].params["guidance_scale"] == 7.5
        assert "steps" not in wf.pipelines[0].params
        assert wf.pipelines[1].params["guidance_scale"] == 3.0
        assert wf.pipelines[1].params["steps"] == 10

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_per_lora_merge_mode(self, mock_manifest, mock_config, mock_ver):
        """Per-LoRA merge_mode overrides the global lora_merge_mode."""
        from scope.core.lora.manifest import LoRAManifest
        from scope.core.workflows.export import build_workflow

        mock_manifest.return_value = LoRAManifest()
        plm = _mock_plugin_manager()

        wf = build_workflow(
            name="merge mode test",
            pipelines_input=[
                {
                    "pipeline_id": "longlive",
                    "params": {"height": 480},
                    "loras": [
                        {"path": "/lora/a.safetensors", "scale": 0.8},
                        {
                            "path": "/lora/b.safetensors",
                            "scale": 1.0,
                            "merge_mode": "on_the_fly",
                        },
                    ],
                },
            ],
            plugin_manager=plm,
            lora_dir=Path("/lora"),
            lora_merge_mode="permanent_merge",
        )

        loras = wf.pipelines[0].loras
        assert loras[0].merge_mode == "permanent_merge"  # falls back to global
        assert loras[1].merge_mode == "on_the_fly"  # per-LoRA override

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch("scope.core.lora.manifest.load_manifest")
    def test_unknown_params_stripped(self, mock_manifest, mock_ver):
        """Params not in the pipeline config schema are dropped."""
        from scope.core.lora.manifest import LoRAManifest
        from scope.core.workflows.export import build_workflow

        class _ConfigWithFields:
            pipeline_version = "1.0.0"
            model_fields = {"height": ..., "width": ...}

        mock_manifest.return_value = LoRAManifest()
        plm = _mock_plugin_manager()

        with patch(
            "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
            return_value=_ConfigWithFields,
        ):
            wf = build_workflow(
                name="filter test",
                pipelines_input=[
                    {
                        "pipeline_id": "longlive",
                        "params": {
                            "height": 480,
                            "width": 640,
                            "bogus_param": "should be dropped",
                        },
                    },
                ],
                plugin_manager=plm,
                lora_dir=Path("/models/lora"),
            )

        assert wf.pipelines[0].params == {"height": 480, "width": 640}
        assert "bogus_param" not in wf.pipelines[0].params


# ---------------------------------------------------------------------------
# Leszko review scenario tests (Task 2 -- wrong export data)
# ---------------------------------------------------------------------------


class TestLeszkoScenario:
    """Reproduce the exact scenario from the PR review: video-depth-anything
    preprocessor + longlive main pipeline + LoRA.  The old backend-driven
    export dropped longlive and attached the LoRA to video-depth-anything.
    The frontend-driven export must return exactly what the frontend sends."""

    @patch("importlib.metadata.version", return_value="0.1.4")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_both_pipelines_present(self, mock_manifest, mock_config, mock_ver):
        from scope.core.lora.manifest import (
            LoRAManifest,
            LoRAManifestEntry,
            LoRAProvenance,
        )
        from scope.core.workflows.export import build_workflow

        manifest = LoRAManifest(
            entries={
                "daydream-scope-dissolve.safetensors": LoRAManifestEntry(
                    filename="daydream-scope-dissolve.safetensors",
                    provenance=LoRAProvenance(
                        source="civitai",
                        url="https://civitai.com/api/download/models/2680702",
                    ),
                    sha256="fd373e09",
                    size_bytes=2048,
                    added_at=datetime(2025, 1, 1, tzinfo=UTC),
                )
            }
        )
        mock_manifest.return_value = manifest
        plm = _mock_plugin_manager()
        lora_dir = Path("/models/lora")

        wf = build_workflow(
            name="My Workflow",
            pipelines_input=[
                {
                    "pipeline_id": "video-depth-anything",
                    "params": {"height": 512, "width": 512},
                },
                {
                    "pipeline_id": "longlive",
                    "params": {
                        "height": 512,
                        "width": 512,
                        "vace_enabled": True,
                    },
                    "loras": [
                        {
                            "path": "/models/lora/daydream-scope-dissolve.safetensors",
                            "scale": 1.0,
                        }
                    ],
                },
            ],
            plugin_manager=plm,
            lora_dir=lora_dir,
        )

        # Both pipelines present
        assert len(wf.pipelines) == 2

        # video-depth-anything has NO LoRAs
        vda = wf.pipelines[0]
        assert vda.pipeline_id == "video-depth-anything"
        assert vda.loras == []
        assert vda.params["height"] == 512

        # longlive HAS the LoRA
        ll = wf.pipelines[1]
        assert ll.pipeline_id == "longlive"
        assert len(ll.loras) == 1
        assert ll.loras[0].filename == "daydream-scope-dissolve.safetensors"
        assert ll.loras[0].provenance is not None
        assert ll.loras[0].provenance.source == "civitai"
        assert ll.loras[0].sha256 == "fd373e09"

        # params on longlive are correct
        assert ll.params["vace_enabled"] is True

    @patch("importlib.metadata.version", return_value="0.1.4")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_user_sees_krea_not_longlive(self, mock_manifest, mock_config, mock_ver):
        """Task 4 scenario: user started longlive, stopped, selected krea.
        Frontend sends only krea (what user currently sees), not longlive."""
        from scope.core.lora.manifest import LoRAManifest
        from scope.core.workflows.export import build_workflow

        mock_manifest.return_value = LoRAManifest()
        plm = _mock_plugin_manager()

        wf = build_workflow(
            name="krea session",
            pipelines_input=[
                {
                    "pipeline_id": "krea",
                    "params": {"height": 720, "width": 1280, "strength": 0.6},
                },
            ],
            plugin_manager=plm,
            lora_dir=Path("/models/lora"),
        )

        assert len(wf.pipelines) == 1
        assert wf.pipelines[0].pipeline_id == "krea"
        assert wf.pipelines[0].params["strength"] == 0.6
        # longlive must NOT appear -- backend no longer injects loaded pipelines
        ids = [p.pipeline_id for p in wf.pipelines]
        assert "longlive" not in ids


# ---------------------------------------------------------------------------
# LoRA isolation and ordering (Task 2 -- LoRAs on wrong pipeline)
# ---------------------------------------------------------------------------


class TestLoRAIsolation:
    """Each pipeline's LoRAs must stay with that pipeline and not leak."""

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_loras_stay_on_their_pipeline(self, mock_manifest, mock_config, mock_ver):
        """3 pipelines: A has 2 LoRAs, B has none, C has 1 LoRA.
        Verify each pipeline gets exactly its own LoRAs."""
        from scope.core.lora.manifest import LoRAManifest
        from scope.core.workflows.export import build_workflow

        mock_manifest.return_value = LoRAManifest()
        plm = _mock_plugin_manager()

        wf = build_workflow(
            name="isolation",
            pipelines_input=[
                {
                    "pipeline_id": "pipe_a",
                    "params": {"height": 480},
                    "loras": [
                        {"path": "/lora/style.safetensors", "scale": 0.5},
                        {"path": "/lora/face.safetensors", "scale": 0.8},
                    ],
                },
                {
                    "pipeline_id": "pipe_b",
                    "params": {"height": 720},
                    # no loras key at all
                },
                {
                    "pipeline_id": "pipe_c",
                    "params": {},
                    "loras": [
                        {"path": "/lora/detail.safetensors", "scale": 1.0},
                    ],
                },
            ],
            plugin_manager=plm,
            lora_dir=Path("/lora"),
        )

        assert len(wf.pipelines) == 3

        # pipe_a: exactly 2 LoRAs in order
        assert len(wf.pipelines[0].loras) == 2
        assert wf.pipelines[0].loras[0].filename == "style.safetensors"
        assert wf.pipelines[0].loras[0].weight == 0.5
        assert wf.pipelines[0].loras[1].filename == "face.safetensors"
        assert wf.pipelines[0].loras[1].weight == 0.8

        # pipe_b: zero LoRAs
        assert wf.pipelines[1].loras == []

        # pipe_c: exactly 1 LoRA
        assert len(wf.pipelines[2].loras) == 1
        assert wf.pipelines[2].loras[0].filename == "detail.safetensors"

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_same_lora_on_multiple_pipelines(
        self, mock_manifest, mock_config, mock_ver
    ):
        """Same LoRA file referenced from two pipelines with different weights.
        Each pipeline must get its own independent copy."""
        from scope.core.lora.manifest import (
            LoRAManifest,
            LoRAManifestEntry,
            LoRAProvenance,
        )
        from scope.core.workflows.export import build_workflow

        manifest = LoRAManifest(
            entries={
                "shared.safetensors": LoRAManifestEntry(
                    filename="shared.safetensors",
                    provenance=LoRAProvenance(source="huggingface", repo_id="u/r"),
                    sha256="aaa",
                    size_bytes=100,
                    added_at=datetime(2025, 1, 1, tzinfo=UTC),
                )
            }
        )
        mock_manifest.return_value = manifest
        plm = _mock_plugin_manager()

        wf = build_workflow(
            name="shared lora",
            pipelines_input=[
                {
                    "pipeline_id": "pipe_a",
                    "params": {},
                    "loras": [
                        {"path": "/lora/shared.safetensors", "scale": 0.3},
                    ],
                },
                {
                    "pipeline_id": "pipe_b",
                    "params": {},
                    "loras": [
                        {"path": "/lora/shared.safetensors", "scale": 0.9},
                    ],
                },
            ],
            plugin_manager=plm,
            lora_dir=Path("/lora"),
        )

        # Both pipelines have the LoRA with their own weight
        assert wf.pipelines[0].loras[0].filename == "shared.safetensors"
        assert wf.pipelines[0].loras[0].weight == 0.3
        assert wf.pipelines[0].loras[0].sha256 == "aaa"

        assert wf.pipelines[1].loras[0].filename == "shared.safetensors"
        assert wf.pipelines[1].loras[0].weight == 0.9
        assert wf.pipelines[1].loras[0].sha256 == "aaa"

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_lora_merge_mode_applies_to_all(self, mock_manifest, mock_config, mock_ver):
        """lora_merge_mode is a top-level setting that applies to every LoRA
        across every pipeline."""
        from scope.core.lora.manifest import LoRAManifest
        from scope.core.workflows.export import build_workflow

        mock_manifest.return_value = LoRAManifest()
        plm = _mock_plugin_manager()

        wf = build_workflow(
            name="merge mode",
            pipelines_input=[
                {
                    "pipeline_id": "pipe_a",
                    "params": {},
                    "loras": [{"path": "/lora/a.safetensors", "scale": 1.0}],
                },
                {
                    "pipeline_id": "pipe_b",
                    "params": {},
                    "loras": [{"path": "/lora/b.safetensors", "scale": 1.0}],
                },
            ],
            plugin_manager=plm,
            lora_dir=Path("/lora"),
            lora_merge_mode="on_the_fly",
        )

        assert wf.pipelines[0].loras[0].merge_mode == "on_the_fly"
        assert wf.pipelines[1].loras[0].merge_mode == "on_the_fly"


# ---------------------------------------------------------------------------
# Pipeline ordering and identity preservation
# ---------------------------------------------------------------------------


class TestPipelineOrdering:
    """Pipeline order and identity must be preserved exactly as sent."""

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_order_preserved_with_many_pipelines(
        self, mock_manifest, mock_config, mock_ver
    ):
        from scope.core.lora.manifest import LoRAManifest
        from scope.core.workflows.export import build_workflow

        mock_manifest.return_value = LoRAManifest()
        plm = _mock_plugin_manager()

        ids = ["alpha", "beta", "gamma", "delta", "epsilon"]
        wf = build_workflow(
            name="ordering",
            pipelines_input=[
                {"pipeline_id": pid, "params": {"index": i}}
                for i, pid in enumerate(ids)
            ],
            plugin_manager=plm,
            lora_dir=Path("/lora"),
        )

        assert len(wf.pipelines) == 5
        for i, pid in enumerate(ids):
            assert wf.pipelines[i].pipeline_id == pid
            assert wf.pipelines[i].params["index"] == i

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_empty_pipelines_list(self, mock_manifest, mock_config, mock_ver):
        from scope.core.lora.manifest import LoRAManifest
        from scope.core.workflows.export import build_workflow

        mock_manifest.return_value = LoRAManifest()
        plm = _mock_plugin_manager()

        wf = build_workflow(
            name="empty",
            pipelines_input=[],
            plugin_manager=plm,
            lora_dir=Path("/lora"),
        )

        assert wf.pipelines == []
        assert wf.metadata.name == "empty"

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_pipeline_with_no_params_no_loras(
        self, mock_manifest, mock_config, mock_ver
    ):
        """Minimal pipeline input: just pipeline_id."""
        from scope.core.lora.manifest import LoRAManifest
        from scope.core.workflows.export import build_workflow

        mock_manifest.return_value = LoRAManifest()
        plm = _mock_plugin_manager()

        wf = build_workflow(
            name="minimal",
            pipelines_input=[{"pipeline_id": "bare"}],
            plugin_manager=plm,
            lora_dir=Path("/lora"),
        )

        assert len(wf.pipelines) == 1
        assert wf.pipelines[0].pipeline_id == "bare"
        assert wf.pipelines[0].params == {}
        assert wf.pipelines[0].loras == []


# ---------------------------------------------------------------------------
# Source type mapping (Task 3 -- all 4 source types)
# ---------------------------------------------------------------------------


class TestSourceTypeMapping:
    """All 4 source types must map correctly: builtin, pypi, git, local."""

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_all_source_types(self, mock_manifest, mock_config, mock_ver):
        from scope.core.lora.manifest import LoRAManifest
        from scope.core.workflows.export import build_workflow

        mock_manifest.return_value = LoRAManifest()
        plm = _mock_plugin_manager(
            plugin_for={
                "pypi_pipe": "plugin-pypi",
                "git_pipe": "plugin-git",
                "local_pipe": "plugin-local",
            },
            plugin_list=[
                {
                    "name": "plugin-pypi",
                    "version": "1.0",
                    "source": "pypi",
                    "package_spec": "plugin-pypi>=1.0",
                },
                {
                    "name": "plugin-git",
                    "version": "abc123",
                    "source": "git",
                    "package_spec": "git+https://github.com/user/plugin-git.git",
                },
                {
                    "name": "plugin-local",
                    "version": "0.0.1",
                    "source": "local",
                    "package_spec": "/home/user/plugin-local",
                },
            ],
        )

        wf = build_workflow(
            name="sources",
            pipelines_input=[
                {"pipeline_id": "builtin_pipe", "params": {}},
                {"pipeline_id": "pypi_pipe", "params": {}},
                {"pipeline_id": "git_pipe", "params": {}},
                {"pipeline_id": "local_pipe", "params": {}},
            ],
            plugin_manager=plm,
            lora_dir=Path("/lora"),
        )

        assert wf.pipelines[0].source.type == "builtin"
        assert wf.pipelines[0].source.plugin_name is None

        assert wf.pipelines[1].source.type == "pypi"
        assert wf.pipelines[1].source.plugin_name == "plugin-pypi"
        assert wf.pipelines[1].source.package_spec == "plugin-pypi>=1.0"

        assert wf.pipelines[2].source.type == "git"
        assert wf.pipelines[2].source.plugin_name == "plugin-git"
        assert wf.pipelines[2].source.plugin_version == "abc123"

        assert wf.pipelines[3].source.type == "local"
        assert wf.pipelines[3].source.plugin_name == "plugin-local"

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_unknown_source_string_defaults_to_pypi(
        self, mock_manifest, mock_config, mock_ver
    ):
        """If plugin_manager returns an unrecognized source string,
        it should fall back to 'pypi'."""
        from scope.core.lora.manifest import LoRAManifest
        from scope.core.workflows.export import build_workflow

        mock_manifest.return_value = LoRAManifest()
        plm = _mock_plugin_manager(
            plugin_for={"ext": "plugin-x"},
            plugin_list=[
                {"name": "plugin-x", "version": "1.0", "source": "something_new"},
            ],
        )

        wf = build_workflow(
            name="fallback",
            pipelines_input=[{"pipeline_id": "ext", "params": {}}],
            plugin_manager=plm,
            lora_dir=Path("/lora"),
        )

        assert wf.pipelines[0].source.type == "pypi"

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_mixed_builtin_and_plugin_in_same_export(
        self, mock_manifest, mock_config, mock_ver
    ):
        """Builtin and plugin pipelines in the same export get different
        source metadata."""
        from scope.core.lora.manifest import LoRAManifest
        from scope.core.workflows.export import build_workflow

        mock_manifest.return_value = LoRAManifest()
        plm = _mock_plugin_manager(
            plugin_for={"faceswap": "scope-plugin-faceswap"},
            plugin_list=[
                {
                    "name": "scope-plugin-faceswap",
                    "version": "0.2.0",
                    "source": "pypi",
                    "package_spec": "scope-plugin-faceswap>=0.2",
                },
            ],
        )

        wf = build_workflow(
            name="mixed",
            pipelines_input=[
                {"pipeline_id": "longlive", "params": {"height": 480}},
                {"pipeline_id": "faceswap", "params": {"swap_mode": "full"}},
            ],
            plugin_manager=plm,
            lora_dir=Path("/lora"),
        )

        assert wf.pipelines[0].source.type == "builtin"
        assert wf.pipelines[0].source.plugin_name is None
        assert wf.pipelines[1].source.type == "pypi"
        assert wf.pipelines[1].source.plugin_name == "scope-plugin-faceswap"


# ---------------------------------------------------------------------------
# Pipeline version (Task 9)
# ---------------------------------------------------------------------------


class TestPipelineVersion:
    """Pipeline version comes from PipelineRegistry.get_config_class()."""

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch("scope.core.pipelines.registry.PipelineRegistry.get_config_class")
    @patch("scope.core.lora.manifest.load_manifest")
    def test_version_from_config_class(self, mock_manifest, mock_config, mock_ver):
        from scope.core.lora.manifest import LoRAManifest
        from scope.core.workflows.export import build_workflow

        class V3Config:
            pipeline_version = "3.1.0"

        mock_config.return_value = V3Config
        mock_manifest.return_value = LoRAManifest()
        plm = _mock_plugin_manager()

        wf = build_workflow(
            name="version",
            pipelines_input=[{"pipeline_id": "longlive", "params": {}}],
            plugin_manager=plm,
            lora_dir=Path("/lora"),
        )

        assert wf.pipelines[0].pipeline_version == "3.1.0"

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=None,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_version_none_when_no_config_class(
        self, mock_manifest, mock_config, mock_ver
    ):
        from scope.core.lora.manifest import LoRAManifest
        from scope.core.workflows.export import build_workflow

        mock_manifest.return_value = LoRAManifest()
        plm = _mock_plugin_manager()

        wf = build_workflow(
            name="unknown",
            pipelines_input=[{"pipeline_id": "nonexistent", "params": {}}],
            plugin_manager=plm,
            lora_dir=Path("/lora"),
        )

        assert wf.pipelines[0].pipeline_version is None

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=None,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_version_none_omitted_with_exclude_none(
        self, mock_manifest, mock_config, mock_ver
    ):
        """When pipeline_version is None, exclude_none=True strips it from output."""
        from scope.core.lora.manifest import LoRAManifest
        from scope.core.workflows.export import build_workflow

        mock_manifest.return_value = LoRAManifest()
        plm = _mock_plugin_manager()

        wf = build_workflow(
            name="omit version",
            pipelines_input=[{"pipeline_id": "nonexistent", "params": {}}],
            plugin_manager=plm,
            lora_dir=Path("/lora"),
        )

        data = wf.model_dump(mode="json", exclude_none=True)
        assert "pipeline_version" not in data["pipelines"][0]


# ---------------------------------------------------------------------------
# LoRA edge cases
# ---------------------------------------------------------------------------


class TestLoRAEdgeCases:
    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_lora_outside_lora_dir_uses_basename(
        self, mock_manifest, mock_config, mock_ver
    ):
        """LoRA path that can't be relativized falls back to file basename."""
        from scope.core.lora.manifest import LoRAManifest
        from scope.core.workflows.export import build_workflow

        mock_manifest.return_value = LoRAManifest()
        plm = _mock_plugin_manager()

        wf = build_workflow(
            name="outside",
            pipelines_input=[
                {
                    "pipeline_id": "longlive",
                    "params": {},
                    "loras": [
                        {
                            "path": "/completely/different/path/style.safetensors",
                            "scale": 0.7,
                        },
                    ],
                },
            ],
            plugin_manager=plm,
            lora_dir=Path("/models/lora"),
        )

        assert wf.pipelines[0].loras[0].filename == "style.safetensors"
        assert wf.pipelines[0].loras[0].weight == 0.7

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_lora_not_in_manifest(self, mock_manifest, mock_config, mock_ver):
        """LoRA with no manifest entry gets no provenance and no sha256."""
        from scope.core.lora.manifest import LoRAManifest
        from scope.core.workflows.export import build_workflow

        mock_manifest.return_value = LoRAManifest()  # empty
        plm = _mock_plugin_manager()

        wf = build_workflow(
            name="no manifest",
            pipelines_input=[
                {
                    "pipeline_id": "longlive",
                    "params": {},
                    "loras": [
                        {"path": "/lora/unknown.safetensors", "scale": 1.0},
                    ],
                },
            ],
            plugin_manager=plm,
            lora_dir=Path("/lora"),
        )

        lora = wf.pipelines[0].loras[0]
        assert lora.filename == "unknown.safetensors"
        assert lora.provenance is None
        assert lora.sha256 is None

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_lora_in_subdirectory(self, mock_manifest, mock_config, mock_ver):
        """LoRA in a subdirectory of lora_dir gets a relative path with
        forward slashes."""
        from scope.core.lora.manifest import (
            LoRAManifest,
            LoRAManifestEntry,
            LoRAProvenance,
        )
        from scope.core.workflows.export import build_workflow

        manifest = LoRAManifest(
            entries={
                "styles/anime.safetensors": LoRAManifestEntry(
                    filename="styles/anime.safetensors",
                    provenance=LoRAProvenance(source="huggingface", repo_id="u/r"),
                    sha256="bbb",
                    size_bytes=500,
                    added_at=datetime(2025, 1, 1, tzinfo=UTC),
                )
            }
        )
        mock_manifest.return_value = manifest
        plm = _mock_plugin_manager()

        wf = build_workflow(
            name="subdir",
            pipelines_input=[
                {
                    "pipeline_id": "longlive",
                    "params": {},
                    "loras": [
                        {"path": "/lora/styles/anime.safetensors", "scale": 1.0},
                    ],
                },
            ],
            plugin_manager=plm,
            lora_dir=Path("/lora"),
        )

        lora = wf.pipelines[0].loras[0]
        assert lora.filename == "styles/anime.safetensors"
        assert "\\" not in lora.filename
        assert lora.sha256 == "bbb"

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_lora_default_weight(self, mock_manifest, mock_config, mock_ver):
        """LoRA without explicit weight defaults to 1.0."""
        from scope.core.lora.manifest import LoRAManifest
        from scope.core.workflows.export import build_workflow

        mock_manifest.return_value = LoRAManifest()
        plm = _mock_plugin_manager()

        wf = build_workflow(
            name="defaults",
            pipelines_input=[
                {
                    "pipeline_id": "longlive",
                    "params": {},
                    "loras": [{"path": "/lora/x.safetensors"}],
                },
            ],
            plugin_manager=plm,
            lora_dir=Path("/lora"),
        )

        assert wf.pipelines[0].loras[0].weight == 1.0

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_default_merge_mode_is_permanent(
        self, mock_manifest, mock_config, mock_ver
    ):
        """When lora_merge_mode is not specified, default is permanent_merge."""
        from scope.core.lora.manifest import LoRAManifest
        from scope.core.workflows.export import build_workflow

        mock_manifest.return_value = LoRAManifest()
        plm = _mock_plugin_manager()

        wf = build_workflow(
            name="default merge",
            pipelines_input=[
                {
                    "pipeline_id": "longlive",
                    "params": {},
                    "loras": [{"path": "/lora/x.safetensors", "scale": 1.0}],
                },
            ],
            plugin_manager=plm,
            lora_dir=Path("/lora"),
        )

        assert wf.pipelines[0].loras[0].merge_mode == "permanent_merge"


# ---------------------------------------------------------------------------
# Params are not mutated
# ---------------------------------------------------------------------------


class TestParamsNotMutated:
    """build_workflow must not modify the input dicts."""

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_input_dicts_unchanged(self, mock_manifest, mock_config, mock_ver):
        from scope.core.lora.manifest import LoRAManifest
        from scope.core.workflows.export import build_workflow

        mock_manifest.return_value = LoRAManifest()
        plm = _mock_plugin_manager()

        original_params = {"height": 480, "width": 640}
        original_loras = [{"path": "/lora/a.safetensors", "scale": 0.5}]
        pipelines_input = [
            {
                "pipeline_id": "longlive",
                "params": original_params.copy(),
                "loras": [d.copy() for d in original_loras],
            },
        ]

        # Snapshot before
        params_before = dict(pipelines_input[0]["params"])
        loras_before = [dict(d) for d in pipelines_input[0]["loras"]]

        build_workflow(
            name="mutation test",
            pipelines_input=pipelines_input,
            plugin_manager=plm,
            lora_dir=Path("/lora"),
        )

        # Input dicts must not have been modified
        assert pipelines_input[0]["params"] == params_before
        assert pipelines_input[0]["loras"] == loras_before


# ---------------------------------------------------------------------------
# Schema field removal tests (Task 5 -- description/author removed)
# ---------------------------------------------------------------------------


class TestMetadataFieldsRemoved:
    """description and author fields must not exist on WorkflowMetadata."""

    def test_no_description_field(self):
        assert "description" not in WorkflowMetadata.model_fields

    def test_no_author_field(self):
        assert "author" not in WorkflowMetadata.model_fields

    def test_metadata_only_has_expected_fields(self):
        assert set(WorkflowMetadata.model_fields.keys()) == {
            "name",
            "created_at",
            "scope_version",
        }

    def test_old_json_with_description_and_author_still_loads(self):
        """Forward-compat: old exports that include description/author
        should still parse because of extra='ignore'."""
        data = {
            "format": "scope-workflow",
            "format_version": "1.0",
            "metadata": {
                "name": "old",
                "description": "some desc",
                "author": "someone",
                "created_at": "2025-01-01T00:00:00Z",
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
        assert wf.metadata.name == "old"
        assert not hasattr(wf.metadata, "description")
        assert not hasattr(wf.metadata, "author")


# ---------------------------------------------------------------------------
# pipeline_version optionality (Task 9)
# ---------------------------------------------------------------------------


class TestPipelineVersionOptional:
    def test_pipeline_version_is_optional_in_schema(self):
        assert WorkflowPipeline.model_fields["pipeline_version"].default is None

    def test_pipeline_without_version(self):
        p = WorkflowPipeline(
            pipeline_id="test",
            source=WorkflowPipelineSource(type="builtin"),
        )
        assert p.pipeline_version is None

    def test_pipeline_with_version(self):
        p = WorkflowPipeline(
            pipeline_id="test",
            pipeline_version="2.0.0",
            source=WorkflowPipelineSource(type="builtin"),
        )
        assert p.pipeline_version == "2.0.0"

    def test_old_json_with_pipeline_version_still_loads(self):
        """Old exports that include pipeline_version should still parse."""
        data = {
            "format": "scope-workflow",
            "format_version": "1.0",
            "metadata": {
                "name": "old",
                "created_at": "2025-01-01T00:00:00Z",
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
        assert wf.pipelines[0].pipeline_version == "1.0.0"

    def test_json_without_pipeline_version_still_loads(self):
        """Workflows omitting pipeline_version should parse with None."""
        data = {
            "format": "scope-workflow",
            "format_version": "1.0",
            "metadata": {
                "name": "new",
                "created_at": "2025-01-01T00:00:00Z",
                "scope_version": "0.2.0",
            },
            "pipelines": [
                {
                    "pipeline_id": "p",
                    "source": {"type": "builtin"},
                }
            ],
        }
        wf = ScopeWorkflow.model_validate(data)
        assert wf.pipelines[0].pipeline_version is None


# ---------------------------------------------------------------------------
# Field rename test (Task 8 -- expected_sha256 -> sha256)
# ---------------------------------------------------------------------------


class TestSha256Rename:
    def test_field_is_named_sha256(self):
        assert "sha256" in WorkflowLoRA.model_fields
        assert "expected_sha256" not in WorkflowLoRA.model_fields

    def test_old_json_with_expected_sha256_ignored(self):
        """Old exports using expected_sha256 are silently dropped
        (extra='ignore')."""
        data = {
            "filename": "x.safetensors",
            "expected_sha256": "abc",
        }
        lora = WorkflowLoRA.model_validate(data)
        assert lora.sha256 is None  # old field name dropped


# ---------------------------------------------------------------------------
# exclude_none serialization (Task 7 -- null values)
# ---------------------------------------------------------------------------


class TestExcludeNoneSerialization:
    """model_dump(exclude_none=True) must strip all None values from output."""

    def test_builtin_source_no_nulls(self):
        src = WorkflowPipelineSource(type="builtin")
        data = src.model_dump(mode="json", exclude_none=True)
        assert data == {"type": "builtin"}
        assert None not in data.values()

    def test_lora_without_provenance_no_nulls(self):
        lora = WorkflowLoRA(filename="test.safetensors")
        data = lora.model_dump(mode="json", exclude_none=True)
        assert "provenance" not in data
        assert "sha256" not in data
        assert "id" not in data
        assert data["filename"] == "test.safetensors"

    def test_full_workflow_no_nulls_anywhere(self):
        """Recursively check that no None values exist in the output when
        using exclude_none=True on a full workflow."""
        wf = _make_workflow()
        data = wf.model_dump(mode="json", exclude_none=True)
        _assert_no_none_values(data)

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_full_export_no_nulls(self, mock_manifest, mock_config, mock_ver):
        """Simulate what the endpoint does: build_workflow then
        model_dump(mode='json', exclude_none=True). No None values anywhere."""
        from scope.core.lora.manifest import LoRAManifest
        from scope.core.workflows.export import build_workflow

        mock_manifest.return_value = LoRAManifest()
        plm = _mock_plugin_manager()

        wf = build_workflow(
            name="no nulls",
            pipelines_input=[
                {
                    "pipeline_id": "longlive",
                    "params": {"height": 480},
                },
            ],
            plugin_manager=plm,
            lora_dir=Path("/lora"),
        )

        data = wf.model_dump(mode="json", exclude_none=True)
        _assert_no_none_values(data)
        # builtin source should only have "type"
        assert data["pipelines"][0]["source"] == {"type": "builtin"}

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_lora_with_provenance_still_has_data_after_exclude_none(
        self, mock_manifest, mock_config, mock_ver
    ):
        """When a LoRA has provenance, the provenance data must survive
        exclude_none (only truly-None fields are stripped)."""
        from scope.core.lora.manifest import (
            LoRAManifest,
            LoRAManifestEntry,
            LoRAProvenance,
        )
        from scope.core.workflows.export import build_workflow

        manifest = LoRAManifest(
            entries={
                "test.safetensors": LoRAManifestEntry(
                    filename="test.safetensors",
                    provenance=LoRAProvenance(
                        source="huggingface",
                        repo_id="user/repo",
                        hf_filename="test.safetensors",
                    ),
                    sha256="deadbeef",
                    size_bytes=1024,
                    added_at=datetime(2025, 1, 1, tzinfo=UTC),
                )
            }
        )
        mock_manifest.return_value = manifest
        plm = _mock_plugin_manager()

        wf = build_workflow(
            name="provenance",
            pipelines_input=[
                {
                    "pipeline_id": "longlive",
                    "params": {},
                    "loras": [{"path": "/lora/test.safetensors", "scale": 1.0}],
                },
            ],
            plugin_manager=plm,
            lora_dir=Path("/lora"),
        )

        data = wf.model_dump(mode="json", exclude_none=True)
        _assert_no_none_values(data)
        lora_data = data["pipelines"][0]["loras"][0]
        assert lora_data["sha256"] == "deadbeef"
        assert lora_data["provenance"]["source"] == "huggingface"
        assert lora_data["provenance"]["repo_id"] == "user/repo"
        assert lora_data["provenance"]["hf_filename"] == "test.safetensors"
        # None provenance fields should be absent
        assert "model_id" not in lora_data["provenance"]
        assert "version_id" not in lora_data["provenance"]
        assert "url" not in lora_data["provenance"]


def _assert_no_none_values(obj, path=""):
    """Recursively assert no None values exist in a dict/list structure."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            current = f"{path}.{key}" if path else key
            assert value is not None, f"None value found at {current}"
            _assert_no_none_values(value, current)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _assert_no_none_values(item, f"{path}[{i}]")


# ---------------------------------------------------------------------------
# Request model tests (Tasks 11, 13 -- endpoint concerns)
# ---------------------------------------------------------------------------


class TestRequestModels:
    """Verify the Pydantic request models used by the endpoint."""

    def test_default_name(self):
        """Task 11: WorkflowExportRequest defaults name to 'Untitled Workflow'."""
        from scope.server.app import WorkflowExportRequest

        req = WorkflowExportRequest(pipelines=[])
        assert req.name == "Untitled Workflow"

    def test_explicit_name(self):
        from scope.server.app import WorkflowExportRequest

        req = WorkflowExportRequest(name="My Session", pipelines=[])
        assert req.name == "My Session"

    def test_default_lora_merge_mode(self):
        from scope.server.app import WorkflowExportRequest

        req = WorkflowExportRequest(pipelines=[])
        assert req.lora_merge_mode == "permanent_merge"

    def test_export_lora_input_defaults(self):
        from scope.server.app import ExportLoRAInput

        lora = ExportLoRAInput(path="/lora/test.safetensors")
        assert lora.scale == 1.0
        assert lora.merge_mode is None

    def test_export_pipeline_input_defaults(self):
        from scope.server.app import ExportPipelineInput

        pipe = ExportPipelineInput(pipeline_id="longlive")
        assert pipe.params == {}
        assert pipe.loras == []

    def test_export_pipeline_input_model_dump_preserves_loras(self):
        """The endpoint calls p.model_dump() on each ExportPipelineInput.
        Verify the LoRA data survives serialization."""
        from scope.server.app import ExportLoRAInput, ExportPipelineInput

        pipe = ExportPipelineInput(
            pipeline_id="longlive",
            params={"height": 480},
            loras=[
                ExportLoRAInput(path="/lora/a.safetensors", scale=0.5),
                ExportLoRAInput(path="/lora/b.safetensors", scale=0.8),
            ],
        )
        dumped = pipe.model_dump()
        assert dumped["pipeline_id"] == "longlive"
        assert dumped["params"] == {"height": 480}
        assert len(dumped["loras"]) == 2
        assert dumped["loras"][0]["path"] == "/lora/a.safetensors"
        assert dumped["loras"][0]["scale"] == 0.5
        assert dumped["loras"][1]["path"] == "/lora/b.safetensors"

    def test_endpoint_is_sync(self):
        """Task 13: export_workflow must be a regular function, not async."""
        import asyncio
        import inspect

        from scope.server.app import export_workflow

        assert not asyncio.iscoroutinefunction(export_workflow)
        assert not inspect.iscoroutinefunction(export_workflow)

    def test_endpoint_does_not_depend_on_pipeline_manager(self):
        """Task 4: the endpoint no longer needs PipelineManager."""
        import inspect

        from scope.server.app import export_workflow

        sig = inspect.signature(export_workflow)
        param_names = list(sig.parameters.keys())
        # Should only have 'request', not 'pm' or 'pipeline_manager'
        assert "pm" not in param_names
        assert "pipeline_manager" not in param_names


# ---------------------------------------------------------------------------
# End-to-end request -> response simulation
# ---------------------------------------------------------------------------


class TestEndToEndExport:
    """Simulate the full endpoint flow: request model -> model_dump() ->
    build_workflow() -> model_dump(exclude_none=True).  This is the closest
    we can get to an integration test without starting the server."""

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_realistic_multi_pipeline_export(
        self, mock_manifest, mock_config, mock_ver
    ):
        from scope.core.lora.manifest import (
            LoRAManifest,
            LoRAManifestEntry,
            LoRAProvenance,
        )
        from scope.core.workflows.export import build_workflow
        from scope.server.app import (
            ExportLoRAInput,
            ExportPipelineInput,
            WorkflowExportRequest,
        )

        manifest = LoRAManifest(
            entries={
                "dissolve.safetensors": LoRAManifestEntry(
                    filename="dissolve.safetensors",
                    provenance=LoRAProvenance(
                        source="civitai",
                        url="https://civitai.com/api/download/models/123",
                    ),
                    sha256="cafe",
                    size_bytes=2048,
                    added_at=datetime(2025, 1, 1, tzinfo=UTC),
                )
            }
        )
        mock_manifest.return_value = manifest

        plm = _mock_plugin_manager()

        # Step 1: Build request as the frontend would
        request = WorkflowExportRequest(
            name="My Session",
            pipelines=[
                ExportPipelineInput(
                    pipeline_id="video-depth-anything",
                    params={"height": 512, "width": 512},
                ),
                ExportPipelineInput(
                    pipeline_id="longlive",
                    params={
                        "height": 512,
                        "width": 512,
                        "vace_enabled": True,
                    },
                    loras=[
                        ExportLoRAInput(
                            path="/models/lora/dissolve.safetensors",
                            weight=1.0,
                        ),
                    ],
                ),
            ],
            lora_merge_mode="permanent_merge",
        )

        # Step 2: Call build_workflow as the endpoint does
        workflow = build_workflow(
            name=request.name,
            pipelines_input=[p.model_dump() for p in request.pipelines],
            plugin_manager=plm,
            lora_dir=Path("/models/lora"),
            lora_merge_mode=request.lora_merge_mode,
        )

        # Step 3: Serialize as the endpoint does
        data = workflow.model_dump(mode="json", exclude_none=True)

        # Verify structure
        assert data["format"] == "scope-workflow"
        assert data["metadata"]["name"] == "My Session"
        assert len(data["pipelines"]) == 2

        # Pipeline 0: video-depth-anything, no LoRAs
        p0 = data["pipelines"][0]
        assert p0["pipeline_id"] == "video-depth-anything"
        assert p0["source"] == {"type": "builtin"}
        assert p0["loras"] == []
        assert p0["params"]["height"] == 512

        # Pipeline 1: longlive, with LoRA
        p1 = data["pipelines"][1]
        assert p1["pipeline_id"] == "longlive"
        assert p1["params"]["vace_enabled"] is True
        assert len(p1["loras"]) == 1
        assert p1["loras"][0]["filename"] == "dissolve.safetensors"
        assert p1["loras"][0]["sha256"] == "cafe"
        assert p1["loras"][0]["provenance"]["source"] == "civitai"

        # No None values anywhere in the output
        _assert_no_none_values(data)

    @patch("importlib.metadata.version", return_value="0.5.0")
    @patch(
        "scope.core.pipelines.registry.PipelineRegistry.get_config_class",
        return_value=_FakeConfigClass,
    )
    @patch("scope.core.lora.manifest.load_manifest")
    def test_default_name_end_to_end(self, mock_manifest, mock_config, mock_ver):
        """Using default name results in 'Untitled Workflow' in output."""
        from scope.core.lora.manifest import LoRAManifest
        from scope.core.workflows.export import build_workflow
        from scope.server.app import ExportPipelineInput, WorkflowExportRequest

        mock_manifest.return_value = LoRAManifest()
        plm = _mock_plugin_manager()

        request = WorkflowExportRequest(
            pipelines=[
                ExportPipelineInput(pipeline_id="longlive", params={"height": 480}),
            ],
        )

        workflow = build_workflow(
            name=request.name,
            pipelines_input=[p.model_dump() for p in request.pipelines],
            plugin_manager=plm,
            lora_dir=Path("/lora"),
            lora_merge_mode=request.lora_merge_mode,
        )

        data = workflow.model_dump(mode="json", exclude_none=True)
        assert data["metadata"]["name"] == "Untitled Workflow"


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
        """This is the key test -- LoRAProvenance from manifest.py lacks
        extra='ignore', so we use WorkflowLoRAProvenance to add it."""
        data = _full_workflow_dict()
        data["pipelines"][0]["loras"][0]["provenance"]["download_count"] = 9999
        wf = ScopeWorkflow.model_validate(data)
        assert wf.pipelines[0].loras[0].provenance.repo_id == "user/repo"


class TestMinimalDocument:
    """The smallest valid workflow -- only required fields."""

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
            "timeline",
            "min_scope_version",
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
        data = src.model_dump(mode="json", exclude_none=True)
        assert data == {"type": "builtin"}

    def test_lora_with_provenance_keys(self):
        from scope.core.workflows.schema import WorkflowLoRAProvenance

        lora = WorkflowLoRA(
            filename="test.safetensors",
            provenance=WorkflowLoRAProvenance(source="huggingface", repo_id="u/r"),
            sha256="abc",
        )
        data = lora.model_dump(mode="json")
        assert data["provenance"]["source"] == "huggingface"
        assert data["provenance"]["repo_id"] == "u/r"
        assert data["sha256"] == "abc"
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
