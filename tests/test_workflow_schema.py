"""Tests for the workflow schema and validation endpoints."""

from __future__ import annotations

from datetime import UTC, datetime

from scope.core.workflows.schema import (
    WORKFLOW_FORMAT_VERSION,
    ScopeWorkflow,
    WorkflowLoRA,
    WorkflowPipelineSource,
)

from .workflow_helpers import make_workflow as _make_workflow

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


# ---------------------------------------------------------------------------
# Forward-compatibility tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


class TestNegativeValidation:
    """Verify that invalid inputs raise ValidationError."""

    def test_invalid_source_type(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            WorkflowPipelineSource(type="invalid")

    def test_metadata_missing_name(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            WorkflowMetadata(
                created_at="2025-01-01T00:00:00Z",
                scope_version="0.1.0",
            )


class TestWorkflowEndpoints:
    """Tests for the /api/v1/workflow/schema and /api/v1/workflow/validate endpoints."""

    def test_workflow_schema_endpoint(self):
        """GET /api/v1/workflow/schema returns valid JSON Schema."""
        from fastapi.testclient import TestClient

        from scope.server.app import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/v1/workflow/schema")
        assert resp.status_code == 200
        data = resp.json()
        assert "properties" in data
        assert "title" in data
        assert "$defs" in data or "definitions" in data or "properties" in data

    def test_workflow_validate_valid(self):
        """POST /api/v1/workflow/validate returns 200 for a valid document."""
        from fastapi.testclient import TestClient

        from scope.server.app import app

        client = TestClient(app, raise_server_exceptions=False)
        doc = _make_workflow().model_dump(mode="json")
        resp = client.post("/api/v1/workflow/validate", json=doc)
        assert resp.status_code == 200
        body = resp.json()
        assert body["format"] == "scope-workflow"
        assert body["metadata"]["name"] == "test"

    def test_workflow_validate_invalid(self):
        """POST /api/v1/workflow/validate returns 422 for a bad document."""
        from fastapi.testclient import TestClient

        from scope.server.app import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/v1/workflow/validate", json={"bad": "data"})
        assert resp.status_code == 422
