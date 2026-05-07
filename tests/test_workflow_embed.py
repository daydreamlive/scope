"""Tests for embedding/extracting media assets in workflow JSON."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from scope.core.workflows.embed import (
    embed_workflow_assets,
    extract_workflow_assets,
)


@pytest.fixture
def cloud_mock():
    """Mock cloud backend that reports connected and records api_request calls."""
    mock = MagicMock()
    mock.is_connected = True
    # Mirror the real cloud's response shape: cloud-side path is returned in
    # the workflow it sends back. Tests can override via .api_request.return_value.
    mock.api_request = AsyncMock(
        return_value={
            "status": 200,
            "data": {
                "metadata": {"name": "t"},
                "pipelines": [
                    {
                        "params": {
                            "vace_ref_images": ["/root/.daydream-scope/assets/ref.png"]
                        }
                    }
                ],
            },
        }
    )
    return mock


def _png_bytes() -> bytes:
    # 1x1 transparent PNG (smallest valid)
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )


def _make_assets_dir(tmp_path: Path, files: dict[str, bytes]) -> Path:
    d = tmp_path / "assets"
    d.mkdir()
    for name, data in files.items():
        (d / name).write_bytes(data)
    return d


def test_embed_collects_referenced_files(tmp_path: Path) -> None:
    png = _png_bytes()
    assets = _make_assets_dir(tmp_path, {"ref.png": png})

    workflow = {
        "format": "scope-workflow",
        "format_version": "1.0",
        "metadata": {"name": "test"},
        "pipelines": [
            {
                "pipeline_id": "longlive",
                "params": {
                    "vace_ref_images": [str(assets / "ref.png")],
                    "noise_scale": 0.3,
                },
            }
        ],
    }

    result = embed_workflow_assets(workflow, assets)
    assert result["format_version"] == "1.1"
    assert "embedded_assets" in result
    assert len(result["embedded_assets"]) == 1
    entry = result["embedded_assets"][0]
    assert entry["filename"] == "ref.png"
    assert entry["mime_type"] == "image/png"
    assert entry["size_bytes"] == len(png)
    assert entry["sha256"] == hashlib.sha256(png).hexdigest()
    assert base64.b64decode(entry["data"]) == png
    # Path references unchanged in the source workflow.
    assert result["pipelines"][0]["params"]["vace_ref_images"][0] == str(
        assets / "ref.png"
    )


def test_embed_skips_missing_files(tmp_path: Path) -> None:
    assets = _make_assets_dir(tmp_path, {})
    workflow = {
        "metadata": {"name": "test"},
        "pipelines": [{"params": {"vace_ref_images": ["/nonexistent/foo.png"]}}],
    }
    result = embed_workflow_assets(workflow, assets)
    assert "embedded_assets" not in result


def test_embed_resolves_basename_when_absolute_path_missing(tmp_path: Path) -> None:
    """Workflow exported on machine A points at /home/alice/.daydream-scope/
    assets/x.png; on machine B that path doesn't exist but x.png lives in the
    local assets dir."""
    png = _png_bytes()
    assets = _make_assets_dir(tmp_path, {"x.png": png})

    workflow = {
        "metadata": {"name": "t"},
        "pipelines": [
            {
                "params": {
                    "vace_ref_images": ["/home/alice/.daydream-scope/assets/x.png"]
                }
            }
        ],
    }
    result = embed_workflow_assets(workflow, assets)
    assert len(result["embedded_assets"]) == 1
    assert result["embedded_assets"][0]["filename"] == "x.png"


def test_embed_dedups_same_basename(tmp_path: Path) -> None:
    png = _png_bytes()
    assets = _make_assets_dir(tmp_path, {"shared.png": png})
    workflow = {
        "metadata": {"name": "t"},
        "pipelines": [
            {"params": {"vace_ref_images": [str(assets / "shared.png")]}},
            {
                "params": {
                    "first_frame_image": str(assets / "shared.png"),
                    "last_frame_image": str(assets / "shared.png"),
                }
            },
        ],
    }
    result = embed_workflow_assets(workflow, assets)
    assert len(result["embedded_assets"]) == 1


def test_extract_writes_assets_and_rewrites_paths(tmp_path: Path) -> None:
    png = _png_bytes()
    target_assets = tmp_path / "target_assets"  # importer's local dir, empty

    workflow = {
        "metadata": {"name": "t"},
        "format_version": "1.1",
        "pipelines": [
            {
                "params": {
                    "vace_ref_images": ["/old/machine/ref.png"],
                }
            }
        ],
        "embedded_assets": [
            {
                "filename": "ref.png",
                "mime_type": "image/png",
                "size_bytes": len(png),
                "sha256": hashlib.sha256(png).hexdigest(),
                "data": base64.b64encode(png).decode("ascii"),
            }
        ],
    }

    result = extract_workflow_assets(workflow, target_assets)

    assert "embedded_assets" not in result
    new_path = result["pipelines"][0]["params"]["vace_ref_images"][0]
    assert Path(new_path).name == "ref.png"
    assert Path(new_path).exists()
    assert Path(new_path).read_bytes() == png
    # Confirm written into target_assets
    assert Path(new_path).is_relative_to(target_assets.resolve())


def test_extract_idempotent_when_sha_matches(tmp_path: Path) -> None:
    png = _png_bytes()
    sha = hashlib.sha256(png).hexdigest()
    target_assets = tmp_path / "target_assets"
    target_assets.mkdir()
    (target_assets / "ref.png").write_bytes(png)
    mtime_before = (target_assets / "ref.png").stat().st_mtime_ns

    workflow = {
        "metadata": {"name": "t"},
        "pipelines": [{"params": {"vace_ref_images": ["/old/ref.png"]}}],
        "embedded_assets": [
            {
                "filename": "ref.png",
                "mime_type": "image/png",
                "size_bytes": len(png),
                "sha256": sha,
                "data": base64.b64encode(png).decode("ascii"),
            }
        ],
    }
    extract_workflow_assets(workflow, target_assets)
    mtime_after = (target_assets / "ref.png").stat().st_mtime_ns
    # Existing identical file is not rewritten
    assert mtime_before == mtime_after


def test_extract_dedups_basename_collision_with_different_content(
    tmp_path: Path,
) -> None:
    other_png = _png_bytes() + b"x"  # different content, same name
    embedded_png = _png_bytes()
    target_assets = tmp_path / "target"
    target_assets.mkdir()
    (target_assets / "ref.png").write_bytes(other_png)

    workflow = {
        "metadata": {"name": "t"},
        "pipelines": [{"params": {"vace_ref_images": ["/old/ref.png"]}}],
        "embedded_assets": [
            {
                "filename": "ref.png",
                "mime_type": "image/png",
                "size_bytes": len(embedded_png),
                "sha256": hashlib.sha256(embedded_png).hexdigest(),
                "data": base64.b64encode(embedded_png).decode("ascii"),
            }
        ],
    }
    result = extract_workflow_assets(workflow, target_assets)
    new_path = result["pipelines"][0]["params"]["vace_ref_images"][0]
    # Should be a renamed file like ref_imported1.png
    assert Path(new_path).name != "ref.png"
    assert Path(new_path).read_bytes() == embedded_png
    # Original untouched
    assert (target_assets / "ref.png").read_bytes() == other_png


def test_round_trip_via_graph_ui_state(tmp_path: Path) -> None:
    """Embedding should reach into graph.ui_state.nodes[].data.imagePath."""
    png = _png_bytes()
    src_assets = _make_assets_dir(tmp_path, {"hero.png": png})

    workflow = {
        "metadata": {"name": "t"},
        "pipelines": [],
        "graph": {
            "nodes": [
                {
                    "id": "media1",
                    "type": "source",
                    "source_mode": "video_file",
                    "source_name": str(src_assets / "hero.png"),
                }
            ],
            "edges": [],
            "ui_state": {
                "nodes": [
                    {
                        "id": "img1",
                        "type": "image",
                        "data": {"imagePath": str(src_assets / "hero.png")},
                    }
                ]
            },
        },
    }

    embedded = embed_workflow_assets(workflow, src_assets)
    assert len(embedded["embedded_assets"]) == 1

    # Now extract on a fresh dir
    target = tmp_path / "fresh"
    extracted = extract_workflow_assets(embedded, target)
    new_source_name = extracted["graph"]["nodes"][0]["source_name"]
    new_image_path = extracted["graph"]["ui_state"]["nodes"][0]["data"]["imagePath"]
    assert Path(new_source_name).read_bytes() == png
    assert Path(new_image_path).read_bytes() == png
    assert Path(new_source_name).is_relative_to(target.resolve())


def test_extract_does_not_rewrite_freeform_text_ending_in_extension(
    tmp_path: Path,
) -> None:
    """A prompt or label that happens to end in ``cat.png`` (no path separator)
    must not be rewritten to a local filesystem path on import. Only strings
    that look path-shaped — contain ``/``/``\\`` or exactly equal an embedded
    filename — are eligible for substitution.
    """
    png = _png_bytes()
    target_assets = tmp_path / "target_assets"

    workflow = {
        "metadata": {"name": "t"},
        "pipelines": [
            {
                "params": {
                    "prompt": "an image of cat.png on a couch",
                    "vace_ref_images": ["/old/cat.png"],
                }
            }
        ],
        "embedded_assets": [
            {
                "filename": "cat.png",
                "mime_type": "image/png",
                "size_bytes": len(png),
                "sha256": hashlib.sha256(png).hexdigest(),
                "data": base64.b64encode(png).decode("ascii"),
            }
        ],
    }

    result = extract_workflow_assets(workflow, target_assets)
    # Path-shaped reference is rewritten to the new on-disk location.
    new_ref = result["pipelines"][0]["params"]["vace_ref_images"][0]
    assert Path(new_ref).name == "cat.png"
    assert Path(new_ref).is_relative_to(target_assets.resolve())
    # Free-form prompt is left alone.
    assert (
        result["pipelines"][0]["params"]["prompt"] == "an image of cat.png on a couch"
    )


def test_extract_rewrites_bare_basename_reference(tmp_path: Path) -> None:
    """A workflow that references a file by bare basename (no separator) is a
    known shape and must still be rewritten to the on-disk path so it
    resolves at session-run time."""
    png = _png_bytes()
    target_assets = tmp_path / "target_assets"

    workflow = {
        "metadata": {"name": "t"},
        "pipelines": [{"params": {"vace_ref_images": ["ref.png"]}}],
        "embedded_assets": [
            {
                "filename": "ref.png",
                "mime_type": "image/png",
                "size_bytes": len(png),
                "sha256": hashlib.sha256(png).hexdigest(),
                "data": base64.b64encode(png).decode("ascii"),
            }
        ],
    }

    result = extract_workflow_assets(workflow, target_assets)
    new_ref = result["pipelines"][0]["params"]["vace_ref_images"][0]
    assert Path(new_ref).name == "ref.png"
    assert Path(new_ref).exists()


def test_extract_handles_windows_style_backslash_paths(tmp_path: Path) -> None:
    """Workflows exported on Windows reference assets with ``\\`` separators.
    On POSIX, ``Path(value).name`` would treat the whole string as a single
    component. The basename extractor must normalize separators so the
    rewrite still fires."""
    png = _png_bytes()
    target_assets = tmp_path / "target_assets"

    workflow = {
        "metadata": {"name": "t"},
        "pipelines": [
            {"params": {"vace_ref_images": ["C:\\Users\\Alice\\assets\\ref.png"]}}
        ],
        "embedded_assets": [
            {
                "filename": "ref.png",
                "mime_type": "image/png",
                "size_bytes": len(png),
                "sha256": hashlib.sha256(png).hexdigest(),
                "data": base64.b64encode(png).decode("ascii"),
            }
        ],
    }

    result = extract_workflow_assets(workflow, target_assets)
    new_ref = result["pipelines"][0]["params"]["vace_ref_images"][0]
    assert Path(new_ref).name == "ref.png"
    assert Path(new_ref).exists()


def test_embed_preserves_higher_format_version(tmp_path: Path) -> None:
    """If a future workflow exports with a higher format_version, embedding
    additional media must not silently downgrade the version."""
    png = _png_bytes()
    assets = _make_assets_dir(tmp_path, {"ref.png": png})
    workflow = {
        "format_version": "1.5",
        "metadata": {"name": "t"},
        "pipelines": [{"params": {"vace_ref_images": [str(assets / "ref.png")]}}],
    }
    result = embed_workflow_assets(workflow, assets)
    assert result["format_version"] == "1.5"


def test_extract_endpoint_in_cloud_mode_writes_local_copy_for_previews(
    tmp_path: Path, cloud_mock: MagicMock
) -> None:
    """In cloud mode the extract endpoint must still materialize embedded
    assets locally, otherwise the frontend's ``<img>``/``<video>`` previews
    (which load from ``/api/v1/assets/{basename}``, a local-only endpoint)
    show as broken images after import.

    Regression test for the case where importing a workflow with
    ``embedded_assets`` while connected to the cloud left the local assets
    dir untouched, so previews 404'd.
    """
    png = _png_bytes()
    local_assets = tmp_path / "local_assets"
    local_assets.mkdir()

    workflow_with_embedded = {
        "format": "scope-workflow",
        "format_version": "1.1",
        "metadata": {"name": "t"},
        "pipelines": [
            {"params": {"vace_ref_images": ["/some/origin/machine/ref.png"]}}
        ],
        "embedded_assets": [
            {
                "filename": "ref.png",
                "mime_type": "image/png",
                "size_bytes": len(png),
                "sha256": hashlib.sha256(png).hexdigest(),
                "data": base64.b64encode(png).decode("ascii"),
            }
        ],
    }

    with (
        patch("scope.server.app.get_assets_dir", return_value=local_assets),
        patch("scope.server.app.livepeer", cloud_mock),
    ):
        from scope.server.app import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/api/v1/workflow/extract", json=workflow_with_embedded)

    assert response.status_code == 200, response.text
    # Cloud's response is what the client sees — paths point at the cloud's
    # filesystem so sessions resolve them on the cloud worker.
    body = response.json()
    assert (
        body["pipelines"][0]["params"]["vace_ref_images"][0]
        == "/root/.daydream-scope/assets/ref.png"
    )
    # And the cloud was actually called.
    cloud_mock.api_request.assert_awaited_once()
    # The fix: a local copy must exist so the preview thumbnail endpoint
    # (which strips the path to its basename) can serve it.
    local_copy = local_assets / "ref.png"
    assert local_copy.exists(), (
        "local copy missing — frontend previews will show broken image"
    )
    assert local_copy.read_bytes() == png
    # Happy path: no warning header when local materialization succeeds.
    assert response.headers.get("X-Scope-Warning") is None


def test_extract_endpoint_cloud_mode_surfaces_local_failure_via_header(
    tmp_path: Path, cloud_mock: MagicMock
) -> None:
    """When the local materialization step in cloud mode fails (e.g. the
    assets dir is unwritable), the frontend won't be able to render previews.
    The endpoint must still proxy to cloud so workflow paths resolve at
    session-run time, but it must surface the local failure via the
    ``X-Scope-Warning`` response header so the import dialog can toast a
    user-visible notice."""
    png = _png_bytes()
    local_assets = tmp_path / "local_assets"  # intentionally not created

    workflow_with_embedded = {
        "format": "scope-workflow",
        "format_version": "1.1",
        "metadata": {"name": "t"},
        "pipelines": [
            {"params": {"vace_ref_images": ["/some/origin/machine/ref.png"]}}
        ],
        "embedded_assets": [
            {
                "filename": "ref.png",
                "mime_type": "image/png",
                "size_bytes": len(png),
                "sha256": hashlib.sha256(png).hexdigest(),
                "data": base64.b64encode(png).decode("ascii"),
            }
        ],
    }

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk on fire")

    with (
        patch("scope.server.app.get_assets_dir", return_value=local_assets),
        patch("scope.server.app.livepeer", cloud_mock),
        patch("scope.server.app.extract_workflow_assets", side_effect=_raise),
    ):
        from scope.server.app import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/api/v1/workflow/extract", json=workflow_with_embedded)

    assert response.status_code == 200, response.text
    # Cloud was still called so the workflow's session-run paths resolve.
    cloud_mock.api_request.assert_awaited_once()
    # Frontend can read this header and toast the warning.
    assert response.headers.get("X-Scope-Warning") is not None
    assert "previews" in response.headers["X-Scope-Warning"].lower()
