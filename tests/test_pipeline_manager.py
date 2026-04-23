"""Tests for PipelineManager multi-pipeline reconciliation logic.

Covers the core behaviour of ``_load_pipelines_sync``:
1. Reuse already-loaded pipelines when the request matches (same pipeline_id + params).
2. Re-key an existing instance when only the node_id changed.
3. Load a fresh instance when no compatible one exists.
4. Unload stale instances that are no longer in the request.
5. Never share an instance between two graph nodes — even if pipeline_id and
   params are identical, each node gets its own load.
"""

from unittest.mock import MagicMock, patch

from scope.server.pipeline_manager import PipelineManager, PipelineStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager() -> PipelineManager:
    return PipelineManager()


def _preload(
    manager: PipelineManager,
    key: str,
    pipeline_id: str,
    params: dict | None = None,
) -> MagicMock:
    """Simulate an already-loaded pipeline in the manager's internal state."""
    mock_pipeline = MagicMock(name=f"pipeline-{key}")
    manager._pipelines[key] = mock_pipeline
    manager._pipeline_statuses[key] = PipelineStatus.LOADED
    manager._pipeline_registry_ids[key] = pipeline_id
    manager._pipeline_load_params[key] = params or {}
    return mock_pipeline


def _load_pipelines(
    manager: PipelineManager,
    pipelines: list[tuple[str, str, dict | None]],
) -> tuple[bool, list, list]:
    """Call ``_load_pipelines_sync`` with mocked load/unload, return results.

    Returns:
        (success, load_calls, unload_calls) where each *_calls list contains
        the positional arguments of each invocation.
    """
    load_calls: list[tuple] = []
    unload_calls: list[tuple] = []

    def fake_load(pipeline_id, load_params=None, **kwargs):
        instance_key = kwargs.get("instance_key") or pipeline_id
        load_calls.append((instance_key, pipeline_id, load_params))
        # Simulate successful load into internal state
        manager._pipelines[instance_key] = MagicMock(name=f"pipeline-{instance_key}")
        manager._pipeline_statuses[instance_key] = PipelineStatus.LOADED
        manager._pipeline_registry_ids[instance_key] = pipeline_id
        manager._pipeline_load_params[instance_key] = load_params or {}
        return True

    def fake_unload(pipeline_id, **kwargs):
        unload_calls.append((pipeline_id,))
        manager._pipelines.pop(pipeline_id, None)
        manager._pipeline_statuses.pop(pipeline_id, None)
        manager._pipeline_registry_ids.pop(pipeline_id, None)
        manager._pipeline_load_params.pop(pipeline_id, None)

    with (
        patch.object(manager, "_load_pipeline_by_id_sync", side_effect=fake_load),
        patch.object(manager, "_unload_pipeline_by_id_unsafe", side_effect=fake_unload),
    ):
        success = manager._load_pipelines_sync(pipelines)

    return success, load_calls, unload_calls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadPipelinesEmpty:
    def test_returns_false_for_empty_list(self):
        manager = _make_manager()
        success = manager._load_pipelines_sync([])
        assert success is False


class TestSinglePipelineLoad:
    def test_loads_new_pipeline(self):
        manager = _make_manager()
        success, loads, unloads = _load_pipelines(
            manager, [("node_a", "longlive", None)]
        )
        assert success is True
        assert len(loads) == 1
        assert loads[0] == ("node_a", "longlive", None)
        assert len(unloads) == 0

    def test_reuses_already_loaded_pipeline(self):
        """An already-loaded pipeline with matching params should NOT be reloaded."""
        manager = _make_manager()
        _preload(manager, "node_a", "longlive", {})

        success, loads, unloads = _load_pipelines(
            manager, [("node_a", "longlive", None)]
        )
        assert success is True
        assert len(loads) == 0, "Should reuse, not reload"
        assert len(unloads) == 0

    def test_reloads_when_params_change(self):
        """Different load_params should trigger a reload."""
        manager = _make_manager()
        _preload(manager, "node_a", "longlive", {"quality": "low"})

        success, loads, unloads = _load_pipelines(
            manager, [("node_a", "longlive", {"quality": "high"})]
        )
        assert success is True
        assert len(loads) == 1
        assert loads[0][2] == {"quality": "high"}

    def test_reloads_when_pipeline_id_changes(self):
        """Switching pipeline type at the same node_id should trigger a reload."""
        manager = _make_manager()
        _preload(manager, "node_a", "longlive", {})

        success, loads, unloads = _load_pipelines(manager, [("node_a", "krea", None)])
        assert success is True
        assert len(loads) == 1
        assert loads[0][1] == "krea"


class TestRekey:
    def test_rekeys_when_node_id_changes(self):
        """If a compatible pipeline exists under a different key, re-key it."""
        manager = _make_manager()
        original = _preload(manager, "old_node", "longlive", {})

        success, loads, unloads = _load_pipelines(
            manager, [("new_node", "longlive", None)]
        )
        assert success is True
        assert len(loads) == 0, "Should re-key, not reload"
        assert len(unloads) == 0, "Old key was re-keyed, not unloaded"
        # The original instance should now be under the new key
        assert manager._pipelines["new_node"] is original
        assert "old_node" not in manager._pipelines

    def test_does_not_steal_reserved_key(self):
        """Don't re-key from a key that another new entry needs at that position."""
        manager = _make_manager()
        _preload(manager, "node_b", "longlive", {})

        # node_a wants longlive, node_b wants krea.
        # node_b's old instance is longlive — but "node_b" is reserved for
        # the second entry, so it must not be stolen by node_a.
        success, loads, unloads = _load_pipelines(
            manager,
            [("node_a", "longlive", None), ("node_b", "krea", None)],
        )
        assert success is True
        # node_a cannot steal from node_b, so both need fresh loads.
        # node_b's old longlive gets unloaded (by the node_b krea load replacing it).
        assert any(key == "node_a" for key, _, _ in loads)
        assert any(key == "node_b" for key, _, _ in loads)


class TestUnloadStale:
    def test_unloads_pipeline_no_longer_requested(self):
        """Pipelines not in the new request set should be unloaded."""
        manager = _make_manager()
        _preload(manager, "node_a", "longlive", {})
        _preload(manager, "node_b", "krea", {})

        # Only request node_a; node_b should be unloaded.
        success, loads, unloads = _load_pipelines(
            manager, [("node_a", "longlive", None)]
        )
        assert success is True
        assert len(loads) == 0
        assert ("node_b",) in unloads

    def test_unloads_all_when_completely_different(self):
        """When no old pipelines match, all should be unloaded."""
        manager = _make_manager()
        _preload(manager, "old_a", "longlive", {})
        _preload(manager, "old_b", "krea", {})

        success, loads, unloads = _load_pipelines(
            manager, [("new_a", "other_pipeline", None)]
        )
        assert success is True
        assert len(loads) == 1
        assert len(unloads) == 2

    def test_cleans_up_stale_statuses(self):
        """Status entries for keys not in the new set should be removed."""
        manager = _make_manager()
        _preload(manager, "node_a", "longlive", {})
        # Simulate an errored status from a previous load
        manager._pipeline_statuses["stale_key"] = PipelineStatus.ERROR

        success, loads, unloads = _load_pipelines(
            manager, [("node_a", "longlive", None)]
        )
        assert "stale_key" not in manager._pipeline_statuses


class TestMultipleInstances:
    """Even if two entries share the same pipeline_id and params, each must
    get its own instance — we don't support sharing a pipeline instance
    between multiple graph nodes."""

    def test_duplicate_pipeline_both_loaded_fresh(self):
        """Two nodes requesting the same pipeline should each trigger a load."""
        manager = _make_manager()
        success, loads, unloads = _load_pipelines(
            manager,
            [("node_a", "longlive", None), ("node_b", "longlive", None)],
        )
        assert success is True
        assert len(loads) == 2
        loaded_keys = {key for key, _, _ in loads}
        assert loaded_keys == {"node_a", "node_b"}

    def test_duplicate_pipeline_one_preloaded(self):
        """One node reuses the preloaded instance; the other gets a fresh load."""
        manager = _make_manager()
        _preload(manager, "node_a", "longlive", {})

        success, loads, unloads = _load_pipelines(
            manager,
            [("node_a", "longlive", None), ("node_b", "longlive", None)],
        )
        assert success is True
        # node_a reuses, node_b loads fresh
        assert len(loads) == 1
        assert loads[0][0] == "node_b"

    def test_duplicate_pipeline_rekey_only_one(self):
        """When an old instance can be re-keyed, only one of the duplicates uses it."""
        manager = _make_manager()
        _preload(manager, "old_node", "longlive", {})

        success, loads, unloads = _load_pipelines(
            manager,
            [("node_a", "longlive", None), ("node_b", "longlive", None)],
        )
        assert success is True
        # One should be re-keyed from old_node, the other loaded fresh
        assert len(loads) == 1
        # The re-keyed one should be in _pipelines, and old_node gone
        assert "old_node" not in manager._pipelines
        assert "node_a" in manager._pipelines
        assert "node_b" in manager._pipelines


class TestHelperMethods:
    def test_is_pipeline_loaded_with_true(self):
        manager = _make_manager()
        _preload(manager, "node_a", "longlive", {"q": 1})
        assert manager._is_pipeline_loaded_with("node_a", "longlive", {"q": 1}) is True

    def test_is_pipeline_loaded_with_wrong_params(self):
        manager = _make_manager()
        _preload(manager, "node_a", "longlive", {"q": 1})
        assert manager._is_pipeline_loaded_with("node_a", "longlive", {"q": 2}) is False

    def test_is_pipeline_loaded_with_wrong_pipeline_id(self):
        manager = _make_manager()
        _preload(manager, "node_a", "longlive", {})
        assert manager._is_pipeline_loaded_with("node_a", "krea", {}) is False

    def test_is_pipeline_loaded_with_not_loaded_status(self):
        manager = _make_manager()
        _preload(manager, "node_a", "longlive", {})
        manager._pipeline_statuses["node_a"] = PipelineStatus.LOADING
        assert manager._is_pipeline_loaded_with("node_a", "longlive", {}) is False

    def test_is_pipeline_loaded_with_missing_key(self):
        manager = _make_manager()
        assert manager._is_pipeline_loaded_with("no_such", "longlive", {}) is False

    def test_rekey_pipeline(self):
        manager = _make_manager()
        original = _preload(manager, "old", "longlive", {"q": 1})

        manager._rekey_pipeline("old", "new")

        assert "old" not in manager._pipelines
        assert manager._pipelines["new"] is original
        assert manager._pipeline_registry_ids["new"] == "longlive"
        assert manager._pipeline_load_params["new"] == {"q": 1}
        assert manager._pipeline_statuses["new"] == PipelineStatus.LOADED

    def test_find_reusable_pipeline_finds_match(self):
        manager = _make_manager()
        _preload(manager, "old_node", "longlive", {})

        result = manager._find_reusable_pipeline(
            "new_node", "longlive", {}, claimed_keys=set(), reserved_keys=set()
        )
        assert result == "old_node"

    def test_find_reusable_pipeline_skips_claimed(self):
        manager = _make_manager()
        _preload(manager, "old_node", "longlive", {})

        result = manager._find_reusable_pipeline(
            "new_node",
            "longlive",
            {},
            claimed_keys={"old_node"},
            reserved_keys=set(),
        )
        assert result is None

    def test_find_reusable_pipeline_skips_reserved(self):
        manager = _make_manager()
        _preload(manager, "reserved_node", "longlive", {})

        result = manager._find_reusable_pipeline(
            "new_node",
            "longlive",
            {},
            claimed_keys=set(),
            reserved_keys={"reserved_node"},
        )
        assert result is None

    def test_find_reusable_pipeline_allows_self_reserved(self):
        """A key that is reserved for the *same* node_id can still match."""
        manager = _make_manager()
        _preload(manager, "node_a", "longlive", {})

        result = manager._find_reusable_pipeline(
            "node_a", "longlive", {}, claimed_keys=set(), reserved_keys={"node_a"}
        )
        assert result == "node_a"


# ---------------------------------------------------------------------------
# Tests for _sanitize_asset_path and _sanitize_initial_params
# ---------------------------------------------------------------------------


class TestSanitizeAssetPath:
    """Tests for PipelineManager._sanitize_asset_path and _sanitize_initial_params."""

    def _mock_assets_dir(self, tmp_path, monkeypatch):
        """Patch get_assets_dir to return tmp_path/assets."""
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(
            "scope.server.pipeline_manager.PipelineManager._sanitize_asset_path.__func__",
            None,
            raising=False,
        )
        return assets_dir

    def test_windows_backslash_path_is_rewritten(self, tmp_path, monkeypatch):
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        monkeypatch.setattr(
            "scope.server.models_config.get_assets_dir", lambda: assets_dir
        )
        result = PipelineManager._sanitize_asset_path(
            r"C:\Users\Joshu\.daydream-scope\assets\ShinraFireForce.webp"
        )
        assert result == (assets_dir / "ShinraFireForce.webp").as_posix()

    def test_windows_forward_slash_drive_path_is_rewritten(self, tmp_path, monkeypatch):
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        monkeypatch.setattr(
            "scope.server.models_config.get_assets_dir", lambda: assets_dir
        )
        result = PipelineManager._sanitize_asset_path(
            "C:/Users/Joshu/.daydream-scope/assets/ShinraFireForce.webp"
        )
        assert result == (assets_dir / "ShinraFireForce.webp").as_posix()

    def test_foreign_linux_tmp_path_is_rewritten(self, tmp_path, monkeypatch):
        """A /tmp/.daydream-scope/assets/… path from a different Linux machine is rewritten."""
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        monkeypatch.setattr(
            "scope.server.models_config.get_assets_dir", lambda: assets_dir
        )
        result = PipelineManager._sanitize_asset_path(
            "/tmp/.daydream-scope/assets/hakoniwa_abc.png"
        )
        assert result == (assets_dir / "hakoniwa_abc.png").as_posix()

    def test_relative_path_unchanged(self, tmp_path, monkeypatch):
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        monkeypatch.setattr(
            "scope.server.models_config.get_assets_dir", lambda: assets_dir
        )
        result = PipelineManager._sanitize_asset_path("image.png")
        assert result == "image.png"

    def test_sanitize_initial_params_none_value(self, tmp_path, monkeypatch):
        """_sanitize_initial_params should leave None values as None."""
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        monkeypatch.setattr(
            "scope.server.models_config.get_assets_dir", lambda: assets_dir
        )
        result = PipelineManager._sanitize_initial_params({"i2v_image": None})
        assert result["i2v_image"] is None

    def test_sanitize_initial_params_i2v_image_windows_path(self, tmp_path, monkeypatch):
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        monkeypatch.setattr(
            "scope.server.models_config.get_assets_dir", lambda: assets_dir
        )
        params = {
            "prompts": [{"text": "test"}],
            "i2v_image": r"C:\Users\Joshu\.daydream-scope\assets\ShinraFireForce.webp",
        }
        result = PipelineManager._sanitize_initial_params(params)
        assert result["i2v_image"] == (assets_dir / "ShinraFireForce.webp").as_posix()

    def test_sanitize_initial_params_i2v_image_linux_tmp_path(self, tmp_path, monkeypatch):
        """Linux /tmp path from a different machine is rewritten (issue #916)."""
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        monkeypatch.setattr(
            "scope.server.models_config.get_assets_dir", lambda: assets_dir
        )
        params = {
            "i2v_image": "/tmp/.daydream-scope/assets/hakoniwa_abc.png",
        }
        result = PipelineManager._sanitize_initial_params(params)
        assert result["i2v_image"] == (assets_dir / "hakoniwa_abc.png").as_posix()

    def test_sanitize_initial_params_images_list(self, tmp_path, monkeypatch):
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        monkeypatch.setattr(
            "scope.server.models_config.get_assets_dir", lambda: assets_dir
        )
        params = {
            "images": [
                r"C:\Users\Joshu\.daydream-scope\assets\foo.webp",
                r"C:\Users\Joshu\.daydream-scope\assets\bar.png",
            ]
        }
        result = PipelineManager._sanitize_initial_params(params)
        assert result["images"] == [
            (assets_dir / "foo.webp").as_posix(),
            (assets_dir / "bar.png").as_posix(),
        ]

    def test_sanitize_initial_params_no_asset_params_unchanged(self, tmp_path, monkeypatch):
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        monkeypatch.setattr(
            "scope.server.models_config.get_assets_dir", lambda: assets_dir
        )
        params = {"prompts": [{"text": "test"}], "seed": 42}
        result = PipelineManager._sanitize_initial_params(params)
        assert result == params
