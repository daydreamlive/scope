"""Unit tests for PluginManager class."""

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from scope.core.plugins.manager import (
    PluginDependencyError,
    PluginInstallError,
    PluginInUseError,
    PluginManager,
    PluginNotEditableError,
    PluginNotFoundError,
    get_plugin_manager,
)


class TestPluginManagerInit:
    """Tests for PluginManager initialization."""

    def test_creates_singleton(self):
        """Verify get_plugin_manager() returns same instance."""
        # Reset singleton for test
        import scope.core.plugins.manager as manager_module

        with patch.object(manager_module, "_plugin_manager", None):
            instance1 = get_plugin_manager()
            instance2 = get_plugin_manager()
            assert instance1 is instance2

    def test_initializes_pluggy_manager(self):
        """Verify pluggy PluginManager is created."""
        pm = PluginManager()
        assert pm._pm is not None
        assert pm.pm is not None

    def test_thread_safe_initialization(self):
        """Verify concurrent calls return same instance."""
        import scope.core.plugins.manager as manager_module

        with patch.object(manager_module, "_plugin_manager", None):
            results = []
            errors = []

            def get_instance():
                try:
                    instance = get_plugin_manager()
                    results.append(instance)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=get_instance) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            assert len(results) == 10
            # All should be the same instance
            assert all(r is results[0] for r in results)


class TestPluginSourceDetection:
    """Tests for plugin source detection from direct_url.json."""

    def test_detects_pypi_source(self, tmp_path):
        """No direct_url.json should mean PyPI source."""
        pm = PluginManager()

        # Mock distribution without direct_url.json
        mock_dist = MagicMock()
        mock_dist._path = tmp_path  # Empty directory - no direct_url.json

        source, editable, editable_path, git_url = pm._get_plugin_source(mock_dist)

        assert source == "pypi"
        assert editable is False
        assert editable_path is None
        assert git_url is None

    def test_detects_git_source(self, tmp_path):
        """Has vcs_info with git should be Git source."""
        pm = PluginManager()

        # Create direct_url.json for git source
        direct_url = tmp_path / "direct_url.json"
        direct_url.write_text(
            json.dumps(
                {
                    "url": "https://github.com/user/repo.git",
                    "vcs_info": {"vcs": "git", "commit_id": "abc123"},
                }
            )
        )

        mock_dist = MagicMock()
        mock_dist._path = tmp_path

        source, editable, editable_path, git_url = pm._get_plugin_source(mock_dist)

        assert source == "git"
        assert editable is False
        assert editable_path is None
        assert git_url == "https://github.com/user/repo.git"

    def test_detects_local_editable(self, tmp_path):
        """Has dir_info.editable=true should be Local source."""
        pm = PluginManager()

        # Create direct_url.json for editable local install
        direct_url = tmp_path / "direct_url.json"
        # Use file:///path format
        local_path = "/path/to/package"
        direct_url.write_text(
            json.dumps({"url": f"file://{local_path}", "dir_info": {"editable": True}})
        )

        mock_dist = MagicMock()
        mock_dist._path = tmp_path

        source, editable, editable_path, git_url = pm._get_plugin_source(mock_dist)

        assert source == "local"
        assert editable is True
        assert editable_path is not None
        assert git_url is None

    def test_handles_missing_direct_url(self):
        """Gracefully defaults to PyPI when no _path."""
        pm = PluginManager()

        mock_dist = MagicMock()
        mock_dist._path = None  # No path

        source, editable, editable_path, git_url = pm._get_plugin_source(mock_dist)

        assert source == "pypi"
        assert editable is False
        assert git_url is None


class TestListPlugins:
    """Tests for list_plugins_async method."""

    def test_returns_empty_list_when_no_plugins(self):
        """No scope entry points should return empty list."""
        pm = PluginManager()

        # Mock distributions to return no scope entry points
        mock_dist = MagicMock()
        mock_dist.entry_points = []

        with patch("importlib.metadata.distributions", return_value=[mock_dist]):
            with pm._lock:
                plugins = pm._list_plugins_sync()

        assert plugins == []

    def test_returns_plugin_info(self):
        """Mock distribution with scope entry point should return plugin info."""
        pm = PluginManager()

        mock_ep = MagicMock()
        mock_ep.group = "scope"
        mock_ep.name = "test-plugin"

        mock_dist = MagicMock()
        mock_dist.entry_points = [mock_ep]
        mock_dist.metadata = {"Name": "test-plugin", "Version": "1.0.0"}
        mock_dist._path = None  # PyPI source

        with patch("importlib.metadata.distributions", return_value=[mock_dist]):
            plugins = pm._list_plugins_sync()

        assert len(plugins) == 1
        assert plugins[0]["name"] == "test-plugin"
        assert plugins[0]["version"] == "1.0.0"
        assert plugins[0]["source"] == "pypi"

    def test_handles_errors_gracefully(self):
        """Plugin info errors should not crash the listing."""
        pm = PluginManager()

        # Mock a distribution that raises an error
        mock_dist = MagicMock()
        mock_dist.entry_points = MagicMock(side_effect=Exception("Test error"))

        with patch("importlib.metadata.distributions", return_value=[mock_dist]):
            plugins = pm._list_plugins_sync()

        # Should return empty list without crashing
        assert plugins == []


class TestCheckUpdates:
    """Tests for check_updates_async method."""

    def test_skips_local_plugins(self):
        """Local plugins should return null for update info."""
        pm = PluginManager()

        # Mock a local plugin
        with patch.object(
            pm,
            "_list_plugins_sync",
            return_value=[
                {
                    "name": "local-plugin",
                    "version": "1.0.0",
                    "source": "local",
                    "editable": True,
                }
            ],
        ):
            updates = pm._check_updates_sync()

        assert len(updates) == 1
        assert updates[0]["name"] == "local-plugin"
        assert updates[0]["latest_version"] is None
        assert updates[0]["update_available"] is None

    def test_handles_pypi_errors(self):
        """Network errors should return null gracefully."""
        pm = PluginManager()

        # Mock a PyPI plugin
        with patch.object(
            pm,
            "_list_plugins_sync",
            return_value=[
                {"name": "pypi-plugin", "version": "1.0.0", "source": "pypi"}
            ],
        ):
            # Mock urllib to raise an error
            with patch(
                "urllib.request.urlopen", side_effect=Exception("Network error")
            ):
                updates = pm._check_updates_sync()

        assert len(updates) == 1
        assert updates[0]["latest_version"] is None
        assert updates[0]["update_available"] is None

    def test_detects_update_available(self):
        """Different version should set update_available=True."""
        pm = PluginManager()

        # Mock a PyPI plugin
        with patch.object(
            pm,
            "_list_plugins_sync",
            return_value=[
                {"name": "pypi-plugin", "version": "1.0.0", "source": "pypi"}
            ],
        ):
            # Mock urllib to return a newer version
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(
                {"info": {"version": "2.0.0"}}
            ).encode()
            mock_response.__enter__ = lambda s: s
            mock_response.__exit__ = MagicMock(return_value=False)

            with patch("urllib.request.urlopen", return_value=mock_response):
                updates = pm._check_updates_sync()

        assert len(updates) == 1
        assert updates[0]["latest_version"] == "2.0.0"
        assert updates[0]["update_available"] is True


class TestGetVersionFromResolved:
    """Tests for _get_version_from_resolved method."""

    def test_extracts_version_from_pypi_package(self, tmp_path):
        """Should extract version from package==version format."""
        pm = PluginManager()

        resolved_file = tmp_path / "resolved.txt"
        resolved_file.write_text("some-package==1.2.3\nother-package==4.5.6\n")

        version = pm._get_version_from_resolved("some-package", str(resolved_file))

        assert version == "1.2.3"

    def test_extracts_commit_from_git_package(self, tmp_path):
        """Should extract commit hash from git URL format."""
        pm = PluginManager()

        resolved_file = tmp_path / "resolved.txt"
        resolved_file.write_text(
            "my-plugin @ git+https://github.com/user/repo@abc123def456\n"
        )

        version = pm._get_version_from_resolved("my-plugin", str(resolved_file))

        assert version == "abc123def456"

    def test_handles_hyphenated_package_names(self, tmp_path):
        """Should correctly match package names with multiple hyphens.

        This tests the fix for a regex bug where chained .replace() calls
        would corrupt the pattern. For example, 'scope-test-generator' would
        incorrectly become 'scope[-[-_]]test[-[-_]]generator' instead of
        'scope[-_]test[-_]generator'.
        """
        pm = PluginManager()

        resolved_file = tmp_path / "resolved.txt"
        resolved_file.write_text(
            "scope-test-generator @ git+https://github.com/user/repo@deadbeef123\n"
        )

        version = pm._get_version_from_resolved(
            "scope-test-generator", str(resolved_file)
        )

        assert version == "deadbeef123"

    def test_matches_underscore_variant_of_hyphenated_name(self, tmp_path):
        """Should match package with underscores when searching with hyphens.

        Python package names treat - and _ as equivalent, so searching for
        'my-package' should match 'my_package' in resolved.txt.
        """
        pm = PluginManager()

        resolved_file = tmp_path / "resolved.txt"
        resolved_file.write_text("my_package==2.0.0\n")

        version = pm._get_version_from_resolved("my-package", str(resolved_file))

        assert version == "2.0.0"

    def test_matches_hyphen_variant_of_underscored_name(self, tmp_path):
        """Should match package with hyphens when searching with underscores."""
        pm = PluginManager()

        resolved_file = tmp_path / "resolved.txt"
        resolved_file.write_text("my-package==3.0.0\n")

        version = pm._get_version_from_resolved("my_package", str(resolved_file))

        assert version == "3.0.0"

    def test_returns_none_for_missing_file(self, tmp_path):
        """Should return None if resolved file doesn't exist."""
        pm = PluginManager()

        version = pm._get_version_from_resolved(
            "any-package", str(tmp_path / "nonexistent.txt")
        )

        assert version is None

    def test_returns_none_for_missing_package(self, tmp_path):
        """Should return None if package not found in resolved file."""
        pm = PluginManager()

        resolved_file = tmp_path / "resolved.txt"
        resolved_file.write_text("other-package==1.0.0\n")

        version = pm._get_version_from_resolved("missing-package", str(resolved_file))

        assert version is None


class TestValidateInstall:
    """Tests for validate_install_async method."""

    def test_returns_valid_when_no_conflicts(self):
        """Return code 0 should mean is_valid=True."""
        pm = PluginManager()

        # Mock DependencyValidator to return valid
        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_result.error_message = None

        with patch(
            "scope.core.plugins.manager.DependencyValidator"
        ) as mock_validator_class:
            mock_validator = MagicMock()
            mock_validator.validate_install.return_value = mock_result
            mock_validator_class.return_value = mock_validator

            is_valid, error = pm._validate_install_sync(["test-package"])

        assert is_valid is True
        assert error is None

    def test_returns_invalid_with_error_message(self):
        """Return code != 0 should include error message."""
        pm = PluginManager()

        mock_result = MagicMock()
        mock_result.is_valid = False
        mock_result.error_message = "Dependency conflict"

        with patch(
            "scope.core.plugins.manager.DependencyValidator"
        ) as mock_validator_class:
            mock_validator = MagicMock()
            mock_validator.validate_install.return_value = mock_result
            mock_validator_class.return_value = mock_validator

            is_valid, error = pm._validate_install_sync(["conflicting-package"])

        assert is_valid is False
        assert error == "Dependency conflict"


class TestInstallPlugin:
    """Tests for install_plugin_async method."""

    def test_uses_compile_based_resolution(self):
        """Verify compile and sync are called for non-editable installs."""
        pm = PluginManager()

        with patch.object(pm, "_read_plugins_file", return_value=[]):
            with patch.object(pm, "_write_plugins_file"):
                with patch.object(
                    pm,
                    "_compile_plugins",
                    return_value=(True, "/tmp/resolved.txt", None),
                ) as mock_compile:
                    with patch.object(
                        pm, "_sync_plugins", return_value=(True, None)
                    ) as mock_sync:
                        pm._install_plugin_sync("test-package")

        mock_compile.assert_called_once()
        mock_sync.assert_called_once_with("/tmp/resolved.txt")

    def test_handles_editable_install(self):
        """Verify --editable flag is included for editable installs."""
        pm = PluginManager()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            with patch.object(
                pm, "_get_package_name_from_path", return_value="test-package"
            ):
                pm._install_plugin_sync("/path/to/package", editable=True)

        args = mock_run.call_args[0][0]
        assert "--editable" in args

    def test_upgrade_uses_upgrade_package_flag(self):
        """Verify --upgrade-package is passed to compile when upgrading."""
        pm = PluginManager()

        with patch.object(pm, "_read_plugins_file", return_value=["test-package"]):
            with patch.object(pm, "_write_plugins_file"):
                with patch.object(
                    pm,
                    "_compile_plugins",
                    return_value=(True, "/tmp/resolved.txt", None),
                ) as mock_compile:
                    with patch.object(pm, "_sync_plugins", return_value=(True, None)):
                        pm._install_plugin_sync("test-package", upgrade=True)

        # Check that upgrade_package was passed to _compile_plugins
        mock_compile.assert_called_once_with(upgrade_package="test-package")

    def test_raises_on_dependency_error(self):
        """PluginDependencyError should be raised on compile fail."""
        pm = PluginManager()

        with patch.object(pm, "_read_plugins_file", return_value=[]):
            with patch.object(pm, "_write_plugins_file"):
                with patch.object(
                    pm, "_compile_plugins", return_value=(False, "", "Conflict error")
                ):
                    with pytest.raises(PluginDependencyError):
                        pm._install_plugin_sync("conflicting-package")

    def test_raises_on_install_error(self):
        """PluginInstallError should be raised on sync fail."""
        pm = PluginManager()

        with patch.object(pm, "_read_plugins_file", return_value=[]):
            with patch.object(pm, "_write_plugins_file"):
                with patch.object(
                    pm,
                    "_compile_plugins",
                    return_value=(True, "/tmp/resolved.txt", None),
                ):
                    with patch.object(
                        pm, "_sync_plugins", return_value=(False, "Install failed")
                    ):
                        with pytest.raises(PluginInstallError):
                            pm._install_plugin_sync("bad-package")

    def test_rollback_plugins_file_on_compile_failure(self):
        """Plugins.txt should be rolled back if compile fails."""
        pm = PluginManager()

        original_plugins = ["existing-package"]
        with patch.object(
            pm, "_read_plugins_file", return_value=original_plugins.copy()
        ):
            with patch.object(pm, "_write_plugins_file") as mock_write:
                with patch.object(
                    pm, "_compile_plugins", return_value=(False, "", "Conflict")
                ):
                    with pytest.raises(PluginDependencyError):
                        pm._install_plugin_sync("new-package")

        # Should have been called twice: once to add, once to rollback
        assert mock_write.call_count == 2
        # Last call should be the rollback with original plugins
        mock_write.assert_called_with(original_plugins)


class TestUninstallPlugin:
    """Tests for uninstall_plugin_async method."""

    def test_runs_uv_pip_uninstall(self):
        """Verify subprocess command is correct."""
        pm = PluginManager()

        # Mock list_plugins to find the plugin
        with patch.object(
            pm,
            "_list_plugins_sync",
            return_value=[{"name": "test-plugin", "pipelines": []}],
        ):
            with patch.object(pm, "_read_plugins_file", return_value=[]):
                with patch.object(pm, "_write_plugins_file"):
                    with patch.object(
                        pm, "_compile_plugins", return_value=(True, "", None)
                    ):
                        with patch("subprocess.run") as mock_run:
                            mock_run.return_value = MagicMock(
                                returncode=0, stdout="", stderr=""
                            )
                            pm._uninstall_plugin_sync("test-plugin")

        # Find the uv pip uninstall call among all subprocess.run calls
        # (other modules like pipeline registry may also call subprocess.run)
        uv_uninstall_call = None
        for call in mock_run.call_args_list:
            args = call[0][0] if call[0] else call[1].get("args", [])
            if isinstance(args, list) and "uv" in args and "uninstall" in args:
                uv_uninstall_call = args
                break

        assert uv_uninstall_call is not None, "uv pip uninstall was not called"
        assert "uv" in uv_uninstall_call
        assert "pip" in uv_uninstall_call
        assert "uninstall" in uv_uninstall_call
        assert "test-plugin" in uv_uninstall_call

    def test_raises_plugin_not_found(self):
        """Unknown plugin should raise PluginNotFoundError."""
        pm = PluginManager()

        with patch.object(pm, "_list_plugins_sync", return_value=[]):
            with pytest.raises(PluginNotFoundError):
                pm._uninstall_plugin_sync("nonexistent-plugin")

    def test_unregisters_pipelines(self):
        """Verify PipelineRegistry.unregister is called."""
        pm = PluginManager()

        with patch.object(
            pm,
            "_list_plugins_sync",
            return_value=[
                {
                    "name": "test-plugin",
                    "pipelines": [{"pipeline_id": "test-pipeline"}],
                }
            ],
        ):
            with patch.object(pm, "_read_plugins_file", return_value=[]):
                with patch.object(pm, "_write_plugins_file"):
                    with patch.object(
                        pm, "_compile_plugins", return_value=(True, "", None)
                    ):
                        with patch("subprocess.run") as mock_run:
                            mock_run.return_value = MagicMock(returncode=0)
                            with patch(
                                "scope.core.pipelines.registry.PipelineRegistry.unregister"
                            ) as mock_unregister:
                                pm._uninstall_plugin_sync("test-plugin")

                                mock_unregister.assert_called_once_with("test-pipeline")

    def test_removes_from_plugins_file(self):
        """Verify plugin is removed from plugins.txt."""
        pm = PluginManager()

        with patch.object(
            pm,
            "_list_plugins_sync",
            return_value=[{"name": "test-plugin", "pipelines": []}],
        ):
            with patch.object(
                pm, "_read_plugins_file", return_value=["test-plugin", "other-plugin"]
            ):
                with patch.object(pm, "_write_plugins_file") as mock_write:
                    with patch.object(
                        pm, "_compile_plugins", return_value=(True, "", None)
                    ):
                        with patch("subprocess.run") as mock_run:
                            mock_run.return_value = MagicMock(returncode=0)
                            pm._uninstall_plugin_sync("test-plugin")

        # Should write plugins file without test-plugin
        mock_write.assert_called_once_with(["other-plugin"])

    def test_recompiles_after_uninstall(self):
        """Verify compile is called after removing from plugins.txt."""
        pm = PluginManager()

        with patch.object(
            pm,
            "_list_plugins_sync",
            return_value=[{"name": "test-plugin", "pipelines": []}],
        ):
            with patch.object(pm, "_read_plugins_file", return_value=["test-plugin"]):
                with patch.object(pm, "_write_plugins_file"):
                    with patch.object(
                        pm, "_compile_plugins", return_value=(True, "", None)
                    ) as mock_compile:
                        with patch("subprocess.run") as mock_run:
                            mock_run.return_value = MagicMock(returncode=0)
                            pm._uninstall_plugin_sync("test-plugin")

        mock_compile.assert_called_once()


class TestReloadPlugin:
    """Tests for reload_plugin_async method."""

    def test_raises_not_found_for_unknown_plugin(self):
        """PluginNotFoundError should be raised for unknown plugin."""
        pm = PluginManager()

        with patch.object(pm, "_list_plugins_sync", return_value=[]):
            with pytest.raises(PluginNotFoundError):
                pm._reload_plugin_sync("nonexistent-plugin")

    def test_raises_not_editable_for_non_editable(self):
        """PluginNotEditableError should be raised for non-editable plugin."""
        pm = PluginManager()

        with patch.object(
            pm,
            "_list_plugins_sync",
            return_value=[{"name": "test-plugin", "editable": False, "pipelines": []}],
        ):
            with pytest.raises(PluginNotEditableError):
                pm._reload_plugin_sync("test-plugin")

    def test_raises_in_use_without_force(self):
        """PluginInUseError should be raised with loaded pipelines."""
        pm = PluginManager()

        mock_pipeline_manager = MagicMock()
        # get_pipeline_by_id returns something (meaning pipeline is loaded)
        mock_pipeline_manager.get_pipeline_by_id.return_value = MagicMock()

        with patch.object(
            pm,
            "_list_plugins_sync",
            return_value=[
                {
                    "name": "test-plugin",
                    "editable": True,
                    "editable_path": "/path/to/plugin",
                    "pipelines": [{"pipeline_id": "test-pipeline"}],
                }
            ],
        ):
            with pytest.raises(PluginInUseError) as exc_info:
                pm._reload_plugin_sync(
                    "test-plugin", force=False, pipeline_manager=mock_pipeline_manager
                )

            assert "test-pipeline" in exc_info.value.loaded_pipelines

    def test_unloads_pipelines_with_force(self):
        """force=True should unload pipelines."""
        pm = PluginManager()

        mock_pipeline_manager = MagicMock()
        mock_pipeline_manager.get_pipeline_by_id.return_value = MagicMock()

        with patch.object(
            pm,
            "_list_plugins_sync",
            return_value=[
                {
                    "name": "test-plugin",
                    "editable": True,
                    "editable_path": "/path/to/plugin",
                    "pipelines": [{"pipeline_id": "test-pipeline"}],
                }
            ],
        ):
            with patch("scope.core.pipelines.registry.PipelineRegistry.unregister"):
                with patch.object(pm, "_reload_module_tree"):
                    with patch.object(pm._pm, "unregister"):
                        with patch.object(pm._pm, "load_setuptools_entrypoints"):
                            with patch.object(pm, "register_plugin_pipelines"):
                                pm._reload_plugin_sync(
                                    "test-plugin",
                                    force=True,
                                    pipeline_manager=mock_pipeline_manager,
                                )

        mock_pipeline_manager.unload_pipeline_by_id.assert_called_with("test-pipeline")

    def test_returns_pipeline_diff(self):
        """Correct added/removed/reloaded lists should be returned."""
        pm = PluginManager()

        call_count = [0]

        def mock_list_plugins():
            call_count[0] += 1
            # First call: initial plugin info lookup (before reload)
            if call_count[0] == 1:
                return [
                    {
                        "name": "test-plugin",
                        "editable": True,
                        "editable_path": "/path/to/plugin",
                        "pipelines": [
                            {"pipeline_id": "old-pipeline"},
                            {"pipeline_id": "unchanged-pipeline"},
                        ],
                    }
                ]
            # Second call: after reload, new pipeline info
            else:
                return [
                    {
                        "name": "test-plugin",
                        "editable": True,
                        "editable_path": "/path/to/plugin",
                        "pipelines": [
                            {"pipeline_id": "unchanged-pipeline"},
                            {"pipeline_id": "new-pipeline"},
                        ],
                    }
                ]

        with patch.object(pm, "_list_plugins_sync", side_effect=mock_list_plugins):
            with patch("scope.core.pipelines.registry.PipelineRegistry.unregister"):
                with patch.object(pm, "_reload_module_tree"):
                    with patch.object(pm._pm, "unregister"):
                        with patch.object(pm._pm, "load_setuptools_entrypoints"):
                            with patch.object(pm, "register_plugin_pipelines"):
                                result = pm._reload_plugin_sync("test-plugin")

        assert "unchanged-pipeline" in result["reloaded_pipelines"]
        assert "new-pipeline" in result["added_pipelines"]
        assert "old-pipeline" in result["removed_pipelines"]


class TestGetPluginForPipeline:
    """Tests for get_plugin_for_pipeline method."""

    def test_returns_plugin_name(self):
        """Known pipeline should return plugin name."""
        pm = PluginManager()
        pm._pipeline_to_plugin["test-pipeline"] = "test-plugin"

        result = pm.get_plugin_for_pipeline("test-pipeline")

        assert result == "test-plugin"

    def test_returns_none_for_unknown(self):
        """Unknown pipeline should return None."""
        pm = PluginManager()

        result = pm.get_plugin_for_pipeline("unknown-pipeline")

        assert result is None


class TestVenvRollback:
    """Tests for venv rollback during plugin installation."""

    def test_snapshot_capture_called_before_install(self):
        """Verify VenvSnapshot.capture() is called before installation."""
        pm = PluginManager()

        with patch.object(pm, "_read_plugins_file", return_value=[]):
            with patch.object(pm, "_write_plugins_file"):
                with patch.object(
                    pm,
                    "_compile_plugins",
                    return_value=(True, "/tmp/resolved.txt", None),
                ):
                    with patch.object(pm, "_sync_plugins", return_value=(True, None)):
                        with patch(
                            "scope.core.plugins.venv_snapshot.VenvSnapshot"
                        ) as mock_snapshot_class:
                            mock_snapshot = MagicMock()
                            mock_snapshot.capture.return_value = True
                            mock_snapshot_class.return_value = mock_snapshot

                            pm._install_plugin_sync("test-package")

                            mock_snapshot.capture.assert_called_once()

    def test_snapshot_discard_on_success(self):
        """Verify VenvSnapshot.discard() is called on successful install."""
        pm = PluginManager()

        with patch.object(pm, "_read_plugins_file", return_value=[]):
            with patch.object(pm, "_write_plugins_file"):
                with patch.object(
                    pm,
                    "_compile_plugins",
                    return_value=(True, "/tmp/resolved.txt", None),
                ):
                    with patch.object(pm, "_sync_plugins", return_value=(True, None)):
                        with patch(
                            "scope.core.plugins.venv_snapshot.VenvSnapshot"
                        ) as mock_snapshot_class:
                            mock_snapshot = MagicMock()
                            mock_snapshot.capture.return_value = True
                            mock_snapshot_class.return_value = mock_snapshot

                            pm._install_plugin_sync("test-package")

                            mock_snapshot.discard.assert_called_once()

    def test_snapshot_discard_on_compile_failure(self):
        """Verify VenvSnapshot.discard() is called on compile failure."""
        pm = PluginManager()

        with patch.object(pm, "_read_plugins_file", return_value=[]):
            with patch.object(pm, "_write_plugins_file"):
                with patch.object(
                    pm, "_compile_plugins", return_value=(False, "", "Compile error")
                ):
                    with patch(
                        "scope.core.plugins.venv_snapshot.VenvSnapshot"
                    ) as mock_snapshot_class:
                        mock_snapshot = MagicMock()
                        mock_snapshot.capture.return_value = True
                        mock_snapshot_class.return_value = mock_snapshot

                        with pytest.raises(PluginDependencyError):
                            pm._install_plugin_sync("test-package")

                        # Snapshot should be discarded (not restored) since
                        # no packages were installed yet
                        mock_snapshot.discard.assert_called_once()
                        mock_snapshot.restore.assert_not_called()

    def test_snapshot_restore_on_sync_failure(self):
        """Verify VenvSnapshot.restore() is called on sync failure."""
        pm = PluginManager()

        with patch.object(pm, "_read_plugins_file", return_value=[]):
            with patch.object(pm, "_write_plugins_file"):
                with patch.object(
                    pm,
                    "_compile_plugins",
                    return_value=(True, "/tmp/resolved.txt", None),
                ):
                    with patch.object(
                        pm, "_sync_plugins", return_value=(False, "Sync error")
                    ):
                        with patch(
                            "scope.core.plugins.venv_snapshot.VenvSnapshot"
                        ) as mock_snapshot_class:
                            mock_snapshot = MagicMock()
                            mock_snapshot.capture.return_value = True
                            mock_snapshot.restore.return_value = (True, None)
                            mock_snapshot_class.return_value = mock_snapshot

                            with pytest.raises(PluginInstallError):
                                pm._install_plugin_sync("test-package")

                            # Snapshot should be restored since packages may have
                            # been partially installed
                            mock_snapshot.restore.assert_called_once()

    def test_rollback_plugins_file_and_venv_on_sync_failure(self):
        """Verify both plugins.txt and venv are rolled back on sync failure."""
        pm = PluginManager()

        original_plugins = ["existing-package"]
        with patch.object(
            pm, "_read_plugins_file", return_value=original_plugins.copy()
        ):
            with patch.object(pm, "_write_plugins_file") as mock_write:
                with patch.object(
                    pm,
                    "_compile_plugins",
                    return_value=(True, "/tmp/resolved.txt", None),
                ):
                    with patch.object(
                        pm, "_sync_plugins", return_value=(False, "Sync error")
                    ):
                        with patch(
                            "scope.core.plugins.venv_snapshot.VenvSnapshot"
                        ) as mock_snapshot_class:
                            mock_snapshot = MagicMock()
                            mock_snapshot.capture.return_value = True
                            mock_snapshot.restore.return_value = (True, None)
                            mock_snapshot_class.return_value = mock_snapshot

                            with pytest.raises(PluginInstallError):
                                pm._install_plugin_sync("new-package")

        # Should have been called twice: once to add, once to rollback
        assert mock_write.call_count == 2
        # Last call should be the rollback with original plugins
        mock_write.assert_called_with(original_plugins)

    def test_logs_warning_on_restore_failure(self):
        """Verify warning is logged if venv restore fails."""
        pm = PluginManager()

        with patch.object(pm, "_read_plugins_file", return_value=[]):
            with patch.object(pm, "_write_plugins_file"):
                with patch.object(
                    pm,
                    "_compile_plugins",
                    return_value=(True, "/tmp/resolved.txt", None),
                ):
                    with patch.object(
                        pm, "_sync_plugins", return_value=(False, "Sync error")
                    ):
                        with patch(
                            "scope.core.plugins.venv_snapshot.VenvSnapshot"
                        ) as mock_snapshot_class:
                            mock_snapshot = MagicMock()
                            mock_snapshot.capture.return_value = True
                            mock_snapshot.restore.return_value = (
                                False,
                                "Restore failed",
                            )
                            mock_snapshot_class.return_value = mock_snapshot

                            with patch(
                                "scope.core.plugins.manager.logger"
                            ) as mock_logger:
                                with pytest.raises(PluginInstallError):
                                    pm._install_plugin_sync("test-package")

                                # Should log error about failed restore
                                mock_logger.error.assert_called()
                                error_call = str(mock_logger.error.call_args)
                                assert "rollback" in error_call.lower()

    def test_proceeds_without_rollback_if_capture_fails(self):
        """Verify installation proceeds if snapshot capture fails."""
        pm = PluginManager()

        with patch.object(pm, "_read_plugins_file", return_value=[]):
            with patch.object(pm, "_write_plugins_file"):
                with patch.object(
                    pm,
                    "_compile_plugins",
                    return_value=(True, "/tmp/resolved.txt", None),
                ):
                    with patch.object(pm, "_sync_plugins", return_value=(True, None)):
                        with patch(
                            "scope.core.plugins.venv_snapshot.VenvSnapshot"
                        ) as mock_snapshot_class:
                            mock_snapshot = MagicMock()
                            mock_snapshot.capture.return_value = False  # Capture fails
                            mock_snapshot_class.return_value = mock_snapshot

                            # Should not raise, installation should proceed
                            result = pm._install_plugin_sync("test-package")

                            assert result["success"] is True

    def test_no_restore_if_capture_failed_and_sync_fails(self):
        """Verify restore is not called if capture failed and sync fails."""
        pm = PluginManager()

        with patch.object(pm, "_read_plugins_file", return_value=[]):
            with patch.object(pm, "_write_plugins_file"):
                with patch.object(
                    pm,
                    "_compile_plugins",
                    return_value=(True, "/tmp/resolved.txt", None),
                ):
                    with patch.object(
                        pm, "_sync_plugins", return_value=(False, "Sync error")
                    ):
                        with patch(
                            "scope.core.plugins.venv_snapshot.VenvSnapshot"
                        ) as mock_snapshot_class:
                            mock_snapshot = MagicMock()
                            mock_snapshot.capture.return_value = False  # Capture fails
                            mock_snapshot_class.return_value = mock_snapshot

                            with pytest.raises(PluginInstallError):
                                pm._install_plugin_sync("test-package")

                            # Restore should not be called since capture failed
                            mock_snapshot.restore.assert_not_called()
                            # But discard should still be called for cleanup
                            mock_snapshot.discard.assert_called_once()


class TestEditableVenvRollback:
    """Tests for venv rollback during editable plugin installs."""

    def test_snapshot_capture_called_for_editable(self):
        """Verify snapshot is captured before editable install."""
        pm = PluginManager()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            with patch.object(
                pm, "_get_package_name_from_path", return_value="test-package"
            ):
                with patch(
                    "scope.core.plugins.venv_snapshot.VenvSnapshot"
                ) as MockSnapshot:
                    mock_instance = MockSnapshot.return_value
                    mock_instance.capture.return_value = True

                    pm._install_editable_plugin("/path/to/plugin")

                    mock_instance.capture.assert_called_once()
                    mock_instance.discard.assert_called_once()

    def test_snapshot_restore_on_editable_failure(self):
        """Verify snapshot restore is called when editable install fails."""
        pm = PluginManager()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="Build failed"
            )
            with patch("scope.core.plugins.venv_snapshot.VenvSnapshot") as MockSnapshot:
                mock_instance = MockSnapshot.return_value
                mock_instance.capture.return_value = True
                mock_instance.restore.return_value = (True, None)

                with pytest.raises(PluginInstallError):
                    pm._install_editable_plugin("/path/to/plugin")

                mock_instance.restore.assert_called_once()
                mock_instance.discard.assert_called_once()

    def test_snapshot_discard_on_editable_success(self):
        """Verify snapshot discard is called on successful editable install."""
        pm = PluginManager()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            with patch.object(
                pm, "_get_package_name_from_path", return_value="test-package"
            ):
                with patch(
                    "scope.core.plugins.venv_snapshot.VenvSnapshot"
                ) as MockSnapshot:
                    mock_instance = MockSnapshot.return_value
                    mock_instance.capture.return_value = True

                    pm._install_editable_plugin("/path/to/plugin")

                    mock_instance.discard.assert_called_once()
                    mock_instance.restore.assert_not_called()

    def test_editable_proceeds_without_rollback_if_capture_fails(self):
        """Verify editable installation proceeds if snapshot capture fails."""
        pm = PluginManager()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            with patch.object(
                pm, "_get_package_name_from_path", return_value="test-package"
            ):
                with patch(
                    "scope.core.plugins.venv_snapshot.VenvSnapshot"
                ) as MockSnapshot:
                    mock_instance = MockSnapshot.return_value
                    mock_instance.capture.return_value = False  # Capture fails

                    # Should not raise, installation should proceed
                    result = pm._install_editable_plugin("/path/to/plugin")

                    assert result["success"] is True

    def test_editable_no_restore_if_capture_failed_and_install_fails(self):
        """Verify restore is not called if capture failed and editable install fails."""
        pm = PluginManager()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="Install error"
            )
            with patch("scope.core.plugins.venv_snapshot.VenvSnapshot") as MockSnapshot:
                mock_instance = MockSnapshot.return_value
                mock_instance.capture.return_value = False  # Capture fails

                with pytest.raises(PluginInstallError):
                    pm._install_editable_plugin("/path/to/plugin")

                # Restore should not be called since capture failed
                mock_instance.restore.assert_not_called()
                # But discard should still be called for cleanup
                mock_instance.discard.assert_called_once()

    def test_editable_logs_warning_on_restore_failure(self):
        """Verify warning is logged if venv restore fails during editable install."""
        pm = PluginManager()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="Install error"
            )
            with patch("scope.core.plugins.venv_snapshot.VenvSnapshot") as MockSnapshot:
                mock_instance = MockSnapshot.return_value
                mock_instance.capture.return_value = True
                mock_instance.restore.return_value = (False, "Restore failed")

                with patch("scope.core.plugins.manager.logger") as mock_logger:
                    with pytest.raises(PluginInstallError):
                        pm._install_editable_plugin("/path/to/plugin")

                    # Should log error about failed restore
                    mock_logger.error.assert_called()
                    error_call = str(mock_logger.error.call_args)
                    assert "rollback" in error_call.lower()


class TestListPluginsDeduplication:
    """Tests for plugin deduplication in _list_plugins_sync."""

    def _make_mock_dist(self, name, source="pypi", editable=False):
        """Helper to create a mock distribution with scope entry points."""
        mock_ep = MagicMock()
        mock_ep.group = "scope"
        mock_ep.name = name

        mock_dist = MagicMock()
        mock_dist.entry_points = [mock_ep]
        mock_dist.metadata = {
            "Name": name,
            "Version": "1.0.0",
            "Author": "Test",
            "Summary": "A test plugin",
        }
        mock_dist._path = None  # Default to no direct_url.json (PyPI)
        return mock_dist

    def test_deduplicates_plugins_by_name(self):
        """Two distributions with same name should return one plugin."""
        pm = PluginManager()

        dist1 = self._make_mock_dist("test-plugin")
        dist2 = self._make_mock_dist("test-plugin")

        with patch("importlib.metadata.distributions", return_value=[dist1, dist2]):
            plugins = pm._list_plugins_sync()

        assert len(plugins) == 1
        assert plugins[0]["name"] == "test-plugin"

    def test_dedup_prefers_editable_over_non_editable(self):
        """When duplicate exists, editable version should win."""
        pm = PluginManager()

        dist_git = self._make_mock_dist("test-plugin")
        dist_editable = self._make_mock_dist("test-plugin")

        # Make dist_git return git source
        def git_source(dist):
            if dist is dist_git:
                return ("git", False, None, "https://github.com/user/repo.git")
            return ("local", True, "/path/to/plugin", None)

        with patch(
            "importlib.metadata.distributions", return_value=[dist_git, dist_editable]
        ):
            with patch.object(pm, "_get_plugin_source", side_effect=git_source):
                plugins = pm._list_plugins_sync()

        assert len(plugins) == 1
        assert plugins[0]["editable"] is True
        assert plugins[0]["source"] == "local"


class TestEditableInstallPluginsTxtCleanup:
    """Tests for plugins.txt cleanup during editable installs."""

    def test_editable_install_cleans_plugins_txt(self):
        """Editable install should remove existing entry from plugins.txt."""
        pm = PluginManager()

        with patch.object(
            pm,
            "_read_plugins_file",
            return_value=["git+https://github.com/user/test-plugin.git"],
        ):
            with patch.object(pm, "_write_plugins_file") as mock_write:
                with patch.object(
                    pm, "_get_package_name_from_path", return_value="test-plugin"
                ):
                    with patch("subprocess.run") as mock_run:
                        mock_run.return_value = MagicMock(
                            returncode=0, stdout="", stderr=""
                        )
                        pm._install_editable_plugin("/path/to/test-plugin")

        # Should have written plugins.txt with the git entry removed
        mock_write.assert_called_once_with([])

    def test_editable_install_noop_if_not_in_plugins_txt(self):
        """No write should happen if plugin not in plugins.txt."""
        pm = PluginManager()

        with patch.object(pm, "_read_plugins_file", return_value=[]):
            with patch.object(pm, "_write_plugins_file") as mock_write:
                with patch.object(
                    pm, "_get_package_name_from_path", return_value="test-plugin"
                ):
                    with patch("subprocess.run") as mock_run:
                        mock_run.return_value = MagicMock(
                            returncode=0, stdout="", stderr=""
                        )
                        pm._install_editable_plugin("/path/to/test-plugin")

        # _write_plugins_file should NOT have been called
        mock_write.assert_not_called()


class TestLoadPluginsErrorHandling:
    """Tests for resilient plugin loading with malformed entry points."""

    def _make_mock_dist(self, ep_name, ep_group="scope", package_name="test-plugin"):
        """Create a mock distribution with a single entry point."""
        mock_ep = MagicMock()
        mock_ep.group = ep_group
        mock_ep.name = ep_name
        mock_ep.load = MagicMock(return_value=MagicMock())

        mock_dist = MagicMock()
        mock_dist.entry_points = [mock_ep]
        mock_dist.metadata = {"Name": package_name}
        return mock_dist, mock_ep

    def test_malformed_entry_point_does_not_crash(self):
        """One bad entry point should not prevent good plugins from loading."""
        pm = PluginManager()

        # Good plugin
        good_dist, good_ep = self._make_mock_dist(
            "good-plugin", package_name="good-pkg"
        )
        good_ep.load.return_value = MagicMock()

        # Bad plugin — load raises ModuleNotFoundError
        bad_dist, bad_ep = self._make_mock_dist("bad-plugin", package_name="bad-pkg")
        bad_ep.load.side_effect = ModuleNotFoundError("No module named 'MIT'")

        with patch(
            "importlib.metadata.distributions", return_value=[good_dist, bad_dist]
        ):
            with patch.object(pm._pm, "load_setuptools_entrypoints"):
                pm.load_plugins()

        # Bad plugin recorded in _failed_plugins
        assert len(pm._failed_plugins) == 1
        failed = pm._failed_plugins[0]
        assert failed.package_name == "bad-pkg"
        assert failed.entry_point_name == "bad-plugin"
        assert failed.error_type == "ModuleNotFoundError"
        assert "MIT" in failed.error_message

    def test_all_plugins_fail_gracefully(self):
        """All entry points failing should not raise an exception."""
        pm = PluginManager()

        dist1, ep1 = self._make_mock_dist("plugin-a", package_name="pkg-a")
        ep1.load.side_effect = ImportError("missing dep")

        dist2, ep2 = self._make_mock_dist("plugin-b", package_name="pkg-b")
        ep2.load.side_effect = ModuleNotFoundError("No module named 'xyz'")

        with patch("importlib.metadata.distributions", return_value=[dist1, dist2]):
            with patch.object(pm._pm, "load_setuptools_entrypoints"):
                # Should not raise
                pm.load_plugins()

        assert len(pm._failed_plugins) == 2
        names = {f.package_name for f in pm._failed_plugins}
        assert names == {"pkg-a", "pkg-b"}

    def test_failed_entry_point_is_blocked(self):
        """After a failed load, the entry point should be blocked in pluggy."""
        pm = PluginManager()

        dist, ep = self._make_mock_dist("broken-ep", package_name="broken-pkg")
        ep.load.side_effect = ModuleNotFoundError("bad module")

        with patch("importlib.metadata.distributions", return_value=[dist]):
            with patch.object(pm._pm, "load_setuptools_entrypoints"):
                pm.load_plugins()

        assert pm._pm.is_blocked("broken-ep")

    def test_multiple_entry_points_rejected(self):
        """A package with != 1 scope entry points should be rejected immediately."""
        pm = PluginManager()

        # Package with two entry points: one bogus (license metadata), one real
        bogus_ep = MagicMock()
        bogus_ep.group = "scope"
        bogus_ep.name = "license"

        real_ep = MagicMock()
        real_ep.group = "scope"
        real_ep.name = "real-plugin"

        mock_dist = MagicMock()
        mock_dist.entry_points = [bogus_ep, real_ep]
        mock_dist.metadata = {"Name": "broken-pkg"}

        with patch("importlib.metadata.distributions", return_value=[mock_dist]):
            with patch.object(pm._pm, "load_setuptools_entrypoints"):
                pm.load_plugins()

        # Both entry points should be blocked
        assert pm._pm.is_blocked("license")
        assert pm._pm.is_blocked("real-plugin")

        # Neither ep.load() should have been called
        bogus_ep.load.assert_not_called()
        real_ep.load.assert_not_called()

        # Failure recorded with InvalidPluginError
        assert len(pm._failed_plugins) == 1
        failed = pm._failed_plugins[0]
        assert failed.package_name == "broken-pkg"
        assert failed.error_type == "InvalidPluginError"
        assert "Expected 1 entry point" in failed.error_message

    def test_load_plugins_clears_previous_failures(self):
        """Calling load_plugins() again should clear previous failures."""
        pm = PluginManager()

        dist, ep = self._make_mock_dist("flaky-ep", package_name="flaky-pkg")
        ep.load.side_effect = ImportError("first failure")

        with patch("importlib.metadata.distributions", return_value=[dist]):
            with patch.object(pm._pm, "load_setuptools_entrypoints"):
                pm.load_plugins()

        assert len(pm._failed_plugins) == 1

        # Second call with no broken plugins — failures should be cleared
        with patch("importlib.metadata.distributions", return_value=[]):
            with patch.object(pm._pm, "load_setuptools_entrypoints"):
                pm.load_plugins()

        assert len(pm._failed_plugins) == 0

    def test_get_failed_plugins_returns_copy(self):
        """get_failed_plugins() should return a copy, not the internal list."""
        pm = PluginManager()

        dist, ep = self._make_mock_dist("fail-ep", package_name="fail-pkg")
        ep.load.side_effect = ImportError("oops")

        with patch("importlib.metadata.distributions", return_value=[dist]):
            with patch.object(pm._pm, "load_setuptools_entrypoints"):
                pm.load_plugins()

        result = pm.get_failed_plugins()
        assert len(result) == 1
        assert result is not pm._failed_plugins
        # Mutating the returned list should not affect internal state
        result.clear()
        assert len(pm._failed_plugins) == 1

    def test_failed_plugin_pipelines_not_registered(self):
        """Pipelines from failed plugins should not appear in the registry."""
        pm = PluginManager()

        # Good plugin
        good_dist, good_ep = self._make_mock_dist(
            "good-plugin", package_name="good-pkg"
        )
        good_module = MagicMock()

        def good_register(register):
            mock_cls = MagicMock()
            mock_cls.get_config_class.return_value.pipeline_id = "good-pipeline"
            register(mock_cls)

        good_module.register_pipelines = good_register
        good_ep.load.return_value = good_module

        # Bad plugin — entry point load fails during prevalidation
        bad_dist, bad_ep = self._make_mock_dist("bad-plugin", package_name="bad-pkg")
        bad_ep.load.side_effect = ImportError("missing dependency")

        with patch(
            "importlib.metadata.distributions",
            return_value=[good_dist, bad_dist],
        ):
            with patch.object(pm._pm, "load_setuptools_entrypoints"):
                pm.load_plugins()

            # Set up a mock registry
            mock_registry = MagicMock()
            mock_registry.list_pipelines.return_value = ["good-pipeline"]

            pm.register_plugin_pipelines(mock_registry)

        # Bad package should NOT be in _registered_plugins
        assert "bad-pkg" not in pm._registered_plugins
        assert "good-pkg" in pm._registered_plugins

        # Bad package's pipeline should NOT be in the mapping
        assert "bad-pkg" not in pm._pipeline_to_plugin.values()


class TestIsPackageInstalled:
    """Tests for _is_package_installed method."""

    def test_returns_true_for_installed_package(self):
        """Should return True for a package that exists."""
        pm = PluginManager()

        mock_dist = MagicMock()
        with patch("importlib.metadata.distribution", return_value=mock_dist):
            assert pm._is_package_installed("some-package") is True

    def test_returns_false_for_missing_package(self):
        """Should return False for a package that doesn't exist."""
        from importlib.metadata import PackageNotFoundError

        pm = PluginManager()

        with patch(
            "importlib.metadata.distribution",
            side_effect=PackageNotFoundError("not found"),
        ):
            assert pm._is_package_installed("missing-package") is False

    def test_normalizes_hyphens_and_underscores(self):
        """Should find package regardless of hyphen/underscore in name."""
        from importlib.metadata import PackageNotFoundError

        pm = PluginManager()

        # First call (underscore variant) fails, second (hyphen variant) succeeds
        def side_effect(name):
            if name == "my_plugin":
                raise PackageNotFoundError("not found")
            return MagicMock()  # "my-plugin" succeeds

        with patch("importlib.metadata.distribution", side_effect=side_effect):
            assert pm._is_package_installed("my_plugin") is True

    def test_both_variants_missing(self):
        """Should return False when neither hyphen nor underscore variant exists."""
        from importlib.metadata import PackageNotFoundError

        pm = PluginManager()

        with patch(
            "importlib.metadata.distribution",
            side_effect=PackageNotFoundError("not found"),
        ):
            assert pm._is_package_installed("totally-missing") is False


class TestEnsurePluginsInstalled:
    """Tests for ensure_plugins_installed method."""

    def test_noop_when_no_plugins_file(self):
        """Should return immediately when plugins.txt is empty."""
        pm = PluginManager()

        with patch.object(pm, "_read_plugins_file", return_value=[]):
            with patch.object(pm, "_is_package_installed") as mock_check:
                pm.ensure_plugins_installed()

        mock_check.assert_not_called()

    def test_noop_when_all_installed(self):
        """Should return without calling sync when all plugins are present."""
        pm = PluginManager()

        with patch.object(
            pm, "_read_plugins_file", return_value=["plugin-a", "plugin-b"]
        ):
            with patch.object(pm, "_is_package_installed", return_value=True):
                with patch.object(pm, "_sync_plugins") as mock_sync:
                    pm.ensure_plugins_installed()

        mock_sync.assert_not_called()

    def test_syncs_from_compile_when_missing(self):
        """Should call _compile_plugins then _sync_plugins when plugins are missing."""
        pm = PluginManager()

        with patch.object(pm, "_read_plugins_file", return_value=["plugin-a"]):
            with patch.object(pm, "_is_package_installed", return_value=False):
                with patch.object(
                    pm,
                    "_compile_plugins",
                    return_value=(True, "/tmp/resolved.txt", None),
                ) as mock_compile:
                    with patch.object(
                        pm, "_sync_plugins", return_value=(True, None)
                    ) as mock_sync:
                        pm.ensure_plugins_installed()

        mock_compile.assert_called_once()
        mock_sync.assert_called_once_with("/tmp/resolved.txt")

    def test_always_recompiles_even_when_resolved_exists(self, tmp_path):
        """Should always recompile, not use existing resolved.txt directly."""
        pm = PluginManager()

        resolved = tmp_path / "resolved.txt"
        resolved.write_text("plugin-a==1.0.0\n")

        with patch.object(pm, "_read_plugins_file", return_value=["plugin-a"]):
            with patch.object(pm, "_is_package_installed", return_value=False):
                with patch.object(
                    pm,
                    "_compile_plugins",
                    return_value=(True, "/tmp/new-resolved.txt", None),
                ) as mock_compile:
                    with patch.object(
                        pm, "_sync_plugins", return_value=(True, None)
                    ) as mock_sync:
                        pm.ensure_plugins_installed()

        mock_compile.assert_called_once()
        mock_sync.assert_called_once_with("/tmp/new-resolved.txt")

    def test_logs_error_on_compile_failure(self, tmp_path):
        """Should log error and return if compile fails."""
        pm = PluginManager()

        resolved = tmp_path / "resolved.txt"  # Does not exist

        with patch.object(pm, "_read_plugins_file", return_value=["plugin-a"]):
            with patch.object(pm, "_is_package_installed", return_value=False):
                with patch(
                    "scope.core.plugins.manager.get_resolved_file",
                    return_value=resolved,
                ):
                    with patch.object(
                        pm,
                        "_compile_plugins",
                        return_value=(False, "", "Resolution error"),
                    ):
                        with patch.object(pm, "_sync_plugins") as mock_sync:
                            with patch(
                                "scope.core.plugins.manager.logger"
                            ) as mock_logger:
                                pm.ensure_plugins_installed()

        mock_sync.assert_not_called()
        mock_logger.error.assert_called()

    def test_logs_error_on_sync_failure(self):
        """Should log error if sync fails."""
        pm = PluginManager()

        with patch.object(pm, "_read_plugins_file", return_value=["plugin-a"]):
            with patch.object(pm, "_is_package_installed", return_value=False):
                with patch.object(
                    pm,
                    "_compile_plugins",
                    return_value=(True, "/tmp/resolved.txt", None),
                ):
                    with patch.object(
                        pm, "_sync_plugins", return_value=(False, "Install failed")
                    ):
                        with patch("scope.core.plugins.manager.logger") as mock_logger:
                            pm.ensure_plugins_installed()

        mock_logger.error.assert_called()
        error_msg = str(mock_logger.error.call_args)
        assert "Install failed" in error_msg

    def test_only_syncs_when_some_missing(self):
        """Should sync when at least one plugin is missing, even if others are present."""
        pm = PluginManager()

        def is_installed(name):
            return name == "plugin-a"  # plugin-b is missing

        with patch.object(
            pm, "_read_plugins_file", return_value=["plugin-a", "plugin-b"]
        ):
            with patch.object(pm, "_is_package_installed", side_effect=is_installed):
                with patch.object(
                    pm,
                    "_compile_plugins",
                    return_value=(True, "/tmp/resolved.txt", None),
                ):
                    with patch.object(
                        pm, "_sync_plugins", return_value=(True, None)
                    ) as mock_sync:
                        pm.ensure_plugins_installed()

        mock_sync.assert_called_once()

    def test_extracts_name_from_specifier(self, tmp_path):
        """Should extract package name from version-pinned specifiers."""
        pm = PluginManager()

        resolved = tmp_path / "resolved.txt"
        resolved.write_text("my-plugin==1.0.0\n")

        with patch.object(pm, "_read_plugins_file", return_value=["my-plugin==1.0.0"]):
            with patch.object(pm, "_is_package_installed", return_value=True):
                with patch.object(pm, "_sync_plugins") as mock_sync:
                    pm.ensure_plugins_installed()

        mock_sync.assert_not_called()

    def test_extracts_name_from_git_specifier(self, tmp_path):
        """Should extract package name from git URL specifiers."""
        pm = PluginManager()

        resolved = tmp_path / "resolved.txt"
        resolved.write_text("my-repo==1.0.0\n")

        with patch.object(
            pm,
            "_read_plugins_file",
            return_value=["git+https://github.com/user/my-repo.git"],
        ):
            with patch.object(pm, "_is_package_installed", return_value=True):
                with patch.object(pm, "_sync_plugins") as mock_sync:
                    pm.ensure_plugins_installed()

        mock_sync.assert_not_called()


class TestGenerateConstraints:
    """Tests for _generate_constraints method."""

    LOCK_FIXTURE = """\
version = 1
requires-python = ">=3.12"

[[package]]
name = "test-project"
version = "0.1.0"
source = { editable = "." }
dependencies = [
    { name = "transformers" },
    { name = "safetensors" },
    { name = "torch" },
]

[[package]]
name = "transformers"
version = "4.57.5"

[[package]]
name = "safetensors"
version = "0.6.3"

[[package]]
name = "torch"
version = "2.9.1"
"""

    def test_returns_none_when_no_lock_file(self, tmp_path):
        """uv.lock doesn't exist in cwd -> returns None."""
        pm = PluginManager()

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            result = pm._generate_constraints()

        assert result is None

    def test_generates_floor_and_ceiling(self, tmp_path):
        """Should generate >=locked,<next_major constraints."""
        pm = PluginManager()

        (tmp_path / "uv.lock").write_text(self.LOCK_FIXTURE)

        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            with patch(
                "scope.core.plugins.manager.get_plugins_dir",
                return_value=plugins_dir,
            ):
                result = pm._generate_constraints()

        assert result is not None
        content = result.read_text()
        assert "transformers>=4.57.5,<5" in content
        assert "safetensors>=0.6.3,<1" in content
        assert "torch>=2.9.1,<3" in content

    def test_only_constrains_direct_dependencies(self, tmp_path):
        """Transitive deps in uv.lock should be skipped."""
        pm = PluginManager()

        lock = """\
version = 1

[[package]]
name = "test-project"
version = "0.1.0"
source = { editable = "." }
dependencies = [
    { name = "transformers" },
]

[[package]]
name = "transformers"
version = "4.57.5"

[[package]]
name = "tokenizers"
version = "0.21.1"
"""
        (tmp_path / "uv.lock").write_text(lock)

        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            with patch(
                "scope.core.plugins.manager.get_plugins_dir",
                return_value=plugins_dir,
            ):
                result = pm._generate_constraints()

        assert result is not None
        content = result.read_text()
        assert "transformers>=4.57.5,<5" in content
        assert "tokenizers" not in content

    def test_deduplicates_packages(self, tmp_path):
        """Same dep listed twice should only appear once in constraints."""
        pm = PluginManager()

        lock = """\
version = 1

[[package]]
name = "test-project"
version = "0.1.0"
source = { editable = "." }
dependencies = [
    { name = "transformers" },
    { name = "transformers" },
]

[[package]]
name = "transformers"
version = "4.57.5"
"""
        (tmp_path / "uv.lock").write_text(lock)

        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            with patch(
                "scope.core.plugins.manager.get_plugins_dir",
                return_value=plugins_dir,
            ):
                result = pm._generate_constraints()

        assert result is not None
        content = result.read_text()
        assert content.count("transformers") == 1

    def test_skips_packages_with_plus_in_version(self, tmp_path):
        """Locked version with + (e.g. 2.9.1+cu128) should be skipped."""
        pm = PluginManager()

        lock = """\
version = 1

[[package]]
name = "test-project"
version = "0.1.0"
source = { editable = "." }
dependencies = [
    { name = "torch" },
    { name = "safetensors" },
]

[[package]]
name = "torch"
version = "2.9.1+cu128"

[[package]]
name = "safetensors"
version = "0.6.3"
"""
        (tmp_path / "uv.lock").write_text(lock)

        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            with patch(
                "scope.core.plugins.manager.get_plugins_dir",
                return_value=plugins_dir,
            ):
                result = pm._generate_constraints()

        assert result is not None
        content = result.read_text()
        assert "torch" not in content
        assert "safetensors>=0.6.3,<1" in content

    def test_handles_marker_deps(self, tmp_path):
        """Deps with markers in uv.lock should still be constrained."""
        pm = PluginManager()

        lock = """\
version = 1

[[package]]
name = "test-project"
version = "0.1.0"
source = { editable = "." }
dependencies = [
    { name = "triton", marker = "sys_platform == 'linux'" },
    { name = "transformers" },
]

[[package]]
name = "triton"
version = "3.5.1"

[[package]]
name = "transformers"
version = "4.57.5"
"""
        (tmp_path / "uv.lock").write_text(lock)

        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            with patch(
                "scope.core.plugins.manager.get_plugins_dir",
                return_value=plugins_dir,
            ):
                result = pm._generate_constraints()

        assert result is not None
        content = result.read_text()
        assert "triton>=3.5.1,<4" in content
        assert "transformers>=4.57.5,<5" in content

    def test_returns_none_on_parse_error(self, tmp_path):
        """Invalid TOML in uv.lock -> returns None, doesn't raise."""
        pm = PluginManager()

        (tmp_path / "uv.lock").write_text("this is not valid { toml [")

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            result = pm._generate_constraints()

        assert result is None


class TestCompilePluginsConstraints:
    """Tests that _compile_plugins() passes the constraint flag."""

    def test_compile_includes_constraint_flag(self, tmp_path):
        """Mock _generate_constraints to return a path -> verify --constraint in args."""
        pm = PluginManager()

        constraints_path = tmp_path / "lock-constraints.txt"
        constraints_path.write_text("transformers==4.57.5\n")

        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]\nname = 'test'\n")

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            with patch.object(
                pm, "_generate_constraints", return_value=constraints_path
            ):
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(
                        returncode=0, stdout="", stderr=""
                    )
                    pm._compile_plugins()

        args = mock_run.call_args[0][0]
        assert "--constraint" in args
        assert str(constraints_path) in args

    def test_compile_works_without_constraints(self, tmp_path):
        """Mock _generate_constraints to return None -> verify no --constraint."""
        pm = PluginManager()

        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]\nname = 'test'\n")

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            with patch.object(pm, "_generate_constraints", return_value=None):
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(
                        returncode=0, stdout="", stderr=""
                    )
                    pm._compile_plugins()

        args = mock_run.call_args[0][0]
        assert "--constraint" not in args


class TestEnsurePluginsInstalledRecompile:
    """Tests for the always-recompile behavior in ensure_plugins_installed."""

    def test_recompiles_when_plugins_missing(self):
        """Plugins missing -> _compile_plugins is called before _sync_plugins."""
        pm = PluginManager()

        with patch.object(pm, "_read_plugins_file", return_value=["plugin-a"]):
            with patch.object(pm, "_is_package_installed", return_value=False):
                with patch.object(
                    pm,
                    "_compile_plugins",
                    return_value=(True, "/tmp/resolved.txt", None),
                ) as mock_compile:
                    with patch.object(
                        pm, "_sync_plugins", return_value=(True, None)
                    ) as mock_sync:
                        pm.ensure_plugins_installed()

        mock_compile.assert_called_once()
        mock_sync.assert_called_once_with("/tmp/resolved.txt")

    def test_falls_back_to_resolved_on_compile_failure(self, tmp_path):
        """_compile_plugins fails, resolved.txt exists -> _sync_plugins still called."""
        pm = PluginManager()

        resolved = tmp_path / "resolved.txt"
        resolved.write_text("plugin-a==1.0.0\n")

        with patch.object(pm, "_read_plugins_file", return_value=["plugin-a"]):
            with patch.object(pm, "_is_package_installed", return_value=False):
                with patch.object(
                    pm,
                    "_compile_plugins",
                    return_value=(False, "", "Resolution error"),
                ):
                    with patch(
                        "scope.core.plugins.manager.get_resolved_file",
                        return_value=resolved,
                    ):
                        with patch.object(
                            pm, "_sync_plugins", return_value=(True, None)
                        ) as mock_sync:
                            pm.ensure_plugins_installed()

        mock_sync.assert_called_once_with(str(resolved))

    def test_errors_when_compile_fails_and_no_resolved(self, tmp_path):
        """_compile_plugins fails, no resolved.txt -> logs error, _sync_plugins not called."""
        pm = PluginManager()

        resolved = tmp_path / "resolved.txt"  # Does not exist

        with patch.object(pm, "_read_plugins_file", return_value=["plugin-a"]):
            with patch.object(pm, "_is_package_installed", return_value=False):
                with patch.object(
                    pm,
                    "_compile_plugins",
                    return_value=(False, "", "Resolution error"),
                ):
                    with patch(
                        "scope.core.plugins.manager.get_resolved_file",
                        return_value=resolved,
                    ):
                        with patch.object(pm, "_sync_plugins") as mock_sync:
                            with patch(
                                "scope.core.plugins.manager.logger"
                            ) as mock_logger:
                                pm.ensure_plugins_installed()

        mock_sync.assert_not_called()
        mock_logger.error.assert_called()


class TestGetNameFromResolved:
    """Tests for _get_name_from_resolved method."""

    def test_finds_name_for_git_url(self, tmp_path):
        """resolved.txt has flashvsr @ git+url -> returns flashvsr."""
        pm = PluginManager()

        resolved = tmp_path / "resolved.txt"
        resolved.write_text(
            "flashvsr @ git+https://github.com/varshith15/FlashVSR-Pro@abc123\n"
        )

        with patch(
            "scope.core.plugins.manager.get_resolved_file", return_value=resolved
        ):
            result = pm._get_name_from_resolved(
                "git+https://github.com/varshith15/FlashVSR-Pro"
            )

        assert result == "flashvsr"

    def test_handles_git_suffix_mismatch(self, tmp_path):
        """URL has .git suffix but resolved.txt doesn't -> still matches."""
        pm = PluginManager()

        resolved = tmp_path / "resolved.txt"
        resolved.write_text(
            "flashvsr @ git+https://github.com/varshith15/FlashVSR-Pro@abc123\n"
        )

        with patch(
            "scope.core.plugins.manager.get_resolved_file", return_value=resolved
        ):
            result = pm._get_name_from_resolved(
                "git+https://github.com/varshith15/FlashVSR-Pro.git"
            )

        assert result == "flashvsr"

    def test_returns_none_when_no_resolved(self, tmp_path):
        """No resolved.txt -> returns None."""
        pm = PluginManager()

        with patch(
            "scope.core.plugins.manager.get_resolved_file",
            return_value=tmp_path / "nonexistent.txt",
        ):
            result = pm._get_name_from_resolved("git+https://github.com/user/repo")

        assert result is None

    def test_returns_none_when_url_not_found(self, tmp_path):
        """URL not in resolved.txt -> returns None."""
        pm = PluginManager()

        resolved = tmp_path / "resolved.txt"
        resolved.write_text("other-pkg @ git+https://github.com/user/other@abc\n")

        with patch(
            "scope.core.plugins.manager.get_resolved_file", return_value=resolved
        ):
            result = pm._get_name_from_resolved(
                "git+https://github.com/user/not-in-resolved"
            )

        assert result is None


class TestExtractPackageName:
    """Tests for _extract_package_name method."""

    def test_git_url_uses_resolved(self, tmp_path):
        """resolved.txt maps URL -> name, _extract_package_name returns that name."""
        pm = PluginManager()

        resolved = tmp_path / "resolved.txt"
        resolved.write_text(
            "flashvsr @ git+https://github.com/varshith15/FlashVSR-Pro@abc123\n"
        )

        with patch(
            "scope.core.plugins.manager.get_resolved_file", return_value=resolved
        ):
            result = pm._extract_package_name(
                "git+https://github.com/varshith15/FlashVSR-Pro"
            )

        assert result == "flashvsr"

    def test_git_url_falls_back_to_repo_name(self, tmp_path):
        """No resolved.txt -> returns repo name (existing behavior)."""
        pm = PluginManager()

        with patch(
            "scope.core.plugins.manager.get_resolved_file",
            return_value=tmp_path / "nonexistent.txt",
        ):
            result = pm._extract_package_name(
                "git+https://github.com/varshith15/FlashVSR-Pro.git"
            )

        assert result == "FlashVSR-Pro"

    def test_local_path_reads_pyproject(self, tmp_path):
        """Local dir with pyproject.toml -> returns project name."""
        pm = PluginManager()

        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "my-cool-plugin"\n')

        result = pm._extract_package_name(str(tmp_path))

        assert result == "my-cool-plugin"

    def test_local_path_uses_dir_basename(self, tmp_path):
        """No pyproject.toml -> returns dir basename."""
        pm = PluginManager()

        sub = tmp_path / "scope-circle-controller"
        sub.mkdir()

        result = pm._extract_package_name(str(sub))

        assert result == "scope-circle-controller"

    def test_pypi_spec_unchanged(self):
        """Existing behavior for PyPI specifiers preserved."""
        pm = PluginManager()

        assert pm._extract_package_name("my-plugin==1.0.0") == "my-plugin"
        assert pm._extract_package_name("my-plugin>=1.0") == "my-plugin"
        assert pm._extract_package_name("my-plugin[extra]") == "my-plugin"
        assert pm._extract_package_name("my-plugin") == "my-plugin"
