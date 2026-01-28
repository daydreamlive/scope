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

    def test_runs_uv_pip_install(self):
        """Verify subprocess command is correct."""
        pm = PluginManager()

        with patch.object(pm, "_validate_install_sync", return_value=(True, None)):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
                with patch.object(pm, "_reload_all_plugins"):
                    with patch.object(pm, "_list_plugins_sync", return_value=[]):
                        pm._install_plugin_sync("test-package")

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "uv" in args
        assert "pip" in args
        assert "install" in args
        assert "test-package" in args

    def test_handles_editable_install(self):
        """Verify --editable flag is included."""
        pm = PluginManager()

        with patch.object(pm, "_validate_install_sync", return_value=(True, None)):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
                with patch.object(pm, "_reload_all_plugins"):
                    with patch.object(pm, "_list_plugins_sync", return_value=[]):
                        pm._install_plugin_sync("/path/to/package", editable=True)

        args = mock_run.call_args[0][0]
        assert "--editable" in args

    def test_handles_upgrade_flag(self):
        """Verify --upgrade flag is included."""
        pm = PluginManager()

        with patch.object(pm, "_validate_install_sync", return_value=(True, None)):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
                with patch.object(pm, "_reload_all_plugins"):
                    with patch.object(pm, "_list_plugins_sync", return_value=[]):
                        pm._install_plugin_sync("test-package", upgrade=True)

        args = mock_run.call_args[0][0]
        assert "--upgrade" in args

    def test_raises_on_dependency_error(self):
        """PluginDependencyError should be raised on validation fail."""
        pm = PluginManager()

        with patch.object(
            pm, "_validate_install_sync", return_value=(False, "Conflict")
        ):
            with pytest.raises(PluginDependencyError):
                pm._install_plugin_sync("conflicting-package")

    def test_raises_on_install_error(self):
        """PluginInstallError should be raised on subprocess fail."""
        pm = PluginManager()

        with patch.object(pm, "_validate_install_sync", return_value=(True, None)):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=1, stdout="", stderr="Install failed"
                )
                with pytest.raises(PluginInstallError):
                    pm._install_plugin_sync("bad-package")


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
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
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
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                with patch(
                    "scope.core.pipelines.registry.PipelineRegistry.unregister"
                ) as mock_unregister:
                    pm._uninstall_plugin_sync("test-plugin")

                    mock_unregister.assert_called_once_with("test-pipeline")


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
