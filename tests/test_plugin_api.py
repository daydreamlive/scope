"""Integration tests for plugin API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_plugin_manager():
    """Mock PluginManager for API tests."""
    manager = MagicMock()
    manager.list_plugins_async = AsyncMock(return_value=[])
    manager.install_plugin_async = AsyncMock(
        return_value={"success": True, "message": "ok", "plugin": None}
    )
    manager.uninstall_plugin_async = AsyncMock(
        return_value={"success": True, "message": "ok", "unloaded_pipelines": []}
    )
    manager.reload_plugin_async = AsyncMock(
        return_value={
            "success": True,
            "message": "ok",
            "reloaded_pipelines": [],
            "added_pipelines": [],
            "removed_pipelines": [],
        }
    )
    return manager


@pytest.fixture
def mock_pipeline_manager():
    """Mock PipelineManager for API tests."""
    manager = MagicMock()
    manager.get_status_info_async = AsyncMock(
        return_value={"status": "not_loaded", "pipeline_id": None}
    )
    return manager


@pytest.fixture
def client(mock_plugin_manager, mock_pipeline_manager):
    """Create test client with mocked dependencies."""
    with patch(
        "scope.core.plugins.get_plugin_manager", return_value=mock_plugin_manager
    ):
        with patch("scope.server.app.pipeline_manager", mock_pipeline_manager):
            with patch("scope.server.app.webrtc_manager", MagicMock()):
                # Import app after patching
                from scope.server.app import app

                yield TestClient(app, raise_server_exceptions=False)


class TestListPluginsEndpoint:
    """Tests for GET /api/v1/plugins endpoint."""

    def test_returns_empty_list(self, client, mock_plugin_manager):
        """GET /api/v1/plugins with no plugins should return empty list."""
        mock_plugin_manager.list_plugins_async.return_value = []

        response = client.get("/api/v1/plugins")

        assert response.status_code == 200
        data = response.json()
        assert data["plugins"] == []
        assert data["total"] == 0

    def test_returns_plugin_list(self, client, mock_plugin_manager):
        """Mock plugins should return correct response structure."""
        mock_plugin_manager.list_plugins_async.return_value = [
            {
                "name": "test-plugin",
                "version": "1.0.0",
                "author": "test-author",
                "description": "A test plugin",
                "source": "pypi",
                "editable": False,
                "editable_path": None,
                "pipelines": [
                    {"pipeline_id": "test-pipeline", "pipeline_name": "Test Pipeline"}
                ],
                "latest_version": "2.0.0",
                "update_available": True,
            }
        ]

        response = client.get("/api/v1/plugins")

        assert response.status_code == 200
        data = response.json()
        assert len(data["plugins"]) == 1
        assert data["total"] == 1
        assert data["plugins"][0]["name"] == "test-plugin"
        assert data["plugins"][0]["version"] == "1.0.0"
        assert len(data["plugins"][0]["pipelines"]) == 1
        assert data["plugins"][0]["latest_version"] == "2.0.0"
        assert data["plugins"][0]["update_available"] is True

    def test_handles_manager_errors(self, client, mock_plugin_manager):
        """Manager exception should return 500 error."""
        mock_plugin_manager.list_plugins_async.side_effect = Exception("Test error")

        response = client.get("/api/v1/plugins")

        assert response.status_code == 500


class TestInstallPluginEndpoint:
    """Tests for POST /api/v1/plugins endpoint."""

    def test_installs_plugin(self, client, mock_plugin_manager):
        """POST with valid package should succeed."""
        mock_plugin_manager.install_plugin_async.return_value = {
            "success": True,
            "message": "Successfully installed test-plugin",
            "plugin": {
                "name": "test-plugin",
                "version": "1.0.0",
                "source": "pypi",
                "editable": False,
                "editable_path": None,
                "pipelines": [],
            },
        }

        response = client.post(
            "/api/v1/plugins", json={"package": "test-plugin", "editable": False}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["plugin"]["name"] == "test-plugin"

    def test_validates_request_body(self, client):
        """Missing package should return error (500 due to manual JSON parsing)."""
        response = client.post("/api/v1/plugins", json={})

        # The endpoint uses manual JSON parsing and Pydantic validation,
        # which results in a 500 error for missing required fields
        # rather than FastAPI's automatic 422 validation
        assert response.status_code == 500

    def test_handles_dependency_error(self, client, mock_plugin_manager):
        """PluginDependencyError should return 422."""
        from scope.core.plugins import PluginDependencyError

        mock_plugin_manager.install_plugin_async.side_effect = PluginDependencyError(
            "Conflict"
        )

        response = client.post("/api/v1/plugins", json={"package": "bad-plugin"})

        assert response.status_code == 422

    def test_handles_collision_error(self, client, mock_plugin_manager):
        """PluginNameCollisionError should return 409."""
        from scope.core.plugins import PluginNameCollisionError

        mock_plugin_manager.install_plugin_async.side_effect = PluginNameCollisionError(
            "Name collision"
        )

        response = client.post("/api/v1/plugins", json={"package": "duplicate-plugin"})

        assert response.status_code == 409

    def test_handles_install_error(self, client, mock_plugin_manager):
        """PluginInstallError should return 500."""
        from scope.core.plugins import PluginInstallError

        mock_plugin_manager.install_plugin_async.side_effect = PluginInstallError(
            "Install failed"
        )

        response = client.post("/api/v1/plugins", json={"package": "broken-plugin"})

        assert response.status_code == 500


class TestUninstallPluginEndpoint:
    """Tests for DELETE /api/v1/plugins/{name} endpoint."""

    def test_uninstalls_plugin(self, client, mock_plugin_manager):
        """DELETE /api/v1/plugins/{name} should uninstall."""
        mock_plugin_manager.uninstall_plugin_async.return_value = {
            "success": True,
            "message": "Successfully uninstalled test-plugin",
            "unloaded_pipelines": ["test-pipeline"],
        }

        response = client.delete("/api/v1/plugins/test-plugin")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "test-pipeline" in data["unloaded_pipelines"]

    def test_handles_not_found(self, client, mock_plugin_manager):
        """PluginNotFoundError should return 404."""
        from scope.core.plugins import PluginNotFoundError

        mock_plugin_manager.uninstall_plugin_async.side_effect = PluginNotFoundError(
            "Not found"
        )

        response = client.delete("/api/v1/plugins/nonexistent")

        assert response.status_code == 404

    def test_returns_unloaded_pipelines(self, client, mock_plugin_manager):
        """Response should include unloaded list."""
        mock_plugin_manager.uninstall_plugin_async.return_value = {
            "success": True,
            "message": "ok",
            "unloaded_pipelines": ["pipeline-1", "pipeline-2"],
        }

        response = client.delete("/api/v1/plugins/test-plugin")

        assert response.status_code == 200
        data = response.json()
        assert len(data["unloaded_pipelines"]) == 2


class TestReloadPluginEndpoint:
    """Tests for POST /api/v1/plugins/{name}/reload endpoint."""

    def test_reloads_editable_plugin(self, client, mock_plugin_manager):
        """POST with force=false should reload."""
        mock_plugin_manager.reload_plugin_async.return_value = {
            "success": True,
            "message": "Successfully reloaded test-plugin",
            "reloaded_pipelines": ["test-pipeline"],
            "added_pipelines": [],
            "removed_pipelines": [],
        }

        response = client.post(
            "/api/v1/plugins/test-plugin/reload", json={"force": False}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_reloads_with_force(self, client, mock_plugin_manager):
        """POST with force=true should reload."""
        mock_plugin_manager.reload_plugin_async.return_value = {
            "success": True,
            "message": "ok",
            "reloaded_pipelines": ["test-pipeline"],
            "added_pipelines": [],
            "removed_pipelines": [],
        }

        response = client.post(
            "/api/v1/plugins/test-plugin/reload", json={"force": True}
        )

        assert response.status_code == 200
        # Verify force was passed
        mock_plugin_manager.reload_plugin_async.assert_called_once()
        call_kwargs = mock_plugin_manager.reload_plugin_async.call_args.kwargs
        assert call_kwargs["force"] is True

    def test_handles_not_found(self, client, mock_plugin_manager):
        """PluginNotFoundError should return 404."""
        from scope.core.plugins import PluginNotFoundError

        mock_plugin_manager.reload_plugin_async.side_effect = PluginNotFoundError(
            "Not found"
        )

        response = client.post(
            "/api/v1/plugins/nonexistent/reload", json={"force": False}
        )

        assert response.status_code == 404

    def test_handles_not_editable(self, client, mock_plugin_manager):
        """PluginNotEditableError should return 400."""
        from scope.core.plugins import PluginNotEditableError

        mock_plugin_manager.reload_plugin_async.side_effect = PluginNotEditableError(
            "Not editable"
        )

        response = client.post(
            "/api/v1/plugins/pypi-plugin/reload", json={"force": False}
        )

        assert response.status_code == 400

    def test_handles_in_use(self, client, mock_plugin_manager):
        """PluginInUseError should return 409 with loaded_pipelines."""
        from scope.core.plugins import PluginInUseError

        error = PluginInUseError("In use", loaded_pipelines=["pipeline-1"])
        mock_plugin_manager.reload_plugin_async.side_effect = error

        response = client.post(
            "/api/v1/plugins/in-use-plugin/reload", json={"force": False}
        )

        assert response.status_code == 409
        data = response.json()
        assert "pipeline-1" in data["detail"]["loaded_pipelines"]
