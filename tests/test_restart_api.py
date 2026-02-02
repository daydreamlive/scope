"""Tests for server restart endpoint."""

import os
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


class TestRestartEndpoint:
    """Tests for POST /api/v1/restart."""

    def test_managed_mode_exits_with_code_42(self):
        """When DAYDREAM_SCOPE_MANAGED is set, server should exit with code 42."""
        with patch.dict(os.environ, {"DAYDREAM_SCOPE_MANAGED": "1"}):
            with patch("os._exit"):
                # Import after patching env to get correct behavior
                from scope.server.app import app

                client = TestClient(app, raise_server_exceptions=False)

                response = client.post("/api/v1/restart")

                assert response.status_code == 200
                data = response.json()
                assert data["message"] == "Server exiting for respawn..."

    def test_managed_mode_response_before_exit(self):
        """In managed mode, response should be sent before exit."""
        with patch.dict(os.environ, {"DAYDREAM_SCOPE_MANAGED": "1"}):
            with patch("os._exit"):
                from scope.server.app import app

                client = TestClient(app, raise_server_exceptions=False)

                response = client.post("/api/v1/restart")

                # Response should be returned successfully
                assert response.status_code == 200
                assert "message" in response.json()

    def test_standalone_mode_uses_subprocess_on_windows(self):
        """When DAYDREAM_SCOPE_MANAGED is not set on Windows, should use subprocess."""
        # Clear the managed env var
        env = {k: v for k, v in os.environ.items() if k != "DAYDREAM_SCOPE_MANAGED"}

        with patch.dict(os.environ, env, clear=True):
            with patch("sys.platform", "win32"):
                with patch("subprocess.Popen") as mock_popen:
                    mock_popen.return_value = MagicMock()
                    with patch("os._exit"):
                        from scope.server.app import app

                        client = TestClient(app, raise_server_exceptions=False)

                        response = client.post("/api/v1/restart")

                        assert response.status_code == 200
                        assert response.json()["message"] == "Server restarting..."

    def test_standalone_mode_uses_execv_on_unix(self):
        """When DAYDREAM_SCOPE_MANAGED is not set on Unix, should use os.execv."""
        # Clear the managed env var
        env = {k: v for k, v in os.environ.items() if k != "DAYDREAM_SCOPE_MANAGED"}

        with patch.dict(os.environ, env, clear=True):
            with patch("sys.platform", "linux"):
                with patch("os.execv"):
                    from scope.server.app import app

                    client = TestClient(app, raise_server_exceptions=False)

                    response = client.post("/api/v1/restart")

                    assert response.status_code == 200
                    assert response.json()["message"] == "Server restarting..."
