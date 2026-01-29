"""Tests for fal.ai cloud API endpoints."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


class TestFalApiEndpoints:
    """Tests for fal API endpoints."""

    @pytest.fixture
    def mock_webrtc_manager(self):
        """Create a mock WebRTC manager."""
        manager = MagicMock()
        manager.sessions = {}
        # Explicitly set get_fal_config to return None (no pending fal config)
        manager.get_fal_config.return_value = None
        return manager

    @pytest.fixture
    def mock_session_with_fal(self):
        """Create a mock session with fal-enabled frame processor."""
        session = MagicMock()
        frame_processor = MagicMock()
        frame_processor.fal_enabled = True
        frame_processor.fal_client = MagicMock()
        frame_processor.fal_client.app_id = "test/app/webrtc"
        frame_processor.connect_to_fal = AsyncMock()
        frame_processor.disconnect_from_fal = AsyncMock()
        session.video_track = MagicMock()
        session.video_track.frame_processor = frame_processor
        return session

    @pytest.fixture
    def mock_session_without_fal(self):
        """Create a mock session without fal enabled."""
        session = MagicMock()
        frame_processor = MagicMock()
        frame_processor.fal_enabled = False
        frame_processor.fal_client = None
        frame_processor.connect_to_fal = AsyncMock()
        frame_processor.disconnect_from_fal = AsyncMock()
        session.video_track = MagicMock()
        session.video_track.frame_processor = frame_processor
        return session

    @pytest.fixture
    def client(self, mock_webrtc_manager):
        """Create a test client with mocked dependencies."""
        from scope.server.app import app, get_webrtc_manager

        app.dependency_overrides[get_webrtc_manager] = lambda: mock_webrtc_manager
        client = TestClient(app)
        yield client
        app.dependency_overrides.clear()

    def test_fal_status_not_connected(self, client, mock_webrtc_manager):
        """Test fal status endpoint when not connected."""
        response = client.get("/api/v1/fal/status")
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is False
        assert data["app_id"] is None

    def test_fal_status_connected(
        self, client, mock_webrtc_manager, mock_session_with_fal
    ):
        """Test fal status endpoint when connected."""
        mock_webrtc_manager.sessions = {"session1": mock_session_with_fal}

        response = client.get("/api/v1/fal/status")
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is True
        assert data["app_id"] == "test/app/webrtc"

    def test_fal_connect_no_sessions(self, client, mock_webrtc_manager):
        """Test fal connect endpoint with no active sessions."""
        response = client.post(
            "/api/v1/fal/connect",
            json={"app_id": "owner/app/webrtc", "api_key": "test-key"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is True
        assert data["app_id"] == "owner/app/webrtc"

    def test_fal_connect_with_sessions(
        self, client, mock_webrtc_manager, mock_session_without_fal
    ):
        """Test fal connect endpoint with active sessions."""
        mock_webrtc_manager.sessions = {"session1": mock_session_without_fal}

        response = client.post(
            "/api/v1/fal/connect",
            json={"app_id": "owner/app/webrtc", "api_key": "test-key"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is True
        assert data["app_id"] == "owner/app/webrtc"

        # Verify connect_to_fal was called
        fp = mock_session_without_fal.video_track.frame_processor
        fp.connect_to_fal.assert_called_once_with(
            app_id="owner/app/webrtc",
            api_key="test-key",
        )

    def test_fal_connect_validation_error(self, client):
        """Test fal connect endpoint with invalid request."""
        # Missing required fields
        response = client.post(
            "/api/v1/fal/connect",
            json={},
        )
        assert response.status_code == 422  # Validation error

    def test_fal_disconnect_no_sessions(self, client, mock_webrtc_manager):
        """Test fal disconnect endpoint with no active sessions."""
        response = client.post("/api/v1/fal/disconnect")
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is False
        assert data["app_id"] is None

    def test_fal_disconnect_with_sessions(
        self, client, mock_webrtc_manager, mock_session_with_fal
    ):
        """Test fal disconnect endpoint with active sessions."""
        mock_webrtc_manager.sessions = {"session1": mock_session_with_fal}

        response = client.post("/api/v1/fal/disconnect")
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is False
        assert data["app_id"] is None

        # Verify disconnect_from_fal was called
        fp = mock_session_with_fal.video_track.frame_processor
        fp.disconnect_from_fal.assert_called_once()

    def test_fal_connect_multiple_sessions(self, client, mock_webrtc_manager):
        """Test fal connect endpoint with multiple active sessions."""
        # Create multiple sessions
        sessions = {}
        for i in range(3):
            session = MagicMock()
            frame_processor = MagicMock()
            frame_processor.connect_to_fal = AsyncMock()
            session.video_track = MagicMock()
            session.video_track.frame_processor = frame_processor
            sessions[f"session{i}"] = session

        mock_webrtc_manager.sessions = sessions

        response = client.post(
            "/api/v1/fal/connect",
            json={"app_id": "owner/app/webrtc", "api_key": "test-key"},
        )
        assert response.status_code == 200

        # Verify all sessions were connected
        for session in sessions.values():
            fp = session.video_track.frame_processor
            fp.connect_to_fal.assert_called_once_with(
                app_id="owner/app/webrtc",
                api_key="test-key",
            )

    def test_fal_disconnect_multiple_sessions(self, client, mock_webrtc_manager):
        """Test fal disconnect endpoint with multiple active sessions."""
        # Create multiple sessions
        sessions = {}
        for i in range(3):
            session = MagicMock()
            frame_processor = MagicMock()
            frame_processor.disconnect_from_fal = AsyncMock()
            session.video_track = MagicMock()
            session.video_track.frame_processor = frame_processor
            sessions[f"session{i}"] = session

        mock_webrtc_manager.sessions = sessions

        response = client.post("/api/v1/fal/disconnect")
        assert response.status_code == 200

        # Verify all sessions were disconnected
        for session in sessions.values():
            fp = session.video_track.frame_processor
            fp.disconnect_from_fal.assert_called_once()

    def test_fal_status_with_session_without_video_track(
        self, client, mock_webrtc_manager
    ):
        """Test fal status handles sessions without video track gracefully."""
        session = MagicMock()
        session.video_track = None
        mock_webrtc_manager.sessions = {"session1": session}

        response = client.get("/api/v1/fal/status")
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is False
        assert data["app_id"] is None

    def test_fal_status_with_session_without_frame_processor(
        self, client, mock_webrtc_manager
    ):
        """Test fal status handles sessions without frame processor gracefully."""
        session = MagicMock()
        session.video_track = MagicMock()
        session.video_track.frame_processor = None
        mock_webrtc_manager.sessions = {"session1": session}

        response = client.get("/api/v1/fal/status")
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is False
        assert data["app_id"] is None
