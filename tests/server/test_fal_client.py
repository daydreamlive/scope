"""Tests for FalClient module."""

from unittest.mock import patch

import pytest


class MockResponse:
    """Mock aiohttp response with async context manager support."""

    def __init__(self, ok=True, status=200, json_data=None, text_data=""):
        self.ok = ok
        self.status = status
        self._json_data = json_data
        self._text_data = text_data

    async def json(self):
        return self._json_data

    async def text(self):
        return self._text_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MockSession:
    """Mock aiohttp ClientSession with async context manager support."""

    def __init__(self, response: MockResponse):
        self._response = response
        self.post_calls = []

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.mark.asyncio
async def test_get_temporary_token_success():
    """Test successful token acquisition from fal API."""
    from scope.server.fal_client import FalClient

    client = FalClient(app_id="owner/app-name/webrtc", api_key="test-key")

    mock_response = MockResponse(ok=True, json_data={"detail": "test-token"})
    mock_session = MockSession(mock_response)

    with patch(
        "scope.server.fal_client.aiohttp.ClientSession", return_value=mock_session
    ):
        token = await client._get_temporary_token()
        assert token == "test-token"


@pytest.mark.asyncio
async def test_get_temporary_token_string_response():
    """Test token acquisition when API returns plain string."""
    from scope.server.fal_client import FalClient

    client = FalClient(app_id="owner/app-name/webrtc", api_key="test-key")

    mock_response = MockResponse(ok=True, json_data="plain-token-string")
    mock_session = MockSession(mock_response)

    with patch(
        "scope.server.fal_client.aiohttp.ClientSession", return_value=mock_session
    ):
        token = await client._get_temporary_token()
        assert token == "plain-token-string"


@pytest.mark.asyncio
async def test_get_temporary_token_failure():
    """Test token acquisition failure handling."""
    from scope.server.fal_client import FalClient

    client = FalClient(app_id="owner/app-name/webrtc", api_key="bad-key")

    mock_response = MockResponse(ok=False, status=401, text_data="Unauthorized")
    mock_session = MockSession(mock_response)

    with patch(
        "scope.server.fal_client.aiohttp.ClientSession", return_value=mock_session
    ):
        with pytest.raises(RuntimeError, match="Token request failed"):
            await client._get_temporary_token()


@pytest.mark.asyncio
async def test_get_temporary_token_extracts_alias():
    """Test that alias is correctly extracted from app_id."""
    from scope.server.fal_client import FalClient

    client = FalClient(app_id="owner/my-cool-app/webrtc", api_key="test-key")

    mock_response = MockResponse(ok=True, json_data={"detail": "token"})
    mock_session = MockSession(mock_response)

    with patch(
        "scope.server.fal_client.aiohttp.ClientSession", return_value=mock_session
    ):
        await client._get_temporary_token()

        # Verify the alias was extracted correctly
        assert len(mock_session.post_calls) == 1
        _, kwargs = mock_session.post_calls[0]
        assert kwargs["json"]["allowed_apps"] == ["my-cool-app"]


def test_build_ws_url():
    """Test WebSocket URL construction."""
    from scope.server.fal_client import FalClient

    client = FalClient(app_id="owner/app-name/webrtc", api_key="test-key")
    url = client._build_ws_url("my-token")
    assert url == "wss://fal.run/owner/app-name/webrtc?fal_jwt_token=my-token"


def test_build_ws_url_strips_slashes():
    """Test URL construction handles leading/trailing slashes."""
    from scope.server.fal_client import FalClient

    client = FalClient(app_id="/owner/app-name/webrtc/", api_key="test-key")
    url = client._build_ws_url("my-token")
    assert url == "wss://fal.run/owner/app-name/webrtc?fal_jwt_token=my-token"


def test_fal_client_initialization():
    """Test FalClient initializes with correct default state."""
    from scope.server.fal_client import FalClient

    client = FalClient(app_id="owner/app/webrtc", api_key="test-key")

    assert client.app_id == "owner/app/webrtc"
    assert client.api_key == "test-key"
    assert client.on_frame_received is None
    assert client.ws is None
    assert client.pc is None
    assert client.output_track is None
    assert not client.stop_event.is_set()


def test_fal_client_with_callback():
    """Test FalClient initializes with frame callback."""
    from scope.server.fal_client import FalClient

    callback = lambda frame: None  # noqa: E731
    client = FalClient(
        app_id="owner/app/webrtc", api_key="test-key", on_frame_received=callback
    )

    assert client.on_frame_received is callback


@pytest.mark.asyncio
async def test_disconnect_when_not_connected():
    """Test disconnect works cleanly when not connected."""
    from scope.server.fal_client import FalClient

    client = FalClient(app_id="owner/app/webrtc", api_key="test-key")

    # Should not raise any exceptions
    await client.disconnect()

    assert client.stop_event.is_set()
    assert client.pc is None
    assert client.ws is None
