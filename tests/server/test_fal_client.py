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


def test_fal_client_initialization_includes_data_channel_attrs():
    """Test FalClient initializes with data channel attributes."""
    from scope.server.fal_client import FalClient

    client = FalClient(app_id="owner/app/webrtc", api_key="test-key")

    assert client.data_channel is None
    assert client._pending_parameters == {}


def test_send_parameters_queues_when_channel_closed():
    """Test parameters are queued when data channel is not open."""

    from scope.server.fal_client import FalClient

    client = FalClient(app_id="owner/app/webrtc", api_key="test-key")

    # No data channel (not connected)
    client.data_channel = None

    result = client.send_parameters({"prompt": "test prompt", "noise_scale": 0.5})

    assert result is False
    assert client._pending_parameters == {"prompt": "test prompt", "noise_scale": 0.5}


def test_send_parameters_queues_when_channel_not_open():
    """Test parameters are queued when data channel exists but not open."""
    from unittest.mock import MagicMock

    from scope.server.fal_client import FalClient

    client = FalClient(app_id="owner/app/webrtc", api_key="test-key")

    # Data channel exists but not open
    mock_channel = MagicMock()
    mock_channel.readyState = "connecting"
    client.data_channel = mock_channel

    result = client.send_parameters({"prompt": "test prompt"})

    assert result is False
    assert client._pending_parameters == {"prompt": "test prompt"}


def test_send_parameters_sends_when_channel_open():
    """Test parameters are sent when data channel is open."""
    from unittest.mock import MagicMock

    from scope.server.fal_client import FalClient

    client = FalClient(app_id="owner/app/webrtc", api_key="test-key")

    # Data channel is open
    mock_channel = MagicMock()
    mock_channel.readyState = "open"
    mock_channel.send = MagicMock()
    client.data_channel = mock_channel

    result = client.send_parameters({"prompt": "test prompt", "noise_scale": 0.5})

    assert result is True
    mock_channel.send.assert_called_once()
    # Verify JSON contains the parameters
    sent_json = mock_channel.send.call_args[0][0]
    import json

    sent_data = json.loads(sent_json)
    assert sent_data == {"prompt": "test prompt", "noise_scale": 0.5}


def test_send_parameters_filters_none_values():
    """Test parameters with None values are filtered out."""
    from unittest.mock import MagicMock

    from scope.server.fal_client import FalClient

    client = FalClient(app_id="owner/app/webrtc", api_key="test-key")

    # Data channel is open
    mock_channel = MagicMock()
    mock_channel.readyState = "open"
    mock_channel.send = MagicMock()
    client.data_channel = mock_channel

    result = client.send_parameters(
        {"prompt": "test", "noise_scale": None, "denoising_steps": 5}
    )

    assert result is True
    # Verify None values are filtered
    sent_json = mock_channel.send.call_args[0][0]
    import json

    sent_data = json.loads(sent_json)
    assert sent_data == {"prompt": "test", "denoising_steps": 5}
    assert "noise_scale" not in sent_data


def test_send_parameters_accumulates_pending():
    """Test multiple send_parameters calls accumulate pending parameters."""
    from scope.server.fal_client import FalClient

    client = FalClient(app_id="owner/app/webrtc", api_key="test-key")
    client.data_channel = None  # Not connected

    client.send_parameters({"prompt": "first"})
    client.send_parameters({"noise_scale": 0.5})
    client.send_parameters({"prompt": "second"})  # Should override first

    assert client._pending_parameters == {"prompt": "second", "noise_scale": 0.5}
