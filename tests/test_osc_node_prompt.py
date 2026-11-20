"""Regression test: node-scoped OSC prompt updates must reach the backend as `prompts`.

Issue #616: after changing the prompt via OSC, reset_cache reverted to the first
prompt because the backend only received `prompt` (ignored by pipelines) and kept
the stale `prompts` value.
"""

from unittest.mock import MagicMock

from scope.server.osc_server import OSCServer


def _server_with_known_path(path_info: dict) -> tuple[OSCServer, MagicMock]:
    server = OSCServer("127.0.0.1", 0)
    webrtc = MagicMock()
    server._webrtc_manager = webrtc
    key = f"{path_info['node_id']}/{path_info['param']}"
    server._known_paths_cache = {key: path_info}
    server._known_paths_cache_time = float("inf")
    return server, webrtc


def test_node_scoped_prompt_is_sent_as_prompts_list():
    server, webrtc = _server_with_known_path(
        {
            "node_id": "node1",
            "pipeline_id": "longlive",
            "param": "prompt",
            "type": "string",
        }
    )

    server._handle_osc_message("/scope/node1/prompt", "flames")

    webrtc.broadcast_parameter_update.assert_called_once_with(
        {"node_id": "node1", "prompts": [{"text": "flames", "weight": 100}]}
    )


def test_node_scoped_non_prompt_param_is_forwarded_unchanged():
    server, webrtc = _server_with_known_path(
        {
            "node_id": "node1",
            "pipeline_id": "longlive",
            "param": "noise_scale",
            "type": "float",
        }
    )

    server._handle_osc_message("/scope/node1/noise_scale", 0.5)

    webrtc.broadcast_parameter_update.assert_called_once_with(
        {"node_id": "node1", "noise_scale": 0.5}
    )
