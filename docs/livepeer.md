# How to Start Livepeer Mode

This guide walks through running Scope with [Livepeer](https://livepeer.org) so that inference is routed through the Livepeer network instead of running locally.

For details on the underlying architecture, WebSocket protocol, and control messages, see the [Livepeer architecture reference](architecture/livepeer.md).

## Prerequisites

Install the `livepeer` extra, which pulls in `uvicorn[standard]` and the Livepeer Gateway SDK:

```bash
uv sync --extra livepeer
```

## Start the Runner

The runner is the process that actually runs inference. It needs to be reachable by the Livepeer orchestrator.

Start it locally:

```bash
uv run livepeer-runner --host 0.0.0.0 --port 8001
```

Available flags:

| Flag       | Default     | Description                        |
| ---------- | ----------- | ---------------------------------- |
| `--host`   | `0.0.0.0`   | Host to bind to                    |
| `--port`   | `8001`       | Port to bind to                    |
| `--reload` | off          | Enable auto-reload for development |

Set `LIVEPEER_DEBUG=1` to enable verbose logging for the runner and the Livepeer Gateway SDK.

### Runner on Fal

To deploy the runner on [Fal](https://fal.ai) instead of running it locally, use the thin wrapper at `scope.cloud.livepeer_fal_app`:

```bash
fal deploy --env main --auth public src/scope/cloud/livepeer_fal_app.py
```

This starts the Livepeer runner as a subprocess inside the Fal container and proxies `/ws` traffic to it.

## Start the Scope Server

Set environment variables and launch the server:

```bash
LIVEPEER_TOKEN=<base64-json-token> \
LIVEPEER_WS_URL=ws://127.0.0.1:8001/ws \
uv run daydream-scope
```

If the runner is deployed on Fal, use `SCOPE_CLOUD_APP_ID` instead of `LIVEPEER_WS_URL`:

```bash
LIVEPEER_TOKEN=<base64-json-token> \
SCOPE_CLOUD_APP_ID=<app-id>/ws \
uv run daydream-scope
```

Or pass the Fal URL explicitly:

```bash
LIVEPEER_TOKEN=<base64-json-token> \
LIVEPEER_WS_URL=wss://fal.run/<app-id>/ws \
uv run daydream-scope
```

To switch away from explicit runner overrides, unset both `LIVEPEER_WS_URL` and `SCOPE_CLOUD_APP_ID`. In that case the runner URL uses the default Livepeer flow.

### Environment Variables

| Variable             | Required | Description |
| -------------------- | -------- | ----------- |
| `LIVEPEER_ORCH_URL`  | No       | Explicit orchestrator URL. Formats: `host[:port]` or `http(s)://host[:port]`. If unset, token discovery is used. |
| `LIVEPEER_SIGNER`    | No       | Override signer URL used for Livepeer payments. To disable payments, set to a falsy value such as `"off"`. |
| `LIVEPEER_WS_URL`    | No       | Explicit runner WebSocket URL (e.g. `ws://127.0.0.1:8001/ws`). |
| `SCOPE_CLOUD_APP_ID` | No       | Fal app id used to construct `ws_url` as `wss://fal.run/<app-id>`. Must include `/ws` suffix. Used when `LIVEPEER_WS_URL` is not set. |
| `LIVEPEER_TOKEN`     | No       | Base64-encoded JSON token used to start the LV2V job. Can be used to override Livepeer orch / payments routing. |
| `LIVEPEER_DEBUG`     | No       | Enables debug logging for the Livepeer Gateway SDK and local Livepeer modules. |
| `LIVEPEER_DEV_MODE`  | No       | Used for developing against a local Livepeer orchestrator with self-signed certificates. |
| `DAYDREAM_API_BASE`  | No       | Override for the Daydream API base used when validating remote plugin installs (runner only). Defaults to `https://api.daydream.live`. |

## Connect and Stream

Once both the runner and the server are up:

1. Connect Scope to the remote backend from the UI, or call one of:
   - `POST /api/v1/cloud/connect`

   Both accept a `CloudConnectRequest` body. In Livepeer mode, `api_key` and
   `user_id` are required up front. `SCOPE_CLOUD_API_KEY` can override auth
   credentials, while `LIVEPEER_TOKEN` can override Livepeer auth/routing
   behavior (for example signer/discovery/orchestrator selection).

2. Start streaming from the Scope UI. Scope creates the Livepeer LV2V job on connect, then opens media channels when the stream starts.

3. To disconnect, use:
   - `POST /api/v1/cloud/disconnect`

4. To check connection status:
   - `GET /api/v1/cloud/status` — reports the active backend status.
