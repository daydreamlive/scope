$commitId = "b0d610774ecfe82976eb0d210043940af68a40fa"

$body = @'
This is great work. I did E2E testing using the Scope MCP: connected to the staging orchestrator, proxied API requests (pipeline load, hardware info, loras, plugins, assets), confirmed the remote H100 runner, and verified that all existing Scope functionality is completely unaffected.

A few things I found during testing and code review (details in inline comments):

### Non-blocking

**Three endpoints not migrated to `ScopeCloudBackend`**

`/api/v1/cloud/stats` (app.py:2984), `/api/v1/webrtc/ice-servers` (app.py:1110), and `/api/v1/lora/download` (app.py:1759) still use `get_cloud_connection_manager` directly instead of `get_scope_cloud`. In Livepeer mode, `/cloud/stats` returns the relay manager's stats (shows "not connected" even when Livepeer is fully connected). The other two fall through to local handlers, which may or may not be the intended behavior.

**Race condition in cleanup** (`livepeer_app.py`)

The cleanup event is cleared and re-awaited inside a reconnection loop. If the outer WebSocket reconnects rapidly, could multiple cleanup requests pile up concurrently?
'@

$comments = @(
    @{
        path = "src/scope/server/livepeer.py"
        line = 99
        side = "RIGHT"
        body = @'
**Bug: `_connecting` stuck after double-connect**

If `/cloud/connect` is called while already connected, `connect_background()` sets `self._connecting = True` (line 171), then `connect()` early-returns here before the `try/finally` block that resets it (lines 111-142). `_connecting` stays `True` permanently. The `/cloud/status` endpoint then reports `connected: true, connecting: true` indefinitely.

Could we reset `_connecting` before the early return, or guard in `connect_background()` so it doesn't set the flag when already connected?
'@
    },
    @{
        path = "src/scope/server/livepeer_client.py"
        line = 125
        side = "RIGHT"
        body = @'
**Reconnect after disconnect: runner session not cleaned up**

After disconnect + reconnect, the orchestrator returns HTTP 500: `"unexpected message type "error" between ready and started"`. The error persists across retries, so once a session goes stale on the runner side, the client can't recover without the runner being recycled. Would retry-with-backoff or a session cleanup request before reconnecting help here?

Also, during the successful connection, `livepeer_gateway` logs: `"No running event loop; per-segment payments not started"`. May be intentional for now but worth a TODO?
'@
    },
    @{
        path = "src/scope/server/livepeer_client.py"
        line = 443
        side = "RIGHT"
        body = @'
**Events loop failure triggers full shutdown without reconnect attempt**

If the events loop encounters an error or stops unexpectedly, `_shutdown()` is called immediately. Combined with `stop_media()` being called from the output loop on unexpected exit (line 389), this can cascade. A transient network blip would tear down the entire session rather than attempting reconnection. Could this try a reconnect before going to full shutdown?
'@
    },
    @{
        path = "src/scope/cloud/livepeer_app.py"
        line = 362
        side = "RIGHT"
        body = @'
**No size limits on base64 content**

`base64.b64decode(body["_base64_content"])` has no size check. A large payload could exhaust memory before reaching the API handler. Would it make sense to cap the encoded length before decoding?
'@
    },
    @{
        path = "src/scope/server/cloud_proxy.py"
        line = 393
        side = "RIGHT"
        body = @'
**Non-blocking: Recording download blocked in Livepeer mode**

This `isinstance` check raises a 400 for Livepeer. There's a TODO here already, just flagging for tracking.
'@
    }
)

$payload = @{
    commit_id = $commitId
    body = $body
    comments = $comments
}

$json = $payload | ConvertTo-Json -Depth 4
$json | gh api repos/daydreamlive/scope/pulls/738/reviews -X POST --input -
