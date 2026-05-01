---
name: fal-runner
description: Verify fal app behavior end to end: deploy aliases, test public websocket endpoints, and inspect live runners. Use when working with `fal deploy`, `fal run`, `fal runners exec`, or websocket readiness.
---

# Fal Runner

## Quick Start

Use this skill when you need to:
- verify that a fal app is reachable from its public URL
- confirm a websocket endpoint returns the expected initial message
- inspect the live runner process tree and filesystem

Default assumptions from this repo:
- `uv run fal ...` is the correct invocation
- realtime health is best checked through the public websocket, not just container logs
- runner behavior must be verified from both the public endpoint and the live runner

## Standard Workflow

1. Confirm the public endpoint behavior first.
2. Inspect the active runner IDs for the deployed alias.
3. Exec into the live runner and inspect:
   - process list
   - `/app/src/...` files
   - `__pycache__`
   - relevant temp directories
   - environment variables
4. If testing temporary deploys, clearly mark them with a name that identifies the user and model (e.g. `<user>-<model>`) and clean them up afterward.

## Public Verification

Check alias runners:

```bash
uv run fal apps runners <alias> --env main --json
```

Check public websocket readiness:

```bash
uv run python - <<'PY'
import asyncio, websockets

URL = "wss://fal.run/<team>/<alias>/ws"

async def main():
    async with websockets.connect(URL) as ws:
        print(await asyncio.wait_for(ws.recv(), timeout=20))

asyncio.run(main())
PY
```

Expected success pattern for this workflow:
- connection succeeds
- first message is a ready-style payload such as `{"type":"ready",...}`

If the websocket fails:
- verify the app is actually public
- verify the outer fal websocket is forwarding to a healthy inner runner
- do not assume a generic forwarding error means the inner app is broken

## Runner Inspection

Use the live runner ID from `fal apps runners`.

Useful checks:

```bash
uv run fal runners exec <runner-id> -- sh -lc 'ps -efww'
```

```bash
uv run fal runners exec <runner-id> -- sh -lc 'ls -la /app/src/scope/cloud'
```

```bash
uv run fal runners exec <runner-id> -- sh -lc 'ls -la /app/src/scope/cloud/__pycache__'
```

```bash
uv run fal runners exec <runner-id> -- sh -lc 'tr "\0" "\n" </proc/1/environ | sed -n "1,160p"'
```

```bash
uv run fal runners exec <runner-id> -- sh -lc 'tr "\0" " " </proc/1/cmdline'
```

For broader artifact inspection:

```bash
uv run fal runners exec <runner-id> -- sh -lc '/usr/bin/python3.12 - <<\"PY\"
import os
for root in ["/app", "/tmp", "/root", "/local", "/alloc"]:
    if not os.path.exists(root):
        continue
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            if "livepeer_fal_app" in name or "fal_app" in name:
                print(os.path.join(dirpath, name))
PY'
```

## Interpretation Rules

- A working public websocket matters more than whether the entrypoint file is visible under `/app/src`.
- Always test the public websocket directly rather than relying only on container logs.

## Discovery And Packaging Rules

`fal deploy` does not blindly execute the file path as the runtime authority. It:
- executes the file with `runpy.run_path(...)`
- finds a `fal.App` symbol in that module dict
- wraps and serializes the resulting callable
- sends both serialized function data and source text to the server

## Repo-Specific Notes

- For cloud app testing here, use `uv run fal ...`.
- The live inner runner may be started from `/app/src/scope/cloud/livepeer_app.py`.

## Additional Resources

- For deeper notes and investigation patterns, see [reference.md](reference.md).
