# Fal Runner Verification Reference

## What We Learned

### 1. Public verification must come first

The most reliable first check is the public websocket URL, not internal logs. For the working app, a successful connection returned a ready payload immediately.

Use:

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

### 2. Runner inspection is necessary even when the public URL works

For this investigation, the public websocket worked while the expected deploy entrypoint file was still missing from `/app/src/scope/cloud`.

Always inspect:
- `ps -efww`
- `/proc/*/cmdline`
- `/proc/*/environ`
- `/app/src/...`
- `__pycache__`
- `/tmp`, `/root`, `/local`, `/alloc`

### 3. `app_auth` is the meaningful public/private field

For `fal.App`, use:

```python
app_auth = "public"
```

Do not rely on an `auth_mode` class attribute to make the deployed app public.

### 4. The deploy target file is not the same thing as the runtime owner

`fal deploy path/to/file.py::MyApp` does not mean the runner will necessarily import `path/to/file.py` from `/app/src`.

Observed behavior:
- `fal deploy` loaded the app object from the entrypoint namespace
- the app object could still belong to another module
- fal then serialized the resolved callable
- the app ran successfully even though the entrypoint file itself did not show up under `/app/src`

## Relevant fal Mechanics

### `load_function_from(...)`

The fal client:
- executes the file with `runpy.run_path(file_path)`
- finds the app/function object from the resulting globals
- reads the file text as `source_code`
- wraps the app for remote execution

Implication:
- discovery is filename-based
- execution packaging is object-based

### `_find_target(...)`

fal searches the globals produced by `runpy.run_path(...)`. If an imported app is assigned into that namespace, fal can find it there.

That does not make it a locally defined class.

### `wrap_app(...)` and `include_modules_from(...)`

fal packaging follows the class object’s module ownership through `__module__`.

Implication:
- `Alias = imported_class` keeps original ownership
- `class Alias(imported_class): pass` creates new ownership
- rewriting `__module__` can also change what fal considers local

## Why alias/import patterns behaved differently

### Imported alias

Example shape:

```python
from scope.cloud.fal_app import LivepeerScopeApp as ImportedApp

LivepeerScopeApp = ImportedApp
```

Effect:
- new name
- same class object
- original `__module__`
- fal still treats the class as owned by `scope.cloud.fal_app`

### Local subclass

Example shape:

```python
from scope.cloud.fal_app import LivepeerScopeApp as ImportedApp

class LivepeerScopeApp(ImportedApp):
    pass
```

Effect:
- new class object
- local module ownership
- fal treats the entrypoint as the owner of the app class

### Module-hack variant

Example shape:

```python
from scope.cloud.fal_app import LivepeerScopeApp as ImportedApp

ImportedApp.__module__ = __name__
LivepeerScopeApp = ImportedApp
```

Effect:
- same class object
- forged local ownership
- useful as an experiment, but less clean than a genuinely local definition

## Interpreting Missing Files On Runner

If the deployed app works but the file is absent from `/app/src`, the likely explanations are:

1. The app is running from a serialized callable, not a normal import from `/app/src`.
2. fal uploaded or reconstructed execution artifacts separately from the copied image tree.
3. The copied filesystem and the executed app source are not the same source of truth.

Do not jump straight to:
- `.dockerignore`
- bad `COPY src/ /app/src/`
- broken image build

Especially if:
- sibling files are present
- the public websocket works
- the process tree shows the inner app is healthy

## High-Signal Commands

List runners for an alias:

```bash
uv run fal apps runners <alias> --env main --json
```

Check public websocket:

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

Inspect runner filesystem:

```bash
uv run fal runners exec <runner-id> -- sh -lc 'ls -la /app/src/scope/cloud'
```

Inspect runner process tree:

```bash
uv run fal runners exec <runner-id> -- sh -lc 'ps -efww'
```

Inspect process command lines:

```bash
uv run fal runners exec <runner-id> -- sh -lc 'for p in /proc/[0-9]*; do tr "\0" " " <"$p/cmdline" 2>/dev/null; echo; done'
```

Search for app artifacts:

```bash
uv run fal runners exec <runner-id> -- sh -lc '/usr/bin/python3.12 - <<\"PY\"
import os
for root in ["/app", "/tmp", "/root", "/local", "/alloc"]:
    if not os.path.exists(root):
        continue
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            if "fal_app" in name or "livepeer_fal_app" in name:
                print(os.path.join(dirpath, name))
PY'
```

Inspect environment metadata:

```bash
uv run fal runners exec <runner-id> -- sh -lc 'tr "\0" "\n" </proc/1/environ | sed -n "1,200p"'
```

## Decision Checklist

When a fal deploy looks wrong, ask these in order:

1. Does the public endpoint work?
2. Is the alias public through `app_auth`?
3. Which runner revision is active?
4. What process is actually running?
5. Is the inner app healthy locally on the runner?
6. Is the expected file missing only from `/app/src`, or missing everywhere?
7. Is the deployed class locally defined, imported, aliased, subclassed, or module-hacked?
8. Is fal likely following the original class owner instead of the deploy file?

## Practical Guidance For This Repo

- Keep `fal_app.py` untouched when testing a separate Livepeer wrapper.
- Make the Livepeer wrapper self-contained if you want the deployed app object to belong to the Livepeer entrypoint module.
- If you must experiment, use clearly named temporary deployments and clean them up.
- Treat the public websocket handshake as the primary success criterion.
