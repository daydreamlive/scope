"""In-process driver for the Scope agent.

Hits ``POST /api/v1/agent/chat`` via ``httpx.ASGITransport`` + ``asgi-lifespan``
so no uvicorn server or port is needed. Parses the SSE stream, captures every
event as a structured trace, and pulls out the final ``workflow_proposal``
payload if the agent produced one.

Contract::

    result = await run_case(app, prompt, model=..., provider=...)
    result.proposal  # dict | None — the `graph` from the workflow_proposal SSE
    result.trace     # list[{event, data}] — every SSE event, in order
    result.error     # str | None — provider/transport error if any
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx
from asgi_lifespan import LifespanManager

logger = logging.getLogger(__name__)


@dataclass
class DriveResult:
    proposal: dict | None = None  # the 'graph' from workflow_proposal SSE
    proposal_id: str | None = None
    rationale: str = ""
    trace: list[dict] = field(default_factory=list)
    error: str | None = None
    session_id: str | None = None


async def _parse_sse_stream(resp: httpx.Response) -> AsyncIterator[dict]:
    """Yield ``{event, data}`` dicts. Swallows malformed lines."""
    current_event: str | None = None
    async for raw_line in resp.aiter_lines():
        if raw_line == "":
            current_event = None
            continue
        line = raw_line.rstrip("\r")
        if line.startswith("event:"):
            current_event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            payload = line.split(":", 1)[1].strip()
            try:
                data = json.loads(payload)
            except Exception:
                data = {"_raw": payload}
            yield {"event": current_event or "message", "data": data}


async def run_case(
    app: Any,
    prompt: str,
    *,
    model_override: str | None = None,
    provider_override: str | None = None,
) -> DriveResult:
    """Drive one agent turn with ``prompt`` and return the captured result.

    ``app`` is the FastAPI app instance (usually ``scope.server.app.app``).
    We pass a fresh session_id=None so the store mints one for each case —
    no cross-case contamination.
    """
    # Apply provider/model overrides by writing to the config file on disk
    # (that's what the app reads). We rely on the caller to have scoped this
    # via EnvOverride if they want to reset it after.
    if model_override or provider_override:
        _patch_agent_config(model=model_override, provider=provider_override)

    result = DriveResult()
    transport = httpx.ASGITransport(app=app)
    try:
        # Scope's startup runs plugin installs, pipeline registration, WebRTC
        # setup, and OSC init — way past asgi-lifespan's 5s default. Give it
        # plenty of headroom; a cold first run on CI can take >30s.
        async with LifespanManager(app, startup_timeout=180, shutdown_timeout=30):
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://scope-eval.local",
                timeout=httpx.Timeout(300.0, connect=10.0),
            ) as client:
                async with client.stream(
                    "POST",
                    "/api/v1/agent/chat",
                    json={"message": prompt},
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        result.error = (
                            f"HTTP {resp.status_code}: "
                            f"{body.decode('utf-8', errors='replace')[:500]}"
                        )
                        return result
                    result.session_id = resp.headers.get("x-agent-session-id")
                    async for evt in _parse_sse_stream(resp):
                        result.trace.append(evt)
                        name = evt["event"]
                        data = evt["data"]
                        if name == "workflow_proposal":
                            # First proposal wins — agent should only emit one.
                            if result.proposal is None:
                                result.proposal = data.get("graph")
                                result.proposal_id = data.get("proposal_id")
                                result.rationale = data.get("rationale") or ""
                        elif name == "error":
                            # Don't short-circuit — the turn_end still arrives
                            # and the trace is useful for debugging.
                            msg = data.get("message") or str(data)
                            result.error = (result.error or "") + msg + "\n"
                        elif name == "turn_end":
                            # Agent finished. We don't need more events.
                            break
    except Exception as e:
        logger.exception("driver transport error")
        result.error = f"{type(e).__name__}: {e}"
    return result


def _patch_agent_config(*, model: str | None, provider: str | None) -> None:
    """Best-effort update of the on-disk agent config. Safe to call repeatedly.

    We do this by loading, mutating, saving via the same helpers the server
    uses, so we respect any fields we don't know about.
    """
    from scope.server.agent_state import (
        AgentConfig,
        load_agent_config,
        save_agent_config,
    )

    cfg = load_agent_config()
    if provider:
        cfg = AgentConfig(
            provider=provider,  # type: ignore[arg-type]
            model=model or cfg.model,
            base_url=cfg.base_url,
        )
    elif model:
        cfg = AgentConfig(provider=cfg.provider, model=model, base_url=cfg.base_url)
    save_agent_config(cfg)
