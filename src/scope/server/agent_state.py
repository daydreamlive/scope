"""Agent session + provider config state (in-memory, single-process).

MVP scope:
- AgentSession: full conversation history per session_id, in memory only.
- AgentConfig: which provider + model to use (persisted to disk so users don't
  have to re-enter settings every server restart).
- WorkflowProposal: pending proposal awaiting user decision.

No database, no cross-process sharing, no persistence of chat history. Sessions
are evicted after 1h idle. This matches the "MVP" posture in the plan.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Config file for provider selection (persisted across restarts).
AGENT_CONFIG_FILE = "~/.daydream-scope/agent_config.json"

# Provider token files (api keys), same 0o600 pattern as CivitAI.
ANTHROPIC_TOKEN_FILE = "~/.daydream-scope/anthropic_token"
OPENAI_TOKEN_FILE = "~/.daydream-scope/openai_token"
LLM_CUSTOM_TOKEN_FILE = "~/.daydream-scope/llm_custom_token"

# Env var names checked before stored files.
ANTHROPIC_ENV = "ANTHROPIC_API_KEY"
OPENAI_ENV = "OPENAI_API_KEY"
LLM_CUSTOM_ENV = "LLM_API_KEY"

ProviderKind = Literal["anthropic", "openai_compatible", "self_hosted"]


# ----------------------------------------------------------------------
# Provider config
# ----------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Provider config persisted to ~/.daydream-scope/agent_config.json."""

    provider: ProviderKind = "anthropic"
    model: str = "claude-sonnet-4-6"
    base_url: str | None = None  # Optional override for any provider
    # The actual API key is NOT stored here; it's resolved at call time from
    # env var or disk.

    def to_json(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> AgentConfig:
        return cls(
            provider=data.get("provider", "anthropic"),
            model=data.get("model", "claude-sonnet-4-6"),
            base_url=data.get("base_url"),
        )


def _config_path() -> Path:
    return Path(AGENT_CONFIG_FILE).expanduser().resolve()


def load_agent_config() -> AgentConfig:
    path = _config_path()
    if not path.exists():
        return AgentConfig()
    try:
        return AgentConfig.from_json(json.loads(path.read_text()))
    except Exception as e:
        logger.warning(f"Failed to load agent config, using defaults: {e}")
        return AgentConfig()


def save_agent_config(cfg: AgentConfig) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg.to_json(), indent=2))


# ----------------------------------------------------------------------
# Provider API key resolution
# ----------------------------------------------------------------------


def _resolve_key(env_var: str, token_file: str) -> str | None:
    env = os.environ.get(env_var)
    if env:
        return env.strip() or None
    path = Path(token_file).expanduser().resolve()
    if not path.exists():
        return None
    try:
        return path.read_text().strip() or None
    except Exception as e:
        logger.warning(f"Failed to read token file {path}: {e}")
        return None


def _save_key(token_file: str, value: str) -> None:
    path = Path(token_file).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    path.chmod(0o600)


def _delete_key(token_file: str) -> None:
    path = Path(token_file).expanduser().resolve()
    if path.exists():
        path.unlink()


def get_provider_key(provider: ProviderKind) -> str | None:
    if provider == "anthropic":
        return _resolve_key(ANTHROPIC_ENV, ANTHROPIC_TOKEN_FILE)
    if provider == "openai_compatible":
        return _resolve_key(OPENAI_ENV, OPENAI_TOKEN_FILE)
    if provider == "self_hosted":
        return _resolve_key(LLM_CUSTOM_ENV, LLM_CUSTOM_TOKEN_FILE)
    return None


def set_provider_key(provider: ProviderKind, value: str) -> None:
    if provider == "anthropic":
        _save_key(ANTHROPIC_TOKEN_FILE, value)
    elif provider == "openai_compatible":
        _save_key(OPENAI_TOKEN_FILE, value)
    elif provider == "self_hosted":
        _save_key(LLM_CUSTOM_TOKEN_FILE, value)


def delete_provider_key(provider: ProviderKind) -> None:
    if provider == "anthropic":
        _delete_key(ANTHROPIC_TOKEN_FILE)
    elif provider == "openai_compatible":
        _delete_key(OPENAI_TOKEN_FILE)
    elif provider == "self_hosted":
        _delete_key(LLM_CUSTOM_TOKEN_FILE)


def get_provider_key_source(provider: ProviderKind) -> str | None:
    """Return "env_var" | "stored" | None."""
    env_var, token_file = {
        "anthropic": (ANTHROPIC_ENV, ANTHROPIC_TOKEN_FILE),
        "openai_compatible": (OPENAI_ENV, OPENAI_TOKEN_FILE),
        "self_hosted": (LLM_CUSTOM_ENV, LLM_CUSTOM_TOKEN_FILE),
    }[provider]
    if os.environ.get(env_var):
        return "env_var"
    if Path(token_file).expanduser().exists():
        return "stored"
    return None


# ----------------------------------------------------------------------
# Workflow proposal
# ----------------------------------------------------------------------


@dataclass
class WorkflowProposal:
    id: str
    graph: dict
    graph_hash_at_propose: str
    rationale: str
    pipeline_load_params: dict = field(default_factory=dict)
    input_mode: str = "video"
    diff: dict = field(default_factory=dict)
    approved: bool = False
    decision_feedback: str | None = None  # e.g. rejection reason


# ----------------------------------------------------------------------
# Agent session
# ----------------------------------------------------------------------


# Messages follow Anthropic's shape: {role: "user"|"assistant", content: [blocks]}.
# Blocks: {"type": "text", "text": "..."}, {"type": "tool_use", "id", "name", "input"},
# {"type": "tool_result", "tool_use_id", "content": [...]}.
Message = dict[str, Any]


@dataclass
class AgentSession:
    id: str
    created_at: float
    last_activity: float
    config_snapshot: AgentConfig
    messages: list[Message] = field(default_factory=list)
    pending_proposal: WorkflowProposal | None = None

    def touch(self) -> None:
        self.last_activity = time.time()


class AgentSessionStore:
    """In-memory store with background TTL eviction (1 hour idle)."""

    def __init__(self, idle_ttl_seconds: float = 3600.0) -> None:
        self._sessions: dict[str, AgentSession] = {}
        self._idle_ttl = idle_ttl_seconds
        self._lock = asyncio.Lock()
        self._janitor_task: asyncio.Task | None = None

    def start_janitor(self) -> None:
        if self._janitor_task is None or self._janitor_task.done():
            self._janitor_task = asyncio.create_task(self._janitor_loop())

    async def stop_janitor(self) -> None:
        if self._janitor_task is not None:
            self._janitor_task.cancel()
            try:
                await self._janitor_task
            except (asyncio.CancelledError, Exception):
                pass
            self._janitor_task = None

    async def _janitor_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(60.0)
                now = time.time()
                async with self._lock:
                    stale = [
                        sid
                        for sid, s in self._sessions.items()
                        if now - s.last_activity > self._idle_ttl
                    ]
                    for sid in stale:
                        logger.info(f"Evicting idle agent session {sid}")
                        del self._sessions[sid]
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning(f"Agent session janitor error: {e}")

    async def create(self, config: AgentConfig) -> AgentSession:
        session = AgentSession(
            id=f"agent_{uuid.uuid4().hex[:12]}",
            created_at=time.time(),
            last_activity=time.time(),
            config_snapshot=config,
        )
        async with self._lock:
            self._sessions[session.id] = session
        return session

    async def get(self, session_id: str) -> AgentSession | None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.touch()
            return session

    async def get_or_create(
        self, session_id: str | None, config: AgentConfig
    ) -> AgentSession:
        if session_id:
            existing = await self.get(session_id)
            if existing is not None:
                return existing
        return await self.create(config)

    async def delete(self, session_id: str) -> bool:
        async with self._lock:
            return self._sessions.pop(session_id, None) is not None

    async def list(self) -> list[dict]:
        async with self._lock:
            return [
                {
                    "id": s.id,
                    "created_at": s.created_at,
                    "last_activity": s.last_activity,
                    "messages": len(s.messages),
                    "has_pending_proposal": s.pending_proposal is not None,
                }
                for s in self._sessions.values()
            ]
