"""LLM provider abstraction.

Three concrete providers:
- AnthropicProvider        — official SDK, default.
- OpenAICompatibleProvider — works with OpenAI, OpenRouter, Groq, together.ai,
                              Fireworks, vLLM, LM Studio, Ollama (OpenAI-shape).
- SelfHostedProvider       — thin subclass of OpenAICompatibleProvider tuned
                              for local endpoints (Ollama default).

All providers yield a uniform ProviderEvent stream so agent_loop doesn't care
which backend is running. Messages follow Anthropic's shape internally; the
OpenAI provider translates to/from OpenAI's chat format at the boundary.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx

from .agent_state import AgentConfig, get_provider_key

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Event types (uniform across providers)
# ----------------------------------------------------------------------


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolUseStart:
    id: str
    name: str


@dataclass
class ToolUseEnd:
    id: str
    name: str
    input: dict


@dataclass
class TurnEnd:
    stop_reason: Literal["end_turn", "tool_use", "max_tokens", "stop_sequence", "error"]
    error_message: str | None = None


ProviderEvent = TextDelta | ToolUseStart | ToolUseEnd | TurnEnd


class ProviderError(RuntimeError):
    pass


# ----------------------------------------------------------------------
# Protocol
# ----------------------------------------------------------------------


class LLMProvider(Protocol):
    async def stream_turn(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        *,
        max_tokens: int = 4096,
    ) -> AsyncIterator[ProviderEvent]: ...

    async def ping(self) -> dict:
        """Trivial round-trip for the Settings "Test connection" button."""
        ...


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


def build_provider(config: AgentConfig) -> LLMProvider:
    key = get_provider_key(config.provider)
    if config.provider == "anthropic":
        if not key:
            raise ProviderError(
                "ANTHROPIC_API_KEY not set. Configure it in Settings → Agent."
            )
        return AnthropicProvider(
            api_key=key,
            model=config.model,
            base_url=config.base_url,
        )
    if config.provider == "openai_compatible":
        if not key:
            raise ProviderError(
                "OpenAI-compatible API key not set. Configure it in Settings → Agent."
            )
        return OpenAICompatibleProvider(
            api_key=key,
            model=config.model,
            base_url=config.base_url or "https://api.openai.com/v1",
        )
    if config.provider == "self_hosted":
        # Self-hosted endpoints (Ollama, vLLM, LM Studio) usually don't
        # require a key — but allow one in case the user fronted their
        # endpoint with an auth proxy.
        return SelfHostedProvider(
            api_key=key or "",
            model=config.model,
            base_url=config.base_url or "http://localhost:11434/v1",
        )
    raise ProviderError(f"unknown provider: {config.provider}")


# ----------------------------------------------------------------------
# Anthropic
# ----------------------------------------------------------------------


class AnthropicProvider:
    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        # Import lazily so the server still starts if anthropic isn't installed
        # for people who only use a self-hosted model.
        from anthropic import AsyncAnthropic

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self._model = model

    async def stream_turn(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        *,
        max_tokens: int = 4096,
    ) -> AsyncIterator[ProviderEvent]:
        # Anthropic's tool-use SDK returns streaming events we forward directly.
        # We rely on the SDK's event iterator instead of re-parsing raw SSE.
        try:
            async with self._client.messages.stream(
                model=self._model,
                system=system,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
            ) as stream:
                current_tool: dict | None = None
                tool_input_buffer: list[str] = []

                async for event in stream:
                    etype = getattr(event, "type", None)

                    if etype == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if (
                            block is not None
                            and getattr(block, "type", None) == "tool_use"
                        ):
                            current_tool = {
                                "id": block.id,
                                "name": block.name,
                            }
                            tool_input_buffer = []
                            yield ToolUseStart(id=block.id, name=block.name)

                    elif etype == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta is None:
                            continue
                        dtype = getattr(delta, "type", None)
                        if dtype == "text_delta":
                            yield TextDelta(text=delta.text)
                        elif dtype == "input_json_delta":
                            tool_input_buffer.append(delta.partial_json)

                    elif etype == "content_block_stop":
                        if current_tool is not None:
                            raw = "".join(tool_input_buffer).strip() or "{}"
                            try:
                                parsed = json.loads(raw)
                            except Exception:
                                parsed = {}
                            yield ToolUseEnd(
                                id=current_tool["id"],
                                name=current_tool["name"],
                                input=parsed,
                            )
                            current_tool = None
                            tool_input_buffer = []

                final = await stream.get_final_message()
                yield TurnEnd(stop_reason=_safe_stop_reason(final.stop_reason))
        except Exception as e:
            logger.exception("Anthropic stream error")
            yield TurnEnd(stop_reason="error", error_message=str(e))

    async def ping(self) -> dict:
        msg = await self._client.messages.create(
            model=self._model,
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with OK."}],
        )
        return {
            "ok": True,
            "provider": "anthropic",
            "model": self._model,
            "sample": _first_text_block(msg),
        }


def _safe_stop_reason(
    raw: Any,
) -> Literal["end_turn", "tool_use", "max_tokens", "stop_sequence", "error"]:
    if raw in ("end_turn", "tool_use", "max_tokens", "stop_sequence"):
        return raw
    return "end_turn"


def _first_text_block(msg: Any) -> str:
    for block in getattr(msg, "content", []) or []:
        if getattr(block, "type", None) == "text":
            return getattr(block, "text", "")
    return ""


# ----------------------------------------------------------------------
# OpenAI-compatible
# ----------------------------------------------------------------------


class OpenAICompatibleProvider:
    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")

    async def stream_turn(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        *,
        max_tokens: int = 4096,
    ) -> AsyncIterator[ProviderEvent]:
        oai_messages = [{"role": "system", "content": system}]
        oai_messages.extend(_anthropic_messages_to_openai(messages))
        oai_tools = [_anthropic_tool_to_openai(t) for t in tools]

        body = {
            "model": self._model,
            "messages": oai_messages,
            "tools": oai_tools,
            "stream": True,
            "max_tokens": max_tokens,
        }
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    json=body,
                    headers=headers,
                ) as resp:
                    if resp.status_code != 200:
                        text = await resp.aread()
                        yield TurnEnd(
                            stop_reason="error",
                            error_message=(
                                f"{resp.status_code}: "
                                f"{text.decode('utf-8', errors='replace')[:500]}"
                            ),
                        )
                        return
                    async for ev in _parse_openai_stream(resp):
                        yield ev
        except Exception as e:
            logger.exception("OpenAI-compatible stream error")
            yield TurnEnd(stop_reason="error", error_message=str(e))

    async def ping(self) -> dict:
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 16,
        }
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions", json=body, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
            sample = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {
                "ok": True,
                "provider": "openai_compatible",
                "model": self._model,
                "base_url": self._base_url,
                "sample": sample,
            }


class SelfHostedProvider(OpenAICompatibleProvider):
    """Same wire protocol, different defaults."""

    pass


# ----------------------------------------------------------------------
# OpenAI stream parsing
# ----------------------------------------------------------------------


async def _parse_openai_stream(resp: httpx.Response) -> AsyncIterator[ProviderEvent]:
    """Parse OpenAI-style SSE and yield uniform ProviderEvents.

    OpenAI tool calls are streamed in pieces: the assistant sends a
    ``tool_calls`` array where each entry has a stable ``index`` and its
    arguments arrive as partial JSON deltas. We buffer per-index until the
    finish_reason lands, then emit ToolUseEnd(s).
    """
    tool_calls_by_index: dict[int, dict] = {}
    saw_any_text = False
    stop_reason: Literal[
        "end_turn", "tool_use", "max_tokens", "stop_sequence", "error"
    ] = "end_turn"

    async for raw_line in resp.aiter_lines():
        if not raw_line:
            continue
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            event = json.loads(payload)
        except Exception:
            continue

        choices = event.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}

        content = delta.get("content")
        if content:
            saw_any_text = True
            yield TextDelta(text=content)

        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            bucket = tool_calls_by_index.setdefault(
                idx, {"id": None, "name": None, "args": []}
            )
            if tc.get("id"):
                bucket["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                bucket["name"] = fn["name"]
                if bucket["id"] is None:
                    # Some servers drop the id; synthesize a stable one.
                    bucket["id"] = f"oai_tool_{idx}"
                yield ToolUseStart(id=bucket["id"], name=bucket["name"])
            if "arguments" in fn:
                bucket["args"].append(fn["arguments"] or "")

        finish = choice.get("finish_reason")
        if finish is not None:
            if finish == "tool_calls":
                stop_reason = "tool_use"
            elif finish == "length":
                stop_reason = "max_tokens"
            elif finish == "stop":
                stop_reason = "end_turn"
            break

    # Emit any buffered tool calls.
    for idx, bucket in sorted(tool_calls_by_index.items()):
        raw = "".join(bucket["args"]).strip() or "{}"
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {}
        tid = bucket["id"] or f"oai_tool_{idx}"
        name = bucket["name"] or ""
        if name:
            yield ToolUseEnd(id=tid, name=name, input=parsed)

    if not saw_any_text and not tool_calls_by_index and stop_reason == "end_turn":
        stop_reason = "end_turn"
    yield TurnEnd(stop_reason=stop_reason)


# ----------------------------------------------------------------------
# Anthropic <-> OpenAI message/tool translation
# ----------------------------------------------------------------------


def _anthropic_tool_to_openai(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema")
            or {"type": "object", "properties": {}},
        },
    }


def _anthropic_messages_to_openai(messages: list[dict]) -> list[dict]:
    """Translate Anthropic-shape messages to OpenAI chat format.

    Anthropic messages are {role, content: [blocks]} where content blocks may
    be text / tool_use (assistant) / tool_result (user). OpenAI splits these:
    - assistant messages with tool_calls  (function-call request)
    - `tool` role messages for each tool_result (one per tool_use_id)

    Images inside tool_result are passed as OpenAI content parts with
    image_url base64 data URIs when present.
    """
    out: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            # User messages can carry either text or tool_result blocks.
            if isinstance(content, str):
                out.append({"role": "user", "content": content})
                continue

            plain_parts: list[Any] = []
            for block in content or []:
                btype = block.get("type")
                if btype == "text":
                    plain_parts.append({"type": "text", "text": block.get("text", "")})
                elif btype == "tool_result":
                    # Convert to a dedicated 'tool' role message.
                    out.append(_tool_result_to_oai_msg(block))
                elif btype == "image":
                    src = block.get("source") or {}
                    if src.get("type") == "base64":
                        data_uri = (
                            f"data:{src.get('media_type', 'image/jpeg')};base64,"
                            f"{src.get('data', '')}"
                        )
                        plain_parts.append(
                            {"type": "image_url", "image_url": {"url": data_uri}}
                        )
            if plain_parts:
                out.append({"role": "user", "content": plain_parts})

        elif role == "assistant":
            if isinstance(content, str):
                out.append({"role": "assistant", "content": content})
                continue

            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for block in content or []:
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input") or {}),
                            },
                        }
                    )
            asst_msg: dict = {"role": "assistant", "content": "".join(text_parts)}
            if tool_calls:
                asst_msg["tool_calls"] = tool_calls
            out.append(asst_msg)

    return out


def _tool_result_to_oai_msg(block: dict) -> dict:
    """Convert an Anthropic tool_result block to an OpenAI 'tool' role msg.

    OpenAI only supports text content in tool messages today; we stringify
    images (they've already been fed back to the model as an earlier image
    block in Anthropic-land, but here we flatten)."""
    result = block.get("content")
    if isinstance(result, str):
        return {
            "role": "tool",
            "tool_call_id": block.get("tool_use_id", ""),
            "content": result,
        }
    parts: list[str] = []
    for item in result or []:
        if isinstance(item, dict):
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif item.get("type") == "image":
                parts.append("[image attached in prior context]")
        else:
            parts.append(str(item))
    return {
        "role": "tool",
        "tool_call_id": block.get("tool_use_id", ""),
        "content": "\n".join(parts) or "(empty)",
    }
