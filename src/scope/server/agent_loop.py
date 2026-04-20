"""Agent loop — orchestrates provider turns, tool dispatch, and SSE emission.

Design notes:
- ``run_turn`` is an async generator that yields SSE-ready dict payloads.
  The FastAPI route converts these to ``"event: ...\\ndata: ...\\n\\n"``.
- Tool results are appended to ``session.messages`` in Anthropic shape so the
  next provider call has full context.
- Vision: when ``capture_frame`` returns a base64 JPEG, we wrap it as an image
  content block inside the tool_result so multimodal models see it.
- Proposals: ``propose_workflow`` returns normally, the loop emits a
  ``workflow_proposal`` SSE event to the frontend, and the model's turn
  continues as usual. Best practice (reinforced in the system prompt) is for
  the model to stop after proposing so the user can approve/reject. On
  approval, the frontend starts a new turn with an auto-generated user
  message, which lets the model call ``apply_workflow``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI

from .agent_providers import (
    LLMProvider,
    ProviderError,
    TextDelta,
    ToolUseEnd,
    ToolUseStart,
    TurnEnd,
    build_provider,
)
from .agent_state import AgentSession
from .agent_tool_impls import TOOL_METHODS, AgentTools, build_tool_specs

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are the Scope Agent, embedded in Daydream Scope — a real-time generative \
AI video tool. Your job is to translate the user's intent into working graphs \
and to iteratively tune parameters by observing output. Favor thoughtful \
defaults over excessive questioning.

CORE PRINCIPLES

1. Introspect, never guess. Call list_pipelines and get_pipeline_schema \
before proposing any graph or parameter change — pipeline authors add new \
fields without warning. Never set a parameter you haven't verified in the \
current schema.

2. Prefer composition over reinvention. Use list_blueprints to find \
pre-built fragments (prompt switcher, LFO, timed cycler, etc.) and graft \
them into proposals. Only hand-roll nodes when no blueprint fits.

3. Propose first, confirm on approval. Structural graph changes MUST go \
through propose_workflow. After calling propose_workflow, write a short \
text summary and stop; the UI will render Approve/Reject. If the user \
approves, a new turn will start where you should call apply_workflow with \
the proposal_id and expected_graph_hash. Important: approval writes the \
graph to the canvas automatically — apply_workflow only confirms it. \
Never call start_session, load_pipeline, or any session-starting tool on \
behalf of the user after a proposal. The user presses Play.

4. Runtime tweaks apply immediately. Prompts, noise_scale, LoRA weights, \
VACE scale, etc. are fair game to auto-apply via update_parameters without \
asking permission — that's why they're runtime.

5. See before you tune. When the user reports a visual issue (\"not \
recognizing depth\", \"too noisy\", \"wrong style\"), call capture_frame \
first so you have concrete evidence before adjusting.

6. Be terse and concrete. Surface what you did, the key parameters, and \
what to try next if the change doesn't stick. Skip filler.

GRAPH SHAPE

Minimal backend graph (GraphConfig):
  {
    \"nodes\": [
      {\"id\": \"input\", \"type\": \"source\", \"source_mode\": \"camera|video_file|ndi|syphon|spout\"},
      {\"id\": \"pipe\", \"type\": \"pipeline\", \"pipeline_id\": \"longlive\"},
      {\"id\": \"output\", \"type\": \"sink\"}
    ],
    \"edges\": [
      {\"from\": \"input\", \"from_port\": \"video\", \"to_node\": \"pipe\", \"to_port\": \"video\", \"kind\": \"stream\"},
      {\"from\": \"pipe\", \"from_port\": \"video\", \"to_node\": \"output\", \"to_port\": \"video\", \"kind\": \"stream\"}
    ]
  }

Record nodes: add {\"id\": \"rec\", \"type\": \"record\"} and fan out from \
pipeline output with a stream edge. Multiple sinks are supported.

STYLE
- One or two sentences when confirming a tool outcome.
- Avoid restating the user's request.
- Don't apologize unless something actually failed.
"""


def _format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def run_turn(
    app: FastAPI,
    session: AgentSession,
    user_message: str,
    *,
    is_system_continuation: bool = False,
) -> AsyncIterator[str]:
    """Run one agent turn (may contain multiple provider calls due to tools).

    Yields raw SSE-formatted strings ready to write to the response stream.

    Args:
        app: the FastAPI app (for in-process httpx ASGITransport).
        session: the agent session to mutate.
        user_message: the human-authored user message OR a system
            continuation message (e.g. "[user approved proposal X]").
        is_system_continuation: True when the "user" message is actually a
            synthetic continuation (approval/rejection). Does not change LLM
            behavior but is tagged so the UI can hide it if it wants.
    """
    # 1. Provider build
    try:
        provider: LLMProvider = build_provider(session.config_snapshot)
    except ProviderError as e:
        yield _format_sse("error", {"message": str(e)})
        yield _format_sse("turn_end", {"stop_reason": "error"})
        return

    # 2. Append user message to session history
    session.messages.append({"role": "user", "content": user_message})
    session.touch()
    yield _format_sse(
        "user_message_appended",
        {"session_id": session.id, "is_continuation": is_system_continuation},
    )

    # 3. Tool loop
    tool_specs = build_tool_specs()

    async with AgentTools(app=app, session=session) as tools:
        # Inner loop: keep calling provider while it asks for tools.
        # Cap iterations to avoid runaway loops. Tuned generously:
        # a single "build a workflow" turn can legitimately chain
        # inspect state → list pipelines → read schema → check
        # blueprints → propose → (apply → verify) and each of those
        # may itself be a few tool calls.
        MAX_TOOL_ROUNDS = 40
        for _round in range(MAX_TOOL_ROUNDS):
            # Accumulate assistant blocks (text + tool_use) so we can append
            # them to session.messages as one assistant message.
            assistant_blocks: list[dict] = []
            tool_uses_this_round: list[dict] = []
            text_buffer: list[str] = []

            stop_reason = "end_turn"
            error_message: str | None = None

            try:
                async for event in provider.stream_turn(
                    SYSTEM_PROMPT, session.messages, tool_specs
                ):
                    if isinstance(event, TextDelta):
                        text_buffer.append(event.text)
                        yield _format_sse("text_delta", {"delta": event.text})

                    elif isinstance(event, ToolUseStart):
                        # Flush any pending text to assistant_blocks.
                        if text_buffer:
                            assistant_blocks.append(
                                {"type": "text", "text": "".join(text_buffer)}
                            )
                            text_buffer = []
                        yield _format_sse(
                            "tool_call_start",
                            {"id": event.id, "name": event.name},
                        )

                    elif isinstance(event, ToolUseEnd):
                        tool_uses_this_round.append(
                            {"id": event.id, "name": event.name, "input": event.input}
                        )
                        assistant_blocks.append(
                            {
                                "type": "tool_use",
                                "id": event.id,
                                "name": event.name,
                                "input": event.input,
                            }
                        )
                        yield _format_sse(
                            "tool_call_input",
                            {
                                "id": event.id,
                                "name": event.name,
                                "input": event.input,
                            },
                        )

                    elif isinstance(event, TurnEnd):
                        stop_reason = event.stop_reason
                        error_message = event.error_message
                        # flush tail text
                        if text_buffer:
                            assistant_blocks.append(
                                {"type": "text", "text": "".join(text_buffer)}
                            )
                            text_buffer = []
                        break
            except Exception as e:
                logger.exception("Provider loop error")
                yield _format_sse("error", {"message": str(e)})
                yield _format_sse("turn_end", {"stop_reason": "error"})
                return

            # Append the assistant turn to history (if non-empty).
            if assistant_blocks:
                session.messages.append(
                    {"role": "assistant", "content": assistant_blocks}
                )

            # If the provider errored, surface and stop.
            if stop_reason == "error":
                yield _format_sse(
                    "error", {"message": error_message or "provider error"}
                )
                yield _format_sse("turn_end", {"stop_reason": "error"})
                return

            # If no tool calls requested, we're done.
            if not tool_uses_this_round:
                yield _format_sse("turn_end", {"stop_reason": stop_reason})
                return

            # Dispatch tools, collect tool_result blocks.
            tool_result_blocks: list[dict] = []
            for call in tool_uses_this_round:
                result_block = await _dispatch_tool(tools, call)
                # Emit a user-visible summary for the UI.
                yield _format_sse(
                    "tool_call_result",
                    {
                        "id": call["id"],
                        "name": call["name"],
                        "ok": result_block.get("_ok", True),
                        "summary": result_block.get("_summary", ""),
                    },
                )
                # Emit workflow_proposal if propose_workflow succeeded.
                if (
                    call["name"] == "propose_workflow"
                    and session.pending_proposal is not None
                    and session.pending_proposal.id == result_block.get("_proposal_id")
                ):
                    pp = session.pending_proposal
                    yield _format_sse(
                        "workflow_proposal",
                        {
                            "proposal_id": pp.id,
                            "graph": pp.graph,
                            "graph_hash": pp.graph_hash_at_propose,
                            "rationale": pp.rationale,
                            "pipelines_to_load": sorted(
                                {
                                    n.get("pipeline_id")
                                    for n in pp.graph.get("nodes", [])
                                    if n.get("type") == "pipeline"
                                    and n.get("pipeline_id")
                                }
                            ),
                            "diff": pp.diff,
                        },
                    )
                tool_result_blocks.append(result_block["_anthropic_block"])

            # Append the tool_result message and continue the loop.
            session.messages.append({"role": "user", "content": tool_result_blocks})

        # Hit iteration cap — force stop.
        yield _format_sse(
            "error",
            {"message": f"exceeded {MAX_TOOL_ROUNDS} tool rounds; stopping"},
        )
        yield _format_sse("turn_end", {"stop_reason": "max_tokens"})


async def _dispatch_tool(tools: AgentTools, call: dict[str, Any]) -> dict:
    """Call the named tool, return a dict with:
    - _anthropic_block: the tool_result content block to feed back to the LLM
    - _summary: short human summary for the UI
    - _ok: bool for UI status
    - _proposal_id: only set for propose_workflow (used to emit SSE)
    """
    name = call["name"]
    method_name = TOOL_METHODS.get(name)
    tool_use_id = call["id"]

    if method_name is None:
        return _error_result(tool_use_id, f"unknown tool: {name}")

    method = getattr(tools, method_name, None)
    if method is None:
        return _error_result(tool_use_id, f"tool not implemented: {name}")

    try:
        result = await method(**(call.get("input") or {}))
    except TypeError as e:
        return _error_result(tool_use_id, f"bad arguments: {e}")
    except Exception as e:
        logger.exception(f"tool {name} failed")
        return _error_result(tool_use_id, f"{type(e).__name__}: {e}")

    # capture_frame → multimodal tool_result.
    if name == "capture_frame" and isinstance(result, dict) and "base64" in result:
        text_summary = f"Captured frame ({result.get('size_bytes', 0)} bytes)" + (
            f" from sink '{result['sink_node_id']}'"
            if result.get("sink_node_id")
            else ""
        )
        return {
            "_anthropic_block": {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": [
                    {"type": "text", "text": text_summary},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": result.get("media_type", "image/jpeg"),
                            "data": result["base64"],
                        },
                    },
                ],
            },
            "_summary": text_summary,
            "_ok": True,
        }

    # propose_workflow bookkeeping.
    proposal_id = None
    if name == "propose_workflow" and isinstance(result, dict) and result.get("ok"):
        proposal_id = result.get("proposal_id")

    summary = _summarize_result(name, result)
    return {
        "_anthropic_block": {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": [{"type": "text", "text": json.dumps(result, default=str)}],
            **(
                {"is_error": True}
                if isinstance(result, dict) and result.get("error")
                else {}
            ),
        },
        "_summary": summary,
        "_ok": not (isinstance(result, dict) and result.get("error")),
        "_proposal_id": proposal_id,
    }


def _summarize_result(name: str, result: Any) -> str:
    if not isinstance(result, dict):
        return str(result)[:200]
    if result.get("error"):
        return f"error: {result['error']}"
    if name == "list_pipelines":
        return f"Found {result.get('count', 0)} pipelines"
    if name == "list_blueprints":
        return f"Found {result.get('count', 0)} blueprints"
    if name == "get_pipeline_schema":
        return f"Schema for pipeline (fields: {len(result.get('config_schema', {}) or {})})"
    if name == "update_parameters":
        applied = result.get("applied") or []
        return f"Applied {len(applied)} parameter(s): {', '.join(applied[:5])}"
    if name == "propose_workflow":
        return (
            f"Proposed workflow {result.get('proposal_id', '?')}: "
            f"{result.get('rationale', '')[:100]}"
        )
    if name == "apply_workflow":
        return "Workflow applied" if result.get("ok") else "Apply failed"
    if name == "capture_frame":
        return f"Captured {result.get('size_bytes', 0)} bytes"
    return json.dumps(result, default=str)[:200]


def _error_result(tool_use_id: str, message: str) -> dict:
    return {
        "_anthropic_block": {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": [{"type": "text", "text": message}],
            "is_error": True,
        },
        "_summary": message,
        "_ok": False,
    }


# ----------------------------------------------------------------------
# Decision handling (approve/reject proposals)
# ----------------------------------------------------------------------


def build_decision_continuation_message(
    approved: bool,
    proposal_id: str,
    graph_hash: str,
    reason: str | None = None,
) -> str:
    """Build a synthetic "user message" that kicks off the next turn after a
    proposal decision. The model reads this, then either calls
    apply_workflow (on approval) or reconsiders (on rejection).
    """
    if approved:
        return (
            f"[System] The user approved proposal {proposal_id}. The graph "
            f"has already been written to the canvas. Call apply_workflow "
            f'with proposal_id="{proposal_id}" and '
            f'expected_graph_hash="{graph_hash}" to clear the pending '
            f"proposal, then briefly confirm (one sentence) that the graph "
            f"is now on the canvas and the user can press Play. Do NOT "
            f"call load_pipeline, session start, or any other tool."
        )
    return (
        f"[System] The user rejected proposal {proposal_id}. "
        f"Reason: {reason or 'no reason given'}. "
        f"Ask a clarifying question or propose a revised workflow."
    )
