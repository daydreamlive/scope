"""Structural graders for workflow proposals.

Each check is a pure function ``(graph, arg) -> CheckResult`` registered in
:data:`CHECKS`. The YAML case format references checks by name (see
:mod:`evals.case`), so adding a new check is: write a function, register it,
reference it from a case file.

We intentionally favor simple boolean-with-reason checks over complex
"structural equivalence" comparisons — the three canonical failure modes
we're trying to catch (missing VACE wire, unwired prompt, missing slider
for a called-out parameter) are all detectable with trivial traversals.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Re-use the backend validator so a regression there immediately shows up
# here too.
from scope.server.agent_tool_impls import (
    _derive_pipeline_handles,
    _validate_proposal,
)


@dataclass
class CheckResult:
    ok: bool
    detail: str

    @classmethod
    def ok_(cls, detail: str = "") -> CheckResult:
        return cls(ok=True, detail=detail)

    @classmethod
    def fail(cls, detail: str) -> CheckResult:
        return cls(ok=False, detail=detail)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pipeline_ids(graph: dict) -> list[str]:
    return [
        n["pipeline_id"]
        for n in graph.get("nodes", []) or []
        if n.get("type") == "pipeline" and n.get("pipeline_id")
    ]


def _pipeline_node_ids(graph: dict) -> set[str]:
    return {
        n["id"]
        for n in graph.get("nodes", []) or []
        if n.get("type") == "pipeline" and n.get("id")
    }


def _ui_nodes(graph: dict) -> list[dict]:
    return (graph.get("ui_state") or {}).get("nodes") or []


def _ui_edges(graph: dict) -> list[dict]:
    return (graph.get("ui_state") or {}).get("edges") or []


def _ui_node_type(graph: dict, node_id: str) -> str | None:
    for n in _ui_nodes(graph):
        if n.get("id") == node_id:
            return n.get("type")
    return None


# ---------------------------------------------------------------------------
# Checks — expect / forbid semantics are both `ok=True` means "assertion
# holds". The runner inverts for forbid.
# ---------------------------------------------------------------------------


def pipelines_equal(graph: dict, arg: Any) -> CheckResult:
    want = set(arg or [])
    got = set(_pipeline_ids(graph))
    if got == want:
        return CheckResult.ok_(f"pipelines={sorted(got)}")
    missing = sorted(want - got)
    extra = sorted(got - want)
    parts = []
    if missing:
        parts.append(f"missing={missing}")
    if extra:
        parts.append(f"extra={extra}")
    return CheckResult.fail(", ".join(parts) or f"got={sorted(got)}")


def pipelines_include(graph: dict, arg: Any) -> CheckResult:
    want = set(arg or [])
    got = set(_pipeline_ids(graph))
    missing = sorted(want - got)
    if missing:
        return CheckResult.fail(f"missing={missing}, got={sorted(got)}")
    return CheckResult.ok_(f"got={sorted(got)}")


def pipelines_count_at_least(graph: dict, arg: Any) -> CheckResult:
    """Assert at least N pipeline nodes exist, without pinning which ones.

    Useful for vague prompts where the agent gets to pick the pipeline.
    """
    min_count = int(arg)
    got = _pipeline_ids(graph)
    if len(got) >= min_count:
        return CheckResult.ok_(f"{len(got)} pipeline(s): {sorted(set(got))}")
    return CheckResult.fail(f"need >= {min_count} pipeline(s), got {len(got)}")


def lora_count_at_least(graph: dict, arg: Any) -> CheckResult:
    min_count = int(arg)
    # Two reasonable places: a dedicated `lora` UI node, or a `lora` node
    # with multiple entries in data.loras[]. Sum across both.
    total = 0
    lora_node_count = 0
    for n in _ui_nodes(graph):
        if n.get("type") == "lora":
            lora_node_count += 1
            inner = (n.get("data") or {}).get("loras") or []
            total += max(1, len(inner))
    if total >= min_count:
        return CheckResult.ok_(
            f"{total} lora entr(ies) across {lora_node_count} node(s)"
        )
    return CheckResult.fail(
        f"need >= {min_count}, found {total} across {lora_node_count} lora node(s)"
    )


def no_validator_errors(graph: dict, _arg: Any) -> CheckResult:
    """Re-run the backend validator. Getting a proposal at all implies this
    passed once, but we re-assert so a silent regression in the validator is
    still surfaced by the harness."""
    # Build a minimal handles lookup so the validator can check pipeline
    # targets. For pipelines we don't know, fall back to an empty shape;
    # validator will treat that as "unknown" and only report errors on
    # clearly-malformed edges rather than on handle-existence.
    handles: dict[str, dict] = {}
    for pid in set(_pipeline_ids(graph)):
        # We don't have the live registry here; synthesize a permissive shape
        # by deriving from an empty schema. Unknown-handle checks still fire
        # for bad prefixes but not for handle names we can't verify.
        handles[pid] = _derive_pipeline_handles(
            pid,
            {
                "supports_prompts": True,
                "supports_vace": True,
                "supports_lora": True,
                "produces_video": True,
                "config_schema": {"properties": {}},
            },
        )
    issues = _validate_proposal(graph, handles)
    errs = [i for i in issues if i.get("severity") == "error"]
    if errs:
        first = errs[0].get("message", "")
        return CheckResult.fail(f"{len(errs)} validator error(s); first: {first}")
    return CheckResult.ok_("0 validator errors")


# ---------------------------------------------------------------------------
# wire_present — one check with a `kind` discriminator.
# ---------------------------------------------------------------------------


_VALUE_SOURCE_TYPES = {
    "slider",
    "knobs",
    "primitive",
    "trigger",
    "control",
    "subgraph",
    "math",
}


def _edges_into(
    graph: dict, target_id: str, target_handle: str | None = None
) -> list[dict]:
    out = []
    for e in _ui_edges(graph):
        if e.get("target") != target_id:
            continue
        if target_handle is not None and e.get("targetHandle") != target_handle:
            continue
        out.append(e)
    return out


def _edges_into_any_pipeline(graph: dict, target_handle: str) -> list[dict]:
    pipe_ids = _pipeline_node_ids(graph)
    out = []
    for e in _ui_edges(graph):
        if e.get("target") in pipe_ids and e.get("targetHandle") == target_handle:
            out.append(e)
    return out


def wire_present(graph: dict, arg: Any) -> CheckResult:
    if not isinstance(arg, dict) or "kind" not in arg:
        return CheckResult.fail(f"wire_present needs {{kind: ...}}, got {arg!r}")
    kind = arg["kind"]

    if kind == "slider_to_pipeline_param":
        target_handle = arg.get("target_handle")
        if not target_handle:
            return CheckResult.fail("slider_to_pipeline_param needs target_handle")
        hits = _edges_into_any_pipeline(graph, target_handle)
        if not hits:
            return CheckResult.fail(
                f"no ui_state edge targets a pipeline's {target_handle}"
            )
        # Source must be a value-producing UI node type.
        for e in hits:
            src_t = _ui_node_type(graph, e.get("source"))
            if src_t in _VALUE_SOURCE_TYPES:
                return CheckResult.ok_(f"{src_t}({e.get('source')}) -> {target_handle}")
        return CheckResult.fail(
            f"edge(s) into {target_handle} exist but none originate from "
            f"a value-producing node (types: {sorted(_VALUE_SOURCE_TYPES)})"
        )

    if kind == "vace_to_pipeline":
        hits = _edges_into_any_pipeline(graph, "param:__vace")
        if not hits:
            return CheckResult.fail("no edge targets pipeline's param:__vace")
        for e in hits:
            if _ui_node_type(graph, e.get("source")) == "vace":
                return CheckResult.ok_(f"vace({e.get('source')}) -> param:__vace")
        return CheckResult.fail(
            "param:__vace edge exists but source is not a vace node"
        )

    if kind == "image_to_vace":
        vace_handles = {"param:ref_image", "param:first_frame", "param:last_frame"}
        for e in _ui_edges(graph):
            tgt_t = _ui_node_type(graph, e.get("target"))
            if tgt_t == "vace" and e.get("targetHandle") in vace_handles:
                src_t = _ui_node_type(graph, e.get("source"))
                # Accept either a dedicated 'image' node or a generic value
                # source (primitive holding a path).
                if src_t in {"image"} | _VALUE_SOURCE_TYPES:
                    return CheckResult.ok_(
                        f"{src_t}({e.get('source')}) -> vace.{e.get('targetHandle')}"
                    )
        return CheckResult.fail(
            "no edge into a vace node's ref_image/first_frame/last_frame"
        )

    if kind == "prompt_to_pipeline":
        hits = _edges_into_any_pipeline(graph, "param:__prompt")
        if hits:
            return CheckResult.ok_(f"{len(hits)} edge(s) -> param:__prompt")
        return CheckResult.fail("no edge targets pipeline's param:__prompt")

    if kind == "lora_to_pipeline":
        hits = _edges_into_any_pipeline(graph, "param:__loras")
        if hits:
            return CheckResult.ok_(f"{len(hits)} edge(s) -> param:__loras")
        return CheckResult.fail("no edge targets pipeline's param:__loras")

    if kind == "pipeline_to_record":
        # Any edge from a pipeline's stream output into a record node.
        pipe_ids = _pipeline_node_ids(graph)
        for e in _ui_edges(graph):
            if e.get("source") not in pipe_ids:
                continue
            if _ui_node_type(graph, e.get("target")) != "record":
                continue
            # Source handle should be a stream (video). Accept any stream: prefix.
            sh = e.get("sourceHandle") or ""
            if isinstance(sh, str) and sh.startswith("stream:"):
                return CheckResult.ok_(
                    f"pipeline({e.get('source')}) -> record({e.get('target')}) via {sh}"
                )
        return CheckResult.fail(
            "no ui_state edge wires a pipeline stream output into a record node"
        )

    if kind == "prompt_list_to_pipeline":
        # prompt_list node's param:prompt output → pipeline's param:__prompt.
        hits = _edges_into_any_pipeline(graph, "param:__prompt")
        if not hits:
            return CheckResult.fail("no edge targets pipeline's param:__prompt")
        for e in hits:
            if _ui_node_type(graph, e.get("source")) == "prompt_list":
                return CheckResult.ok_(
                    f"prompt_list({e.get('source')}) -> param:__prompt"
                )
        return CheckResult.fail(
            "param:__prompt edge exists but source is not a prompt_list node"
        )

    if kind == "trigger_to_prompt_list":
        # Some value source → prompt_list's param:trigger (or param:cycle).
        accepted = {"param:trigger", "param:cycle"}
        for e in _ui_edges(graph):
            if _ui_node_type(graph, e.get("target")) != "prompt_list":
                continue
            if e.get("targetHandle") not in accepted:
                continue
            src_t = _ui_node_type(graph, e.get("source"))
            if src_t in _VALUE_SOURCE_TYPES:
                return CheckResult.ok_(
                    f"{src_t}({e.get('source')}) -> prompt_list.{e.get('targetHandle')}"
                )
        return CheckResult.fail(
            "no edge from a value-producing source into a prompt_list's "
            "param:trigger or param:cycle"
        )

    return CheckResult.fail(f"unknown wire_present kind: {kind!r}")


def node_present(graph: dict, arg: Any) -> CheckResult:
    """Assert at least N UI nodes of a given type exist.

    arg: ``{type: "record", count: 1, min_items: 5}``
    - ``type`` (required) — ui_state node type.
    - ``count`` (default 1) — minimum number of nodes of that type.
    - ``min_items`` (optional) — if set AND type=="prompt_list", at least one
      such node must have ``data.promptListItems`` of length ≥ min_items.
    """
    if not isinstance(arg, dict) or "type" not in arg:
        return CheckResult.fail(f"node_present needs {{type: ...}}, got {arg!r}")
    want_type = arg["type"]
    want_count = int(arg.get("count", 1))
    min_items = arg.get("min_items")

    nodes = [n for n in _ui_nodes(graph) if n.get("type") == want_type]
    if len(nodes) < want_count:
        return CheckResult.fail(
            f"need >= {want_count} node(s) of type {want_type!r}, got {len(nodes)}"
        )

    if min_items is not None:
        # Look for at least one node whose item list is long enough.
        threshold = int(min_items)
        max_seen = 0
        for n in nodes:
            items = (n.get("data") or {}).get("promptListItems") or []
            if isinstance(items, list):
                max_seen = max(max_seen, len(items))
        if max_seen < threshold:
            return CheckResult.fail(
                f"{want_type} exists but longest promptListItems is {max_seen}, "
                f"need >= {threshold}"
            )
        return CheckResult.ok_(
            f"{len(nodes)} {want_type} node(s); longest list has {max_seen} item(s)"
        )

    return CheckResult.ok_(f"{len(nodes)} {want_type} node(s)")


# ---------------------------------------------------------------------------
# forbid checks
# ---------------------------------------------------------------------------


def bad_handle_prefix(graph: dict, arg: Any) -> CheckResult:
    """Forbid check: returns ok=True if NO edge uses the given prefix."""
    prefix = str(arg)
    for e in _ui_edges(graph):
        for side in ("sourceHandle", "targetHandle"):
            h = e.get(side)
            if isinstance(h, str) and h.startswith(prefix):
                return CheckResult.fail(
                    f"edge {e.get('id', '?')} {side}={h!r} starts with forbidden {prefix!r}"
                )
    return CheckResult.ok_(f"no edge handle starts with {prefix!r}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


CHECKS: dict[str, Callable[[dict, Any], CheckResult]] = {
    "pipelines_equal": pipelines_equal,
    "pipelines_include": pipelines_include,
    "pipelines_count_at_least": pipelines_count_at_least,
    "lora_count_at_least": lora_count_at_least,
    "no_validator_errors": no_validator_errors,
    "wire_present": wire_present,
    "node_present": node_present,
    "bad_handle_prefix": bad_handle_prefix,
}


def run_check(name: str, graph: dict, arg: Any) -> CheckResult:
    fn = CHECKS.get(name)
    if fn is None:
        return CheckResult.fail(f"unknown check: {name!r}")
    try:
        return fn(graph, arg)
    except Exception as e:  # defensive — a buggy check must not kill the run
        return CheckResult.fail(f"{type(e).__name__} in check {name}: {e}")
