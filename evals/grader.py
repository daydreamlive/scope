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


# Top-level (backend) graph helpers. The backend graph only accepts these
# four node types; anything else lives in ui_state (see SYSTEM_PROMPT's
# GRAPH SHAPE section).
_TOP_LEVEL_TYPES = {"source", "pipeline", "sink", "record"}


def _top_level_nodes(graph: dict) -> list[dict]:
    return graph.get("nodes") or []


def _top_level_edges(graph: dict) -> list[dict]:
    return graph.get("edges") or []


def _top_level_node_type(graph: dict, node_id: str) -> str | None:
    for n in _top_level_nodes(graph):
        if n.get("id") == node_id:
            return n.get("type")
    return None


def _nodes_of_type(graph: dict, want_type: str) -> list[dict]:
    """Return all nodes of ``want_type`` from wherever they legally live.

    Top-level kinds (source/pipeline/sink/record) are searched in the
    backend graph; everything else (slider, vace, prompt_list, ...)
    lives in ui_state. This matches the producer-side split enforced by
    the SYSTEM_PROMPT + backend validator.
    """
    if want_type in _TOP_LEVEL_TYPES:
        return [n for n in _top_level_nodes(graph) if n.get("type") == want_type]
    return [n for n in _ui_nodes(graph) if n.get("type") == want_type]


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
        # A record node is a top-level node type; the canonical wiring is a
        # top-level stream edge `pipeline -> record`. We also accept a
        # ui_state-shaped edge from a pipeline to a record node, since
        # either is permissible at the schema level.
        pipe_ids = _pipeline_node_ids(graph)

        # Top-level form: {"from": <pipe>, "to_node": <rec>, "kind": "stream"}.
        for e in _top_level_edges(graph):
            if e.get("from") not in pipe_ids:
                continue
            if _top_level_node_type(graph, e.get("to_node")) != "record":
                continue
            if e.get("kind") != "stream":
                continue
            return CheckResult.ok_(
                f"pipeline({e.get('from')}) -> record({e.get('to_node')}) "
                f"(top-level stream edge)"
            )

        # ui_state form (less common but legal for composed graphs).
        for e in _ui_edges(graph):
            if e.get("source") not in pipe_ids:
                continue
            if _ui_node_type(graph, e.get("target")) != "record":
                continue
            sh = e.get("sourceHandle") or ""
            if isinstance(sh, str) and sh.startswith("stream:"):
                return CheckResult.ok_(
                    f"pipeline({e.get('source')}) -> record({e.get('target')}) "
                    f"via ui_state {sh}"
                )
        return CheckResult.fail(
            "no stream edge (top-level or ui_state) wires a pipeline "
            "output into a record node"
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
    """Assert at least N nodes of a given type exist.

    arg: ``{type: "record", count: 1, min_items: 5}``
    - ``type`` (required) — node type. Top-level kinds
      (source/pipeline/sink/record) are searched in the backend graph;
      everything else (slider, vace, prompt_list, ...) in ui_state.
    - ``count`` (default 1) — minimum number of nodes of that type.
    - ``min_items`` (optional) — if set AND type=="prompt_list", at least one
      such node must have ``data.promptListItems`` of length ≥ min_items.
    """
    if not isinstance(arg, dict) or "type" not in arg:
        return CheckResult.fail(f"node_present needs {{type: ...}}, got {arg!r}")
    want_type = arg["type"]
    want_count = int(arg.get("count", 1))
    min_items = arg.get("min_items")

    nodes = _nodes_of_type(graph, want_type)
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


def overlapping_nodes(graph: dict, _arg: Any) -> CheckResult:
    """Forbid check: no two nodes on the canvas may overlap.

    Observed failure: the agent picks UI-node positions like (0,0), (0,80),
    (320,40) that look "neat" in isolation but collide with the frontend's
    top-level auto-layout strip (sources at x=50, pipelines at x=350, sinks
    at x=650, records at x=950). The server-side ``_reflow_ui_nodes`` should
    catch this and reassign, so this check is a regression detector: if it
    ever fires in an eval, either the agent is producing new layout patterns
    reflow doesn't cover OR reflow has a bug.

    We use the same bounding-box logic as ``_reflow_ui_nodes``: UI nodes are
    240×140 (280 tall for image/vace/subgraph), top-level nodes are the
    200×60 that ``graphConfigToFlow`` drops at x=50/350/650/950, row-spaced
    by 160 starting at y=50.
    """
    # Mirror the constants used by the server-side reflow (keeping them
    # duplicated here is intentional — if either set drifts, the eval is
    # exactly the place we want to catch it).
    FE_START_X = 50
    FE_START_Y = 50
    FE_COLUMN_GAP = 300
    FE_ROW_GAP = 100
    FE_NODE_W = 200
    FE_NODE_H = 60

    UI_NODE_W = 240
    UI_NODE_H_DEFAULT = 140
    UI_NODE_H_TALL = 280
    TALL_TYPES = {"image", "vace", "subgraph"}

    type_to_col = {"source": 0, "pipeline": 1, "sink": 2, "record": 3}

    def rects_overlap(
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
    ) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)

    # Predict top-level rectangles the frontend will render.
    top_by_col: dict[int, list[str]] = {}
    for n in _top_level_nodes(graph):
        col = type_to_col.get(n.get("type"))
        if col is None or not n.get("id"):
            continue
        top_by_col.setdefault(col, []).append(n["id"])

    rects: list[tuple[tuple[float, float, float, float], str]] = []
    for col, ids in top_by_col.items():
        for i, nid in enumerate(ids):
            rects.append(
                (
                    (
                        float(FE_START_X + col * FE_COLUMN_GAP),
                        float(FE_START_Y + i * (FE_NODE_H + FE_ROW_GAP)),
                        float(FE_NODE_W),
                        float(FE_NODE_H),
                    ),
                    f"top:{nid}",
                )
            )

    # UI-state rectangles use whatever position the agent (or reflow) set.
    for n in _ui_nodes(graph):
        pos = n.get("position") or {}
        try:
            x = float(pos.get("x", 0))
            y = float(pos.get("y", 0))
        except (TypeError, ValueError):
            return CheckResult.fail(
                f"ui node {n.get('id')!r} has invalid position {pos!r}"
            )
        h = UI_NODE_H_TALL if n.get("type") in TALL_TYPES else UI_NODE_H_DEFAULT
        rects.append(((x, y, float(UI_NODE_W), float(h)), f"ui:{n.get('id') or '?'}"))

    for i, (ra, ida) in enumerate(rects):
        for j in range(i + 1, len(rects)):
            rb, idb = rects[j]
            if rects_overlap(ra, rb):
                return CheckResult.fail(f"{ida} overlaps {idb}")

    return CheckResult.ok_(f"no overlaps among {len(rects)} node(s)")


def orphan_sinks(graph: dict, _arg: Any) -> CheckResult:
    """Forbid check: every top-level sink must have an incoming stream edge.

    Observed failure: agent occasionally emits a second ``sink`` node not
    wired to anything, producing a valid-but-dead canvas element. Passes
    validation (disconnected sinks aren't illegal) but is obviously wrong.

    We scan top-level ``graph.edges`` for any ``stream`` edge whose
    ``to_node`` is each top-level sink. A sink with zero such edges is an
    orphan.
    """
    sinks = [n for n in _top_level_nodes(graph) if n.get("type") == "sink"]
    if not sinks:
        # No sinks at all isn't what this check is about — other checks
        # can assert presence if they need to.
        return CheckResult.ok_("no sinks to inspect")

    orphans: list[str] = []
    for s in sinks:
        sink_id = s.get("id")
        has_incoming = any(
            e.get("to_node") == sink_id and e.get("kind") == "stream"
            for e in _top_level_edges(graph)
        )
        if not has_incoming:
            orphans.append(str(sink_id))

    if orphans:
        return CheckResult.fail(
            f"{len(orphans)}/{len(sinks)} sink(s) have no incoming stream edge: "
            f"{orphans}"
        )
    return CheckResult.ok_(f"all {len(sinks)} sink(s) wired")


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
    "orphan_sinks": orphan_sinks,
    "overlapping_nodes": overlapping_nodes,
}


def run_check(name: str, graph: dict, arg: Any) -> CheckResult:
    fn = CHECKS.get(name)
    if fn is None:
        return CheckResult.fail(f"unknown check: {name!r}")
    try:
        return fn(graph, arg)
    except Exception as e:  # defensive — a buggy check must not kill the run
        return CheckResult.fail(f"{type(e).__name__} in check {name}: {e}")
