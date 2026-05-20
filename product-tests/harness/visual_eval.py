"""Multimodal visual verification via the Anthropic vision API.

This is the "look at it like a human would" half of the testing system. The
rest of the harness asserts on measurements (fps, round-trip ms, retry count)
and testids ("is there a button with this id?"). Neither can answer:

  - "Is the onboarding tooltip placed over the Run button?"
  - "Does the workflow picker show three distinct cards with thumbnails?"
  - "Does the live frame look like a normal scene, or is it all black?"
  - "Are the recorded frames showing visible pixelation?"

The questions above are exactly the class a human spotting a bug asks — and
the class that silently passes today's CI because no selector fails. This
module bridges that gap by routing captured images through Claude with
vision, gated behind an opt-in env var so local runs don't burn API credit.

## Gating

- ``SCOPE_MULTIMODAL_EVAL=1`` — required to actually call the API. Default
  off; tests marked ``@pytest.mark.multimodal`` that skip cleanly when
  disabled return a ``Verdict`` with ``status="uncertain"`` and a "disabled"
  reason so the suite doesn't red.
- ``ANTHROPIC_API_KEY`` — required when ``SCOPE_MULTIMODAL_EVAL=1``. Missing
  key raises so misconfigured CI fails loudly, not silently.
- ``SCOPE_MULTIMODAL_BUDGET_USD`` — optional daily spend cap. Tracked via a
  tiny on-disk ledger at ``~/.daydream-scope/multimodal_ledger.json``. Once
  exhausted, further calls return an ``uncertain`` verdict with a "budget"
  reason and skip the API. Fail-safe, not fail-closed.
- ``SCOPE_MULTIMODAL_TRIAGE=1`` — opt-in triage pass on failure. The caller
  side of this is in ``scenario.py`` teardown; this module just exposes the
  ``triage()`` entry point.

## Caching

Calls are content-hash cached keyed on (sorted image bytes, question text,
must_contain). Identical inputs are served from the cache without a network
call so rerunning the same test suite is free. Cache lives at
``~/.daydream-scope/multimodal_cache/``.

## Why Anthropic, not OpenAI / Gemini

We use the vendor we already ship (Claude). One API key to manage. The
``eval_images`` interface is deliberately thin so a different vendor could
slot in later.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

VerdictStatus = Literal["pass", "fail", "uncertain"]

_CACHE_DIR = Path.home() / ".daydream-scope" / "multimodal_cache"
_LEDGER_PATH = Path.home() / ".daydream-scope" / "multimodal_ledger.json"

# Rough cost model for budgeting. Numbers are conservative — the ledger is
# a safety cap, not an accountant. Adjust once we have real usage data.
_COST_PER_IMAGE_USD = 0.0075  # claude-sonnet-4.x vision ballpark
_COST_PER_CALL_USD = 0.003  # base request overhead

_DEFAULT_MODEL = os.environ.get("SCOPE_MULTIMODAL_MODEL", "claude-sonnet-4-5")


@dataclasses.dataclass(frozen=True)
class Verdict:
    """Structured outcome from a multimodal evaluation call.

    - ``status``: ``"pass"`` / ``"fail"`` / ``"uncertain"``. ``uncertain``
      is used for disabled/budget-exhausted/API-errored calls so tests can
      choose to skip vs. fail.
    - ``reasoning``: one or two sentences of the model's explanation.
    - ``observations``: bullet-list of concrete visual features the model
      named. Useful for triage reports.
    - ``missing_required``: any ``must_contain`` items the model said
      were absent. Empty unless ``status == "fail"`` due to requirements.
    - ``raw``: the raw JSON body from the model. Logged for debug.
    """

    status: VerdictStatus
    reasoning: str
    observations: list[str] = dataclasses.field(default_factory=list)
    missing_required: list[str] = dataclasses.field(default_factory=list)
    raw: dict | None = None

    @property
    def passed(self) -> bool:
        return self.status == "pass"


# ---------------------------------------------------------------------------
# Gating / bookkeeping
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
    """True iff the caller explicitly opted into multimodal evaluation AND a
    usable API key is present. Either missing → graceful disable. This is
    what lets CI "run the multimodal step always, skip when no secret" work:
    steps whose ``if:`` can't reference secrets rely on this function to
    fail-safe silently when the secret isn't plumbed through (forks, local
    runs without a key)."""
    if os.environ.get("SCOPE_MULTIMODAL_EVAL") != "1":
        return False
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    return True


def _disabled_verdict(reason: str) -> Verdict:
    return Verdict(
        status="uncertain",
        reasoning=f"multimodal evaluation disabled ({reason})",
        observations=[],
        missing_required=[],
        raw=None,
    )


def _load_ledger() -> dict:
    if not _LEDGER_PATH.exists():
        return {}
    try:
        return json.loads(_LEDGER_PATH.read_text())
    except Exception:
        return {}


def _save_ledger(ledger: dict) -> None:
    _LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LEDGER_PATH.write_text(json.dumps(ledger, indent=2))


def _today_key() -> str:
    return time.strftime("%Y-%m-%d")


def _budget_remaining_usd() -> float | None:
    """Return seconds-remaining budget for today. ``None`` means no cap set."""
    cap_raw = os.environ.get("SCOPE_MULTIMODAL_BUDGET_USD")
    if not cap_raw:
        return None
    try:
        cap = float(cap_raw)
    except ValueError:
        return None
    ledger = _load_ledger()
    spent = float(ledger.get(_today_key(), 0.0))
    return max(cap - spent, 0.0)


def _record_spend(usd: float) -> None:
    ledger = _load_ledger()
    key = _today_key()
    ledger[key] = round(float(ledger.get(key, 0.0)) + usd, 6)
    # Keep the last 30 days only; trim older keys so the file doesn't grow.
    keys = sorted(ledger.keys())
    for k in keys[:-30]:
        ledger.pop(k, None)
    _save_ledger(ledger)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def _cache_key(images: list[Path], question: str, must_contain: list[str]) -> str:
    h = hashlib.sha256()
    for p in sorted(images, key=lambda x: str(x)):
        h.update(p.read_bytes())
        h.update(b"\x00")
    h.update(question.encode("utf-8"))
    h.update(b"\x00")
    h.update("|".join(sorted(must_contain)).encode("utf-8"))
    h.update(b"\x00")
    h.update(_DEFAULT_MODEL.encode("utf-8"))
    return h.hexdigest()


def _cache_read(key: str) -> Verdict | None:
    p = _CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        return Verdict(
            status=data["status"],
            reasoning=data["reasoning"],
            observations=data.get("observations", []),
            missing_required=data.get("missing_required", []),
            raw=data.get("raw"),
        )
    except Exception:
        return None


def _cache_write(key: str, v: Verdict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (_CACHE_DIR / f"{key}.json").write_text(json.dumps(dataclasses.asdict(v), indent=2))


# ---------------------------------------------------------------------------
# Anthropic call
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = """You are a visual QA reviewer for a real-time video AI \
tool. You will be shown images captured during an automated test run. They \
may be browser screenshots (UI) or frames from a video stream. Your job is \
to answer the caller's question with one of three verdicts:

- "pass": the images match the caller's expected state with high confidence.
- "fail": the images clearly show the expected state is NOT met.
- "uncertain": the images are ambiguous, corrupted, or don't contain enough \
  information to decide. Prefer "uncertain" over a guess.

Respond ONLY with a JSON object of the exact shape:

  {
    "status": "pass" | "fail" | "uncertain",
    "reasoning": "<one or two sentences>",
    "observations": ["<concrete visual detail>", ...],
    "missing_required": ["<any must_contain item absent from the images>", ...]
  }

Rules:
- "observations" must be concrete things you actually see (e.g. "3 card \
  components arranged in a row"), not inferences.
- If the caller provides "must_contain", list any items you cannot visually \
  confirm in "missing_required" AND set status to "fail".
- Never claim to see text or elements that aren't there. When unsure, \
  return "uncertain"."""


def _encode_image(p: Path) -> dict:
    data = base64.standard_b64encode(p.read_bytes()).decode("ascii")
    # Content-type inferred from extension. We only emit jpg/png.
    ext = p.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
        ext, "image/jpeg"
    )
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": mime, "data": data},
    }


def _build_user_content(
    images: list[Path], question: str, must_contain: list[str]
) -> list[dict]:
    content: list[dict] = [_encode_image(p) for p in images]
    prompt = question.strip()
    if must_contain:
        prompt += "\n\nThe images MUST contain all of:\n- " + "\n- ".join(must_contain)
    prompt += (
        "\n\nRespond with the JSON object exactly as specified in the system "
        "prompt. Do not wrap the JSON in markdown fences or any prose."
    )
    content.append({"type": "text", "text": prompt})
    return content


def _call_anthropic(
    images: list[Path], question: str, must_contain: list[str]
) -> Verdict:
    """The only network call in this module. Runs when is_enabled()."""
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise RuntimeError(
            "anthropic package not installed; add it to the product-tests "
            "dependency group: `pip install anthropic`"
        ) from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required when SCOPE_MULTIMODAL_EVAL=1")

    client = Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=_DEFAULT_MODEL,
        max_tokens=800,
        system=_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": _build_user_content(images, question, must_contain),
            }
        ],
    )

    # Parse the first text block as our JSON verdict.
    text = ""
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            text = block.text.strip()
            break
    if not text:
        return Verdict(
            status="uncertain",
            reasoning="model returned no text content",
            raw={"model": _DEFAULT_MODEL, "id": msg.id},
        )

    # Be forgiving if the model wraps in ```json fences despite the instruction.
    if text.startswith("```"):
        text = text.strip("`")
        # Drop optional "json\n" header.
        text = text.split("\n", 1)[1] if "\n" in text else text

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return Verdict(
            status="uncertain",
            reasoning=f"model returned non-JSON text: {text[:200]}",
            raw={"text": text, "id": msg.id},
        )

    status = data.get("status")
    if status not in ("pass", "fail", "uncertain"):
        status = "uncertain"
    return Verdict(
        status=status,
        reasoning=str(data.get("reasoning", ""))[:1000],
        observations=[str(x) for x in data.get("observations", [])][:20],
        missing_required=[str(x) for x in data.get("missing_required", [])][:20],
        raw={"text": text, "id": msg.id, "model": _DEFAULT_MODEL},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def eval_images(
    images: Iterable[Path | str],
    question: str,
    *,
    must_contain: list[str] | None = None,
) -> Verdict:
    """Ask Claude whether ``images`` satisfy ``question``.

    Args:
        images: paths to JPEG/PNG files. Sink frames, screenshots, and
            sample frames from a recording are all valid inputs and can be
            mixed in the same call.
        question: plain-English question (e.g. "Does the workflow picker
            show three distinct cards?").
        must_contain: optional list of items that MUST be visually
            present. If any is absent, the verdict is forced to ``fail``
            and the missing items are listed.

    Returns a ``Verdict``. Never raises on API / network errors — returns
    an ``uncertain`` verdict with the error reason so tests can decide.
    """
    imgs = [Path(p) for p in images]
    must = list(must_contain or [])

    # Graceful-disable must come BEFORE any input validation — a test that
    # captured zero frames for whatever reason should still skip cleanly
    # when multimodal is off, not crash with a ValueError.
    if not is_enabled():
        if os.environ.get("SCOPE_MULTIMODAL_EVAL") != "1":
            return _disabled_verdict("SCOPE_MULTIMODAL_EVAL is not set to 1")
        return _disabled_verdict("ANTHROPIC_API_KEY is not set")

    if not imgs:
        raise ValueError("eval_images requires at least one image")
    for p in imgs:
        if not p.exists():
            raise FileNotFoundError(f"image does not exist: {p}")

    key = _cache_key(imgs, question, must)
    cached = _cache_read(key)
    if cached is not None:
        return cached

    # Budget check.
    est_cost = _COST_PER_CALL_USD + _COST_PER_IMAGE_USD * len(imgs)
    remaining = _budget_remaining_usd()
    if remaining is not None and remaining < est_cost:
        return _disabled_verdict(
            f"daily multimodal budget exhausted "
            f"(cap={os.environ.get('SCOPE_MULTIMODAL_BUDGET_USD')}, "
            f"remaining=${remaining:.3f}, need=${est_cost:.3f})"
        )

    try:
        verdict = _call_anthropic(imgs, question, must)
    except Exception as e:
        return Verdict(
            status="uncertain",
            reasoning=f"anthropic call failed: {type(e).__name__}: {e}",
        )

    # Best-effort budget accounting based on our cost model.
    _record_spend(est_cost)
    # Cache the result. ``uncertain`` caused by transient API errors we
    # skip caching so a later retry can succeed.
    if verdict.raw is not None and verdict.status != "uncertain":
        _cache_write(key, verdict)
    return verdict


def triage(images: Iterable[Path | str], context: str = "") -> Verdict:
    """Post-failure triage pass — "what does this failure look like?".

    Used by the ``SCOPE_MULTIMODAL_TRIAGE=1`` pathway in
    ``scenario.py``. Returns a ``Verdict`` whose ``observations`` list
    describes the visible symptoms in plain English.
    """
    q = (
        "A test has failed. Describe what you see in these captured images "
        "(screenshots + stream frames + recorded samples). Be concrete: name "
        "specific visible symptoms (layout issues, missing elements, wrong "
        "colors, frozen frames, error toasts) that a human reviewer should "
        "look at first."
    )
    if context:
        q += f"\n\nContext from the test run: {context}"
    # Triage is informational — we still use eval_images so it respects
    # the gate + cache. Verdicts map: "fail" = visible symptoms found.
    return eval_images(images, q)


__all__ = [
    "Verdict",
    "eval_images",
    "is_enabled",
    "triage",
]
