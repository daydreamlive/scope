# Scope Agent Eval Harness

Measures how often the agentic workflow builder produces a workflow that
matches the user's intent from a single natural-language prompt.

Each **case** = one prompt + structural checks. The runner drives the real
agent via in-process ASGI (no uvicorn, no port) N times and prints a
pass-rate table per case.

## Quickstart

```bash
# Install deps (one-time):
uv sync --group dev

# Ensure an Anthropic key is set:
export ANTHROPIC_API_KEY=sk-ant-...

# Run everything, 5 samples per case (default):
uv run python -m evals

# Run just one case, 1 sample (fast smoke):
uv run python -m evals --case starter-ltx-text-to-video --runs 1

# Cheaper iteration:
uv run python -m evals --model claude-haiku-4-5

# Enforce a bar in CI-like mode:
uv run python -m evals --runs 10 --fail-threshold 90
```

Artifacts land in `evals/outputs/<case>/r<NN>/`:

- `proposal.json` — the full graph the agent proposed.
- `meta.json` — pass/fail, failures, rationale, wall time.
- `trace.jsonl` — every SSE event the agent emitted (one per line).

## Authoring a case

Drop a file in `evals/cases/my-case.yaml`:

```yaml
name: my-case
description: one-line explanation of what good looks like
prompt: |
  A natural-language prompt — as if a user typed it into the agent chat.
runs: 5
expect:
  # Each entry is a single-key mapping: {check_name: argument}.
  - pipelines_include: [longlive]
  - wire_present: { kind: vace_to_pipeline }
  - no_validator_errors: true
forbid:
  - bad_handle_prefix: "parameter:"
```

### Available checks

Registered in [`grader.py`](grader.py):

| Check | Argument | Passes when… |
| ----- | -------- | ------------ |
| `pipelines_equal` | `[ids]` | Pipeline nodes' `pipeline_id`s exactly equal the set. |
| `pipelines_include` | `[ids]` | Pipeline nodes include every id in the list (extras ok). |
| `lora_count_at_least` | `int` | Total LoRA entries across `lora` UI nodes ≥ N. |
| `wire_present` | `{kind, …}` | An edge of the named kind exists. See below. |
| `no_validator_errors` | _(any)_ | `_validate_proposal()` returns zero errors on the graph. |
| `bad_handle_prefix` | `"parameter:"` | (Forbid) No edge handle starts with the prefix. |

`wire_present` kinds:

| Kind | Extra args | Matches |
| ---- | ---------- | ------- |
| `slider_to_pipeline_param` | `target_handle: "param:noise_scale"` | UI-value node → pipeline's `targetHandle`. |
| `vace_to_pipeline` | — | VACE UI node → pipeline's `param:__vace`. |
| `image_to_vace` | — | Image (or value) node → VACE node's `param:ref_image`/`first_frame`/`last_frame`. |
| `prompt_to_pipeline` | — | Any source → pipeline's `param:__prompt`. |
| `lora_to_pipeline` | — | LoRA node → pipeline's `param:__loras`. |

Adding a new check type = adding a function to `grader.py` and registering
it in `CHECKS`. The YAML format picks it up automatically.

## Pytest integration

A single smoke test at `tests/test_evals_smoke.py` runs one case under
`@pytest.mark.eval`. Default `pytest` skips it (pytest-ini addopts
`-m 'not eval'`). To include it:

```bash
uv run pytest -m eval
```

This only verifies the harness wires up end-to-end — it doesn't enforce
pass-rates. For pass-rate enforcement, use `python -m evals`.

## CI

There is a `.github/workflows/eval.yml` that runs on manual dispatch only
(`workflow_dispatch`). It is **not** hooked into `pull_request` or `push`
— LLM evals cost money and are inherently noisy at the edges. Gate launch
decisions on the number, not on PR green.

## Design notes

- The driver uses `httpx.ASGITransport` + `asgi-lifespan` so we hit the
  real `/api/v1/agent/chat` endpoint without spawning a server. This is
  the same endpoint the frontend uses, so behavior is identical to
  production.
- Each case spins up an isolated `AgentSession`; no cross-case
  contamination. Conversation history does not leak between runs.
- Grading is deterministic and structural. No LLM-as-judge in v1.
- Model/provider overrides flow through the on-disk agent config file so
  runs respect the same resolution order the server uses.
