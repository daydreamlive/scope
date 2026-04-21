---
name: onboarding-test
description: Pre-release onboarding test via Chrome browser automation. Tests the full new-user flow — provider selection, workflow picker, and streaming all three starter workflows. Use when asked to test onboarding, first-run experience, or starter workflows.
---

# Onboarding Browser Test

## Prerequisites

- Chrome browser automation tools (claude-in-chrome MCP)
- Build frontend first: `cd frontend && npm run build`

## Server Setup

Use port **8080** (not 8000 — the OSC server binds to the same port as the HTTP server and port 8000 is commonly in use).

```bash
mkdir -p /tmp/scope-onboarding-test/data /tmp/scope-onboarding-test/models
lsof -ti:8080 | xargs kill -9 2>/dev/null
DAYDREAM_SCOPE_DIR=/tmp/scope-onboarding-test/data \
DAYDREAM_SCOPE_MODELS_DIR=/tmp/scope-onboarding-test/models \
SCOPE_CLOUD_APP_ID="daydream/scope-livepeer--prod/ws" \
uv run daydream-scope --port 8080 > /tmp/scope-onboarding.log 2>&1 &
for i in $(seq 1 30); do curl -s http://localhost:8080/health > /dev/null 2>&1 && break; sleep 1; done
```

## Onboarding UI Flow (exact sequence)

Navigate to `http://localhost:8080`. The onboarding screens appear in this order:

1. **Provider selection** — "Welcome to Daydream Scope" with "Use Daydream Cloud" and "Run Locally" cards. Select Cloud, click **Continue**.
2. **Usage Analytics dialog** — appears as a modal overlay. Click **No thanks** (privacy-preserving default).
3. **Onboarding style** — "Teaching Mode" vs "Simple". Pick either, click **Continue**.
4. **Workflow picker** — "Pick a workflow to get started" showing 3 starter workflows:
   - **Mythical Creature** (Style LoRA)
   - **Dissolving Sunflower** (Depth Map)
   - **LTX 2.3** (Text to Video)
   
   Select one, click **Get Started**.

5. **Graph editor with onboarding tooltips** — Two tooltip popups appear sequentially over the Sink/Run area:
   - Tooltip 1: "Click Play to start generation" (1 of 2) — click **Next**
   - Tooltip 2: "Explore Workflows" (2 of 2) — click **Done**
   
   **IMPORTANT:** These tooltips intercept clicks on the Run button. You MUST dismiss both tooltips (using `read_page` to find the Next/Done button refs) BEFORE clicking Run.

6. **Click Run** — use `read_page(filter="interactive")` to find the Run button ref and click it. Do NOT click by coordinates near the tooltip area.

## Streaming Each Workflow

- After clicking Run, the status bar shows "Loading diffusion model..." / "Starting..."
- Cloud model loading takes **30-60 seconds** on first run. Wait in 10s increments, then screenshot.
- When ready, the Sink node shows video output with FPS/bitrate overlay.
- Click **Stop** to end the stream.

### Switching workflows

Click **Workflows** in the top nav bar to reopen the workflow panel. The "Getting Started" section shows all three starter workflows. Click a different one to load it, then click Run.

## Expected Results

| Workflow | Nodes | Notes |
|----------|-------|-------|
| Mythical Creature | Source, VACE, LoRA, longlive, rife, Sink | Style LoRA, video input |
| Dissolving Sunflower | Source, video-depth-anything, VACE, LoRA, longlive, rife, Sink | Depth map, video input |
| LTX 2.3 | Primitive (String), ltx2, Sink | Text-to-video, no Source node |

## Cleanup

```bash
lsof -ti:8080 | xargs kill -9 2>/dev/null
rm -rf /tmp/scope-onboarding-test
```
