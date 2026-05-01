# Scope Telemetry

Scope can collect anonymous usage data to help us understand how the app is
used. Telemetry is **off by default** — you choose whether to enable it during
first launch or in Settings. This page documents exactly what is collected and
how to opt in.

**Last updated:** 2026-03-30

## What We Collect

We track a small number of journey-level events. Every event is an explicit
`track()` call in the source code — no auto-capture, no session replay, no user
identification.

| Event | Fires When | Properties |
|-------|------------|------------|
| `onboarding_completed` | Onboarding finishes | — |
| `generation_started` | User starts a stream | `surface` |
| `generation_stopped` | User stops a stream | `surface` |
| `workflow_exported` | User exports a workflow | `node_count`, `surface` |
| `workflow_imported` | User imports a workflow | `node_count`, `source`, `surface` |

### What Is NOT Collected

- Prompt text or any user-generated creative content
- Generated images or video frames
- File paths (local or remote)
- Model file names or specific LoRA names
- IP addresses (disabled in SDK config)
- Email, username, or any personally identifiable information
- Window position or screen arrangement
- Clipboard content
- Individual keystrokes or mouse movements
- Specific error messages

### Super Properties (Attached to Every Event)

| Property | Description |
|----------|-------------|
| `app_version` | Scope version string |
| `platform` | OS platform (darwin, win32, linux) |
| `session_id` | Random UUID per session |
| `device_id` | Random UUID per device (localStorage) |
| `timestamp` | Unix timestamp (ms) |

## Identity

All events are **anonymous**. Scope generates a random device ID (UUID v4)
stored in localStorage. There is no `identify()` call — users are never linked
to an email, username, or Daydream account for analytics purposes.

## How to Enable / Disable

Telemetry is off by default. You can enable it during first launch when
prompted, or at any time in Settings.

### 1. Settings Toggle (UI)

Open **Settings > General > Privacy** and toggle **"Send anonymous usage
data"** on or off. Takes effect immediately — no restart required.

### 2. Environment Variable (Scope-specific) — Force Disable

```bash
SCOPE_TELEMETRY_DISABLED=1 daydream-scope
```

### 3. Environment Variable (Global Convention)

```bash
DO_NOT_TRACK=1 daydream-scope
```

This follows the [Console Do Not Track](https://consoledonottrack.com/)
convention used by Next.js, Astro, Gatsby, and others.

### Precedence

`SCOPE_TELEMETRY_DISABLED` > `DO_NOT_TRACK` > UI setting > default (OFF).

If an environment variable disables telemetry, the Settings toggle shows as
disabled with a note explaining why.

## Analytics Tokens

Analytics provider tokens (PostHog, Mixpanel) are set via environment variables (`VITE_POSTHOG_KEY`,
`VITE_MIXPANEL_TOKEN`) at build time. This means:

- **Building from source** produces a binary with no analytics — the noop
  provider is used when no token is set.
- **Official releases** (`.exe`, `.dmg`) include tokens set during the CI build.

## Technical Details

- **SDK:** [PostHog JS](https://posthog.com/docs/libraries/js) (default for
  production builds)
- **Persistence:** localStorage
- **IP collection:** Disabled (`ip: false`)
- **Auto-capture:** Disabled — every event is an explicit `track()` call
- **Pre-disclosure queue:** Events generated before the user sees the telemetry
  disclosure are queued in memory. If the user accepts, they're sent. If they
  decline, they're dropped.

## Source Code

All tracking calls are in the open-source codebase. Search for `trackEvent(`
or `track(` in the `frontend/src/` directory to see every event.

The telemetry module lives at `frontend/src/lib/telemetry.ts`.
