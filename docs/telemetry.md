# Scope Telemetry

Scope can collect anonymous usage data to help us improve the app. Telemetry
is **off by default** — you choose whether to enable it during first launch or
in Settings. This page documents exactly what is collected, how to opt in, and
how identity works.

**Last updated:** 2026-03-23

## What We Collect

We track UI interactions and feature usage patterns. Every event is an explicit
`track()` call in the source code — no auto-capture, no session replay.

### What Is NOT Collected

- Prompt text or any user-generated creative content
- Generated images or video frames
- File paths (local or remote)
- Model file names or specific LoRA names
- IP addresses (disabled in Mixpanel SDK config)
- Window position or screen arrangement
- Clipboard content
- Individual keystrokes or mouse movements
- Specific error messages (only error categories)

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

## Identity

- **Anonymous by default:** On first launch, Scope generates a random device ID
  (UUID v4) stored in localStorage. All events use this anonymous ID.
- **Linked on login:** If you sign in to a Daydream account, events are linked
  to your Daydream user ID so we can see the journey from website to app. On
  logout, tracking reverts to the anonymous device ID.
- **No cross-device tracking:** The device ID is local to your machine.

## Event Taxonomy

### Onboarding (`surface: "onboarding"`)

| Event | Fires When | Properties |
|-------|------------|------------|
| `onboarding_started` | Onboarding overlay renders | `is_first_launch` |
| `onboarding_inference_selected` | User picks cloud or local | `mode` |
| `onboarding_auth_completed` | OAuth succeeds | — |
| `onboarding_workflow_selected` | User picks a starter workflow | `workflowId` |
| `onboarding_workflow_downloaded` | Download completes | `workflowId` |
| `onboarding_started_from_scratch` | User picks "Start from scratch" | — |
| `onboarding_imported_workflow` | User imports a workflow file | — |
| `onboarding_completed` | Onboarding finishes | — |
| `telemetry_disclosure_shown` | Disclosure card renders | `path` |
| `telemetry_disclosure_responded` | User clicks accept or disable | `action`, `path`, `time_to_respond_ms`, `auto_advanced` |

### Performance Mode (`surface: "performance_mode"`)

| Event | Fires When | Properties |
|-------|------------|------------|
| `generation_started` | User hits Play | `surface` |
| `generation_stopped` | User hits Stop | `surface` |
| `parameter_changed` | User adjusts a parameter (debounced 2s) | `parameter_type`, `surface` |
| `prompt_edited` | User modifies prompt (debounced 2s) | `prompt_length` |
| `lora_applied` | User selects a LoRA | `lora_count` |
| `lora_removed` | User removes a LoRA | `lora_count` |
| `input_source_changed` | User switches input source | `source_type` |
| `output_configured` | User sets up an output | `output_type` |
| `mapping_created` | User maps a control | — |
| `mapping_deleted` | User removes a mapping | — |
| `fullscreen_toggled` | User enters/exits fullscreen | `entered` |
| `fps_reported` | Periodic (every 60s during generation) | `fps_avg`, `fps_min`, `fps_max` |
| `resolution_changed` | User changes output resolution | `from_resolution`, `to_resolution` |

### Graph Mode (`surface: "graph_mode"`)

| Event | Fires When | Properties |
|-------|------------|------------|
| `graph_mode_entered` | User switches to graph mode | `node_count`, `connection_count` |
| `graph_mode_exited` | User switches back | `duration_ms` |
| `node_added` | User adds a node | `node_type` |
| `node_removed` | User deletes a node | `node_type` |
| `connection_created` | User connects two nodes | — |
| `connection_removed` | User disconnects nodes | — |
| `node_registry_opened` | User opens the node browser | — |
| `node_registry_searched` | User types in registry search (debounced 2s) | `query_length`, `results_count` |
| `workflow_saved` | User saves the workflow | `node_count` |
| `workflow_exported` | User exports workflow to file | `node_count` |
| `workflow_imported` | User imports a workflow | `node_count`, `source` |

### Settings (`surface: "settings"`)

| Event | Fires When | Properties |
|-------|------------|------------|
| `settings_opened` | User opens Settings panel | `entry_point` |
| `settings_section_viewed` | User navigates to a section | `section` |
| `inference_mode_changed` | User switches cloud/local | `from_mode`, `to_mode` |
| `telemetry_opt_out` | User disables telemetry | `source` |
| `telemetry_opt_in` | User re-enables telemetry | `source` |

### Hub / Plugins (`surface: "hub_browser"`)

| Event | Fires When | Properties |
|-------|------------|------------|
| `hub_opened` | User opens the hub browser | `entry_point` |
| `hub_searched` | User searches in hub (debounced 2s) | `query_length`, `results_count` |
| `hub_item_viewed` | User clicks into an item detail | `item_type`, `item_id` |
| `hub_item_installed` | User installs from hub | `item_type`, `item_id` |
| `hub_item_install_failed` | Install fails | `item_type`, `error_type` |

### I/O Configuration (`surface: "io_config"`)

| Event | Fires When | Properties |
|-------|------------|------------|
| `io_panel_opened` | User opens an I/O config panel | `io_type` |
| `io_config_changed` | User modifies I/O settings (debounced 2s) | `io_type` |

### App Chrome (`surface: "app_chrome"`)

| Event | Fires When | Properties |
|-------|------------|------------|
| `app_launched` | Scope main window renders | `load_time_ms` |
| `app_closed` | User quits Scope | `session_duration_ms` |
| `mode_switched` | User toggles performance/graph | `from_mode`, `to_mode` |
| `error_dialog_shown` | An error dialog appears | `error_category` |
| `user_logged_in` | Auth completes in Scope | `source` |
| `user_logged_out` | User logs out | `source` |

### Super Properties (Attached to Every Event)

| Property | Description |
|----------|-------------|
| `app_version` | Scope version string |
| `platform` | OS platform (darwin, win32, linux) |
| `session_id` | Random UUID per session |
| `timestamp` | Unix timestamp (ms) |

## Technical Details

- **SDK:** [Mixpanel Browser SDK](https://developer.mixpanel.com/docs/javascript-quickstart)
- **Persistence:** localStorage
- **IP collection:** Disabled (`ip: false`)
- **Auto-capture:** Disabled — every event is an explicit `track()` call
- **Debouncing:** Continuous inputs (sliders, search) are debounced at 2 seconds
- **Pre-disclosure queue:** Events generated before the user sees the telemetry
  disclosure are queued in memory. If the user accepts, they're sent. If they
  decline, they're dropped.

## Source Code

All tracking calls are in the open-source codebase. Search for `trackEvent(`
or `track(` in the `frontend/src/` directory to see every event.

The telemetry module lives at `frontend/src/lib/telemetry.ts`.
