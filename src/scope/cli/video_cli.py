"""CLI interface for Scope realtime video generation.

Designed for agent automation. All commands return JSON.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

import click
import httpx

DEFAULT_URL = "http://localhost:8000"


def get_client(ctx) -> httpx.Client:
    return httpx.Client(base_url=ctx.obj["url"], timeout=60.0)


def output(data: dict, ctx):
    if ctx.obj.get("pretty"):
        click.echo(json.dumps(data, indent=2))
    else:
        click.echo(json.dumps(data))


def handle_error(response: httpx.Response):
    if response.status_code >= 400:
        try:
            error = response.json()
        except Exception:
            error = {"error": response.text}
        click.echo(json.dumps(error), err=True)
        sys.exit(1)


# Default content directory for playlists
CONTENT_DIR = Path(__file__).parent.parent.parent.parent / "content" / "playlists"


def _resolve_playlist_path(file_or_name: str) -> Path | None:
    """Resolve a playlist name or file path to an actual file."""
    direct = Path(file_or_name)
    if direct.exists() and direct.is_file():
        return direct

    playlist_dir = CONTENT_DIR / file_or_name
    if playlist_dir.exists() and playlist_dir.is_dir():
        captioning_dir = playlist_dir / "Captioning"
        if captioning_dir.exists():
            for txt_file in captioning_dir.glob("*_captions.txt"):
                return txt_file
        for pattern in ["captions.txt", f"{file_or_name}.txt"]:
            candidate = playlist_dir / pattern
            if candidate.exists():
                return candidate
    return None


@click.group()
@click.option("--url", envvar="VIDEO_API_URL", default=DEFAULT_URL, help="API base URL")
@click.option("--pretty/--no-pretty", default=True, help="Pretty print JSON output")
@click.pass_context
def cli(ctx, url, pretty):
    """Scope realtime video CLI - control video generation via REST API."""
    ctx.ensure_object(dict)
    ctx.obj["url"] = url
    ctx.obj["pretty"] = pretty


# --- State ---


@cli.command()
@click.pass_context
def state(ctx):
    """Get current session state."""
    with get_client(ctx) as client:
        r = client.get("/api/v1/realtime/state")
        handle_error(r)
        output(r.json(), ctx)


# --- Generation ---


@cli.command()
@click.pass_context
def step(ctx):
    """Generate one chunk."""
    with get_client(ctx) as client:
        r = client.post("/api/v1/realtime/step")
        handle_error(r)
        output(r.json(), ctx)


@cli.command()
@click.option("--chunks", type=int, default=None, help="Number of chunks to generate")
@click.pass_context
def run(ctx, chunks):
    """Start or run generation."""
    with get_client(ctx) as client:
        params = {"chunks": chunks} if chunks else {}
        r = client.post("/api/v1/realtime/run", params=params)
        handle_error(r)
        output(r.json(), ctx)


@cli.command()
@click.pass_context
def pause(ctx):
    """Pause generation."""
    with get_client(ctx) as client:
        r = client.post("/api/v1/realtime/pause")
        handle_error(r)
        output(r.json(), ctx)


# --- Prompt ---


@cli.command()
@click.argument("text", required=False)
@click.option("--get", "get_only", is_flag=True, help="Only get current prompt")
@click.pass_context
def prompt(ctx, text, get_only):
    """Set or get prompt."""
    with get_client(ctx) as client:
        if get_only or text is None:
            # Get current state which includes prompt
            r = client.get("/api/v1/realtime/state")
            handle_error(r)
            data = r.json()
            output({"prompt": data.get("prompt")}, ctx)
        else:
            r = client.put("/api/v1/realtime/prompt", json={"prompt": text})
            handle_error(r)
            output(r.json(), ctx)


@cli.command()
@click.option(
    "--direction",
    "-d",
    default=None,
    help='Steering direction (e.g., "sadder", "more dynamic")',
)
@click.option(
    "--intensity",
    "-i",
    default=0.3,
    type=float,
    show_default=True,
    help="Variation intensity 0-1",
)
@click.option(
    "--count",
    "-n",
    default=3,
    type=int,
    show_default=True,
    help="Number of variations",
)
@click.option(
    "--mode",
    "-m",
    default="attentional",
    type=click.Choice(["attentional", "semantic"]),
    show_default=True,
    help="Jiggle mode",
)
@click.option(
    "--apply",
    "apply_index",
    type=int,
    default=None,
    help="Immediately apply variation at index (0-based)",
)
@click.option("--soft-cut", is_flag=True, help="Use soft cut when applying")
@click.option("--soft-cut-bias", type=float, default=None, help="Soft cut temp_bias (server default if omitted)")
@click.option("--soft-cut-chunks", type=int, default=None, help="Soft cut num_chunks (server default if omitted)")
@click.option("--hard-cut", is_flag=True, help="Use hard cut when applying")
@click.option("--prompt", "-p", default=None, help="Prompt to jiggle (default: current active)")
@click.option(
    "--print",
    "print_human",
    is_flag=True,
    help="Print a compact list to stderr (JSON still on stdout)",
)
@click.pass_context
def jiggle(
    ctx,
    direction,
    intensity,
    count,
    mode,
    apply_index,
    soft_cut,
    soft_cut_bias,
    soft_cut_chunks,
    hard_cut,
    prompt,
    print_human,
):
    """Generate prompt variations."""
    with get_client(ctx) as client:
        r = client.post(
            "/api/v1/prompt/jiggle",
            json={
                "prompt": prompt,
                "intensity": intensity,
                "count": count,
                "direction": direction,
                "mode": mode,
            },
        )
        handle_error(r)
        response = r.json()

        if print_human:
            click.echo(f"\nOriginal:\n{response.get('original_prompt', '')}\n", err=True)
            click.echo(f"Variations ({mode}):", err=True)
            for i, v in enumerate(response.get("variations", [])):
                click.echo(f"  [{i}] {v}", err=True)

        applied = None
        if apply_index is not None:
            variations = response.get("variations", [])
            if not isinstance(variations, list):
                variations = []
            if apply_index < 0 or apply_index >= len(variations):
                click.echo(
                    json.dumps(
                        {
                            "error": "apply index out of range",
                            "apply_index": apply_index,
                            "variations_count": len(variations),
                        }
                    ),
                    err=True,
                )
                sys.exit(1)

            selected = variations[apply_index]
            if hard_cut:
                r2 = client.post("/api/v1/realtime/hard-cut", json={"prompt": selected})
            elif soft_cut:
                payload: dict[str, object] = {"prompt": selected}
                if soft_cut_bias is not None:
                    payload["temp_bias"] = soft_cut_bias
                if soft_cut_chunks is not None:
                    payload["num_chunks"] = soft_cut_chunks
                r2 = client.post("/api/v1/realtime/soft-cut", json=payload)
            else:
                r2 = client.put("/api/v1/realtime/prompt", json={"prompt": selected})
            handle_error(r2)
            applied = {"index": apply_index, "prompt": selected, "response": r2.json()}

        output({**response, "applied": applied}, ctx)


# --- Frame ---


@cli.command()
@click.option("--out", type=click.Path(), help="Output file path")
@click.pass_context
def frame(ctx, out):
    """Get current frame."""
    with get_client(ctx) as client:
        if out:
            r = client.get("/api/v1/realtime/frame/latest")
            handle_error(r)
            Path(out).write_bytes(r.content)
            output({"saved": out, "size_bytes": len(r.content)}, ctx)
        else:
            # Just report that frame exists
            r = client.get("/api/v1/realtime/state")
            handle_error(r)
            output({"chunk_index": r.json().get("chunk_index")}, ctx)


# --- World State ---


@cli.command()
@click.argument("json_data", required=False)
@click.option("--get", "get_only", is_flag=True, help="Only get current world state")
@click.pass_context
def world(ctx, json_data, get_only):
    """Set or get WorldState.

    Examples:
        video-cli world                              # Get current world state
        video-cli world '{"action":"run"}'           # Set world state
    """
    with get_client(ctx) as client:
        if get_only or json_data is None:
            r = client.get("/api/v1/realtime/state")
            handle_error(r)
            data = r.json()
            output({"world_state": data.get("world_state")}, ctx)
        else:
            try:
                world_state = json.loads(json_data)
            except json.JSONDecodeError as e:
                click.echo(json.dumps({"error": f"Invalid JSON: {e}"}), err=True)
                sys.exit(1)
            r = client.put("/api/v1/realtime/world", json={"world_state": world_state})
            handle_error(r)
            output(r.json(), ctx)


# --- Parameters ---


@cli.command()
@click.argument("json_data", required=False)
@click.pass_context
def params(ctx, json_data):
    """Update realtime parameters via the generic parameters endpoint.

    Examples:
        video-cli params '{"vace_context_scale": 0.5}'
        video-cli params '{"kv_cache_attention_bias": 0.3, "noise_scale": 0.98}'
        echo '{"reset_cache": true}' | video-cli params -
    """
    if not json_data:
        click.echo(
            json.dumps(
                {
                    "error": "Missing JSON payload",
                    "examples": [
                        """video-cli params '{"vace_context_scale": 0.5}'""",
                        """video-cli params '{"kv_cache_attention_bias": 0.3}'""",
                        """echo '{\"reset_cache\": true}' | video-cli params -""",
                    ],
                }
            ),
            err=True,
        )
        sys.exit(2)

    if json_data.strip() == "-":
        json_data = sys.stdin.read()

    try:
        payload = json.loads(json_data)
    except json.JSONDecodeError as e:
        click.echo(json.dumps({"error": f"Invalid JSON: {e}"}), err=True)
        sys.exit(1)

    if not isinstance(payload, dict):
        click.echo(json.dumps({"error": "Payload must be a JSON object"}), err=True)
        sys.exit(1)

    with get_client(ctx) as client:
        r = client.post("/api/v1/realtime/parameters", json=payload)
        handle_error(r)
        output(r.json(), ctx)


# --- Style ---


@cli.group()
@click.pass_context
def style(ctx):
    """Manage active style."""
    pass


@style.command("list")
@click.pass_context
def style_list(ctx):
    """List available styles."""
    with get_client(ctx) as client:
        r = client.get("/api/v1/realtime/style/list")
        handle_error(r)
        output(r.json(), ctx)


@style.command("set")
@click.argument("name")
@click.pass_context
def style_set(ctx, name):
    """Set active style by name."""
    with get_client(ctx) as client:
        r = client.put("/api/v1/realtime/style", json={"name": name})
        handle_error(r)
        output(r.json(), ctx)


@style.command("get")
@click.pass_context
def style_get(ctx):
    """Get currently active style."""
    with get_client(ctx) as client:
        r = client.get("/api/v1/realtime/state")
        handle_error(r)
        data = r.json()
        output(
            {
                "active_style": data.get("active_style"),
                "compiled_prompt": data.get("compiled_prompt"),
            },
            ctx,
        )


@style.command("blend")
@click.argument("mode", required=False, type=click.Choice(["on", "off"]))
@click.pass_context
def style_blend(ctx, mode):
    """Toggle or check style blend mode.

    When blend mode is ON, style switches don't reset the KV cache,
    creating interesting visual artifacts during transitions.

    Examples:
        video-cli style blend        # Get current blend mode
        video-cli style blend on     # Enable blend mode
        video-cli style blend off    # Disable blend mode (clean transitions)
    """
    with get_client(ctx) as client:
        if mode is None:
            # Get current blend mode
            r = client.get("/api/v1/realtime/style/blend-mode")
            handle_error(r)
            output(r.json(), ctx)
        else:
            # Set blend mode
            enabled = mode == "on"
            r = client.put("/api/v1/realtime/style/blend-mode", json={"enabled": enabled})
            handle_error(r)
            output(r.json(), ctx)


@style.command("nav")
@click.pass_context
def style_nav(ctx):
    """Interactive style navigation mode.

    Controls:
        j, ↓          Move selection down
        k, ↑          Move selection up
        ENTER, SPACE  Activate selected style
        b             Toggle blend mode
        r             Refresh display
        q, ESC        Quit

    Run this in a second terminal alongside 'video-cli playlist nav' for
    full control over both prompts and styles during live performance.
    """
    import os
    import select
    import termios
    import tty

    def get_char_nonblocking(timeout=0.2):
        """Read a char with timeout. Returns None if no input."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            if select.select([fd], [], [], timeout)[0]:
                ch = os.read(fd, 1).decode("utf-8", errors="ignore")
                if ch == "\x1b":
                    extra = ""
                    for _ in range(5):
                        if select.select([fd], [], [], 0.05)[0]:
                            byte = os.read(fd, 1).decode("utf-8", errors="ignore")
                            extra += byte
                            if len(extra) >= 2 and extra[0] == "[" and extra[-1] in "ABCD":
                                break
                        else:
                            break
                    ch = ch + extra
                return ch
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def display_styles(client, styles, selected_idx, active_style, blend_mode):
        """Display styles list with selection and active indicators."""
        import shutil

        term_width = shutil.get_terminal_size().columns

        click.echo("\n" + "=" * term_width)
        status = f"  Styles: {len(styles)} available"
        if active_style:
            status += f"  [Active: {active_style}]"
        if blend_mode:
            status += "  [🌀 BLEND]"
        click.echo(status)
        click.echo("=" * term_width)

        for i, style_info in enumerate(styles):
            name = style_info.get("name", "unknown")
            trigger = style_info.get("trigger_words", [])
            trigger_str = trigger[0] if trigger else ""

            # Markers: > for selected, * for active
            is_selected = i == selected_idx
            is_active = name == active_style

            if is_selected and is_active:
                marker = "▶*"
                click.echo(
                    click.style(f"{marker} {name:<15} {trigger_str}", fg="green", bold=True)
                )
            elif is_selected:
                marker = "▶ "
                click.echo(
                    click.style(f"{marker} {name:<15} {trigger_str}", fg="cyan", bold=True)
                )
            elif is_active:
                marker = " *"
                click.echo(
                    click.style(f"{marker} {name:<15} {trigger_str}", fg="green")
                )
            else:
                marker = "  "
                click.echo(f"{marker} {name:<15} {trigger_str}")

        click.echo("=" * term_width)
        click.echo("  j/k nav | ENTER activate | b blend | r refresh | q quit")
        click.echo("=" * term_width + "\n")

    click.echo("\nStyle Navigation Mode")
    click.echo("Press q or ESC to quit\n")

    selected_idx = 0
    blend_mode = False

    with get_client(ctx) as client:
        # Fetch styles
        r = client.get("/api/v1/realtime/style/list")
        if r.status_code != 200:
            click.echo("Failed to fetch styles")
            return
        styles = r.json().get("styles", [])
        if not styles:
            click.echo("No styles available")
            return

        # Fetch current active style
        r = client.get("/api/v1/realtime/state")
        active_style = None
        if r.status_code == 200:
            active_style = r.json().get("active_style")

        # Set initial selection to active style if found
        for i, s in enumerate(styles):
            if s.get("name") == active_style:
                selected_idx = i
                break

        # Fetch blend mode
        try:
            r = client.get("/api/v1/realtime/style/blend-mode")
            if r.status_code == 200:
                blend_mode = r.json().get("blend_mode", False)
        except Exception:
            pass

        display_styles(client, styles, selected_idx, active_style, blend_mode)

        while True:
            try:
                ch = get_char_nonblocking(timeout=0.2)

                if ch is not None:
                    # Quit
                    if ch in ("q", "Q", "\x03"):
                        click.echo("\nExiting style navigation mode.\n")
                        break
                    elif ch == "\x1b" and len(ch) == 1:
                        click.echo("\nExiting style navigation mode.\n")
                        break

                    # Navigation
                    elif ch in ("j", "\x1b[B"):  # Down
                        selected_idx = (selected_idx + 1) % len(styles)
                        display_styles(client, styles, selected_idx, active_style, blend_mode)

                    elif ch in ("k", "\x1b[A"):  # Up
                        selected_idx = (selected_idx - 1) % len(styles)
                        display_styles(client, styles, selected_idx, active_style, blend_mode)

                    # Activate selected style
                    elif ch in ("\r", "\n", " "):
                        style_name = styles[selected_idx].get("name")
                        r = client.put("/api/v1/realtime/style", json={"name": style_name})
                        if r.status_code == 200:
                            active_style = style_name
                            click.echo(f"  ✓ Activated: {style_name}")
                        else:
                            click.echo(f"  ✗ Failed to activate: {style_name}")
                        display_styles(client, styles, selected_idx, active_style, blend_mode)

                    # Toggle blend mode
                    elif ch == "b":
                        blend_mode = not blend_mode
                        try:
                            r = client.put(
                                "/api/v1/realtime/style/blend-mode",
                                json={"enabled": blend_mode}
                            )
                            if r.status_code == 200:
                                status = "ON" if blend_mode else "OFF"
                                click.echo(f"  🌀 Blend mode: {status}")
                            else:
                                blend_mode = not blend_mode
                                click.echo("  Failed to set blend mode")
                        except Exception as e:
                            blend_mode = not blend_mode
                            click.echo(f"  Error: {e}")
                        display_styles(client, styles, selected_idx, active_style, blend_mode)

                    # Refresh
                    elif ch == "r":
                        # Re-fetch styles and state
                        r = client.get("/api/v1/realtime/style/list")
                        if r.status_code == 200:
                            styles = r.json().get("styles", [])
                        r = client.get("/api/v1/realtime/state")
                        if r.status_code == 200:
                            active_style = r.json().get("active_style")
                        display_styles(client, styles, selected_idx, active_style, blend_mode)

            except KeyboardInterrupt:
                click.echo("\nExiting style navigation mode.\n")
                break
            except Exception as e:
                click.echo(f"\nError: {e}\n")
                break


# --- Snapshots (placeholder - uses existing WebRTC API indirectly) ---


@cli.command()
@click.pass_context
def snapshot(ctx):
    """Create snapshot (requires WebRTC session to handle response)."""
    # Note: Snapshot creation goes through update_parameters which
    # sends response via WebRTC data channel. For full REST support,
    # we'd need to add dedicated snapshot endpoints.
    output(
        {
            "status": "not_implemented",
            "message": "Snapshot creation requires dedicated REST endpoint",
        },
        ctx,
    )


@cli.command()
@click.argument("snapshot_id")
@click.pass_context
def restore(ctx, snapshot_id):
    """Restore from snapshot (requires WebRTC session to handle response)."""
    output(
        {
            "status": "not_implemented",
            "message": "Snapshot restore requires dedicated REST endpoint",
        },
        ctx,
    )


# --- VACE ---


@cli.group()
@click.pass_context
def vace(ctx):
    """Manage VACE (video-to-video editing) settings."""
    pass


@vace.command("control-map")
@click.argument("mode", required=False, type=click.Choice(["none", "canny", "pidinet", "depth", "composite", "external"]))
@click.option("--low", type=int, help="Canny low threshold (default: adaptive)")
@click.option("--high", type=int, help="Canny high threshold (default: adaptive)")
@click.option("--safe/--no-safe", default=None, help="PiDiNet safe mode (cleaner edges)")
@click.option("--filter/--no-filter", default=None, help="PiDiNet apply filter")
@click.option("--edge-strength", type=float, help="Composite: edge strength 0-1 (default: 0.6)")
@click.option("--edge-source", type=click.Choice(["canny", "pidinet"]), help="Composite: edge source")
@click.option("--edge-thickness", type=int, help="Composite: edge thickness in pixels (default: 8)")
@click.option("--sharpness", type=float, help="Composite: soft max sharpness (default: 10.0)")
@click.option("--depth-input-size", type=int, help="Depth: VDA input_size (default: 518; lower is faster)")
@click.option("--depth-fp32/--depth-no-fp32", default=None, help="Depth: force FP32 (default: autocast)")
@click.option("--depth-temporal-mode", type=click.Choice(["stream", "stateless"]), help="Depth: temporal mode (stream=stable, stateless=no trails)")
@click.option("--depth-contrast", type=float, help="Depth: contrast/gamma (default: 1.0; >1.0 increases mid-tone contrast for close-ups)")
@click.option("--temporal-ema", type=float, help="Temporal EMA smoothing 0.0-0.95 (0.0=none, 0.5=smooth, 0.9=very smooth)")
@click.option("--worker/--no-worker", default=None, help="Enable/disable control-map worker (preview; generation with --control-buffer)")
@click.option("--worker-allow-heavy/--no-worker-allow-heavy", default=None, help="Allow heavy modes (depth/pidinet/composite) in worker")
@click.option("--worker-max-fps", type=float, help="Cap worker FPS (0 disables cap)")
@click.option("--control-buffer/--no-control-buffer", default=None, help="Enable/disable control buffer (use worker outputs for generation)")
@click.pass_context
def vace_control_map(ctx, mode, low, high, safe, filter, edge_strength, edge_source, edge_thickness, sharpness, depth_input_size, depth_fp32, depth_temporal_mode, depth_contrast, temporal_ema, worker, worker_allow_heavy, worker_max_fps, control_buffer):
    """Get or set VACE control map mode.

    Control maps transform webcam/video input before VACE conditioning:
    - "none": Use raw video frames (default)
    - "canny": Apply Canny edge detection (fast, CPU-based)
    - "pidinet": Neural edge detection (higher quality, requires controlnet_aux)
    - "depth": Apply VDA depth estimation (depth/layout control)
    - "composite": Depth + edges fused with soft max (best composition lock)
    - "external": Passthrough mode (input frames are already control maps)

    Examples:
        video-cli vace control-map          # Get current mode
        video-cli vace control-map canny    # Enable Canny edge detection
        video-cli vace control-map pidinet  # Enable PiDiNet neural edges
        video-cli vace control-map depth    # Enable depth estimation
        video-cli vace control-map depth --depth-input-size 320  # Faster depth (lower input_size)
        video-cli vace control-map depth --depth-temporal-mode stream  # Stable depth (can trail)
        video-cli vace control-map depth --depth-temporal-mode stateless  # No trails/ghosting
        video-cli vace control-map composite  # Enable depth+edges composite
        video-cli vace control-map external   # Passthrough (precomputed control maps)
        video-cli vace control-map none     # Disable (use raw frames)
        video-cli vace control-map canny --low 50 --high 150  # Custom thresholds
        video-cli vace control-map pidinet --no-safe  # PiDiNet with more detail
        video-cli vace control-map composite --edge-strength 0.7  # Stronger edges
        video-cli vace control-map depth --temporal-ema 0.5  # Smooth depth maps
        video-cli vace control-map depth --depth-contrast 1.5  # More contrast for webcam close-ups
        video-cli vace control-map depth --worker  # Enable background worker (preview)
        video-cli vace control-map depth --worker --control-buffer  # Worker-assisted generation
        video-cli vace control-map depth --worker --control-buffer --worker-allow-heavy  # Allow heavy modes in worker
        video-cli vace control-map depth --worker-max-fps 15  # Cap worker FPS
    """
    with get_client(ctx) as client:
        if mode is None:
            # Get current control map mode
            r = client.get("/api/v1/realtime/vace/control-map-mode")
            handle_error(r)
            output(r.json(), ctx)
        else:
            # Set control map mode
            payload = {"mode": mode}
            # Canny options
            if low is not None:
                payload["canny_low_threshold"] = low
            if high is not None:
                payload["canny_high_threshold"] = high
            # PiDiNet options
            if safe is not None:
                payload["pidinet_safe"] = safe
            if filter is not None:
                payload["pidinet_filter"] = filter
            # Composite options
            if edge_strength is not None:
                payload["composite_edge_strength"] = edge_strength
            if edge_source is not None:
                payload["composite_edge_source"] = edge_source
            if edge_thickness is not None:
                payload["composite_edge_thickness"] = edge_thickness
            if sharpness is not None:
                payload["composite_sharpness"] = sharpness
            # Depth options
            if depth_input_size is not None:
                payload["depth_input_size"] = depth_input_size
            if depth_fp32 is not None:
                payload["depth_fp32"] = depth_fp32
            if depth_temporal_mode is not None:
                payload["depth_temporal_mode"] = depth_temporal_mode
            if depth_contrast is not None:
                payload["depth_contrast"] = depth_contrast
            # Worker options
            if worker is not None:
                payload["worker_enabled"] = worker
            if worker_allow_heavy is not None:
                payload["worker_allow_heavy"] = worker_allow_heavy
            if worker_max_fps is not None:
                payload["worker_max_fps"] = worker_max_fps
            if control_buffer is not None:
                payload["control_buffer_enabled"] = control_buffer
            # Temporal smoothing
            if temporal_ema is not None:
                payload["temporal_ema"] = temporal_ema
            r = client.put("/api/v1/realtime/vace/control-map-mode", json=payload)
            handle_error(r)
            output(r.json(), ctx)


@vace.command("external-stale-ms")
@click.argument("ms", required=False, type=float)
@click.pass_context
def vace_external_stale_ms(ctx, ms):
    """Get or set external control-map staleness threshold (ms).

    When in `vace_control_map_mode=external` + NDI input, Scope stalls generation
    if the newest NDI control frame is older than this threshold.
    """
    with get_client(ctx) as client:
        if ms is None:
            r = client.get("/api/v1/realtime/debug/fps")
            handle_error(r)
            fp = r.json().get("frame_processor") or {}
            ndi = fp.get("ndi") if isinstance(fp, dict) else None
            if not isinstance(ndi, dict):
                ndi = {}
            output({"vace_external_stale_ms": ndi.get("external_stale_ms")}, ctx)
            return

        ms = max(0.0, float(ms))
        r = client.post(
            "/api/v1/realtime/parameters",
            json={"vace_external_stale_ms": ms},
        )
        handle_error(r)
        output({"status": "ok", "vace_external_stale_ms": ms}, ctx)


@vace.command("external-resume-hard-cut")
@click.argument("enabled", required=False, type=click.Choice(["on", "off"]))
@click.pass_context
def vace_external_resume_hard_cut(ctx, enabled):
    """Get or set external-control resume hard cut behavior.

    When enabled, Scope forces `reset_cache=True` once when external control
    transitions from stale -> fresh. Disable this to preserve temporal coherence
    across brief external-control dropouts.
    """
    with get_client(ctx) as client:
        if enabled is None:
            r = client.get("/api/v1/realtime/debug/fps")
            handle_error(r)
            fp = r.json().get("frame_processor") or {}
            ndi = fp.get("ndi") if isinstance(fp, dict) else None
            if not isinstance(ndi, dict):
                ndi = {}
            output(
                {"vace_external_resume_hard_cut": ndi.get("external_resume_hard_cut")},
                ctx,
            )
            return

        val = enabled == "on"
        r = client.post(
            "/api/v1/realtime/parameters",
            json={"vace_external_resume_hard_cut": val},
        )
        handle_error(r)
        output({"status": "ok", "vace_external_resume_hard_cut": val}, ctx)


# --- Playlist ---


@cli.group()
@click.pass_context
def playlist(ctx):
    """Manage prompt playlist from caption files."""
    pass


@playlist.command("load")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--swap", nargs=2, help="Trigger swap: OLD NEW")
@click.pass_context
def playlist_load(ctx, file_path, swap):
    """Load prompts from a caption file.

    Examples:
        video-cli playlist load captions.txt
        video-cli playlist load captions.txt --swap "1988 Cel Animation" "Rankin/Bass Animagic Stop-Motion"
    """
    with get_client(ctx) as client:
        payload = {"file_path": str(Path(file_path).absolute())}
        if swap:
            payload["old_trigger"] = swap[0]
            payload["new_trigger"] = swap[1]
        r = client.post("/api/v1/realtime/playlist/load", json=payload)
        handle_error(r)
        output(r.json(), ctx)


@playlist.command("status")
@click.pass_context
def playlist_status(ctx):
    """Get current playlist state."""
    with get_client(ctx) as client:
        r = client.get("/api/v1/realtime/playlist")
        handle_error(r)
        output(r.json(), ctx)


@playlist.command("switch")
@click.option("--nav", is_flag=True, help="Enter nav mode after loading")
@click.option("--context", "-c", type=int, default=5, help="Lines of context for nav mode (default: 5)")
@click.pass_context
def playlist_switch(ctx, nav, context):
    """Interactive playlist switcher."""
    import shutil
    import termios
    import tty

    if not CONTENT_DIR.exists():
        click.echo("Content directory not found")
        return

    playlists = []
    for playlist_dir in sorted(CONTENT_DIR.iterdir()):
        if playlist_dir.is_dir():
            resolved = _resolve_playlist_path(playlist_dir.name)
            if resolved:
                try:
                    prompt_count = sum(1 for line in resolved.read_text().splitlines() if line.strip())
                except Exception:
                    prompt_count = 0
                playlists.append({"name": playlist_dir.name, "path": resolved, "count": prompt_count})

    if not playlists:
        click.echo("No playlists found")
        return

    def display():
        term_width = shutil.get_terminal_size().columns
        click.clear()
        click.echo("\n" + "=" * term_width)
        click.echo("  Playlist Switcher")
        click.echo("=" * term_width + "\n")
        for i, pl in enumerate(playlists[:9], 1):
            click.echo(f"   [{i}]  {pl['name']:20s}  ({pl['count']} prompts)")
        click.echo("\n" + "=" * term_width)
        click.echo("  Press 1-9 to load, q to quit")
        click.echo("=" * term_width + "\n")

    def get_char():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    display()
    while True:
        ch = get_char()
        if ch in ("q", "Q", "\x1b"):
            click.echo("\nCancelled.")
            return
        if ch.isdigit() and 1 <= int(ch) <= len(playlists):
            selected = playlists[int(ch) - 1]
            click.echo(f"\nLoading {selected['name']}...")
            with get_client(ctx) as client:
                r = client.post("/api/v1/realtime/playlist/load", json={"file_path": str(selected["path"].absolute())})
                if r.status_code >= 400:
                    click.echo(f"Error: {r.text}")
                    return
                click.echo(f"Loaded {selected['name']} ({selected['count']} prompts)")
            if nav:
                click.echo("Entering nav mode...\n")
                ctx.invoke(playlist_nav, context=context)
            return


@playlist.command("source-trigger")
@click.argument("trigger")
@click.pass_context
def playlist_source_trigger(ctx, trigger):
    """Set the source trigger phrase for auto-swap.

    The source trigger is what trigger phrase exists in your original prompts.
    This enables auto-trigger-swap when switching styles.

    Examples:
        # If your prompts contain "Hidari Animation":
        video-cli playlist source-trigger "Hidari Animation"

        # Now when you switch styles, the trigger will auto-swap:
        video-cli style set yeti  # "Hidari Animation" -> "Yeti Animation"
    """
    with get_client(ctx) as client:
        r = client.put("/api/v1/realtime/playlist/source-trigger", json={"trigger": trigger})
        handle_error(r)
        output(r.json(), ctx)


@playlist.command("preview")
@click.option("--context", "-c", type=int, default=2, help="Lines of context around current")
@click.pass_context
def playlist_preview(ctx, context):
    """Preview prompts around current position."""
    with get_client(ctx) as client:
        r = client.get("/api/v1/realtime/playlist/preview", params={"context": context})
        handle_error(r)
        output(r.json(), ctx)


@playlist.command("next")
@click.option("--apply/--no-apply", default=True, help="Apply prompt after navigating")
@click.pass_context
def playlist_next(ctx, apply):
    """Move to next prompt."""
    with get_client(ctx) as client:
        r = client.post("/api/v1/realtime/playlist/next", params={"apply": apply})
        handle_error(r)
        output(r.json(), ctx)


@playlist.command("prev")
@click.option("--apply/--no-apply", default=True, help="Apply prompt after navigating")
@click.pass_context
def playlist_prev(ctx, apply):
    """Move to previous prompt."""
    with get_client(ctx) as client:
        r = client.post("/api/v1/realtime/playlist/prev", params={"apply": apply})
        handle_error(r)
        output(r.json(), ctx)


@playlist.command("goto")
@click.argument("index", type=int)
@click.option("--apply/--no-apply", default=True, help="Apply prompt after navigating")
@click.pass_context
def playlist_goto(ctx, index, apply):
    """Go to a specific prompt index."""
    with get_client(ctx) as client:
        r = client.post("/api/v1/realtime/playlist/goto", json={"index": index}, params={"apply": apply})
        handle_error(r)
        output(r.json(), ctx)


@playlist.command("apply")
@click.pass_context
def playlist_apply(ctx):
    """Apply current prompt to generation."""
    with get_client(ctx) as client:
        r = client.post("/api/v1/realtime/playlist/apply")
        handle_error(r)
        output(r.json(), ctx)


@playlist.command("nav")
@click.option("--context", "-c", type=int, default=5, help="Lines of context around current (default: 5 = 11 total)")
@click.pass_context
def playlist_nav(ctx, context):
    """Interactive navigation mode with autoplay.

    Controls:
        →, n, l, SPACE  Next prompt
        ←, p            Previous prompt (stops autoplay)
        m               Toggle bookmark on current prompt (★)
        N               Jump to next bookmarked prompt
        P               Jump to previous bookmarked prompt
        B               Toggle bookmark filter (show only bookmarked)
        o               Toggle autoplay (default 5s interval)
        h/H             Toggle hard cut mode (reset cache on each transition)
        s               Toggle soft cut mode (temporary KV-bias override)
        t               Toggle embedding transition mode (temporal interpolation)
        T               Toggle transition method (linear ↔ slerp)
        x               One-shot hard cut (doesn't change mode)
        z               Randomize seed
        Z               Soft cut + new seed (same prompt, new variation)
        S               Set specific seed (prompts for number)
        *               Bookmark current seed
        #               Show seed info (current, history, bookmarks)
        b               Toggle blend mode (style switch without cache reset)
        1-5             Set soft cut bias (when soft cut active)
        !@#$%           Set soft cut duration in chunks (when soft cut active)
        6-0             Set transition chunks (1-5) (when transition active)
        +/-             Adjust autoplay speed (1-30s)
        g               Go to index (prompts for number)
        a               Apply current prompt
        j               Jiggle active prompt (generate variations)
        r               Refresh display
        q, ESC          Quit

    Jiggle mode:
        1-4             Apply variation (respects current hard/soft/transition mode)
        j               Regenerate variations (attentional mode)
        J               Regenerate variations (semantic mode, requires direction)
        d               Set/clear direction (blank clears)
        ESC             Cancel and return to playlist

    Changes are auto-applied by default.
    Hard cut mode resets the KV cache on each prompt change for clean scene transitions.
    Soft cut mode temporarily lowers kv_cache_attention_bias for faster adaptation without a full reset.
    Embedding transition mode interpolates embeddings over N chunks for smoother prompt morphs.
    Hard cut and embedding transition are mutually exclusive (hard cut resets state).
    One-shot hard cut (x) does a single cache reset without changing your current mode.
    """
    import builtins
    import os
    import select
    import shutil
    import termios
    import time
    import textwrap
    import tty

    def get_char_nonblocking(timeout=0.2):
        """Read a char with timeout. Returns None if no input."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            if select.select([fd], [], [], timeout)[0]:
                ch = os.read(fd, 1).decode("utf-8", errors="ignore")
                if ch == "\x1b":
                    extra = ""
                    for _ in range(5):
                        if select.select([fd], [], [], 0.05)[0]:
                            byte = os.read(fd, 1).decode("utf-8", errors="ignore")
                            extra += byte
                            if len(extra) >= 2 and extra[0] == "[" and extra[-1] in "ABCD":
                                break
                        else:
                            break
                    ch = ch + extra
                return ch
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def display_preview(
        client,
        autoplay=False,
        interval=5.0,
        hard_cut=False,
        soft_cut=False,
        soft_cut_bias=0.20,
        soft_cut_chunks=4,
        transition=False,
        transition_chunks=4,
        transition_method="slerp",
        blend_mode=True,
        context_lines=8,
        bookmark_filter=False,
    ):
        """Fetch and display preview."""
        import shutil

        term_width = shutil.get_terminal_size().columns

        params = {"context": context_lines}
        if bookmark_filter:
            params["bookmarks_only"] = True
        r = client.get("/api/v1/realtime/playlist/preview", params=params)
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("status") == "no_playlist":
            click.echo("\n  No playlist loaded. Use: video-cli playlist load <file>\n")
            return None

        click.echo("\n" + "=" * term_width)
        bookmarked_indices = data.get("bookmarked_indices", [])
        bookmark_count = len(bookmarked_indices)
        status = f"  Playlist: {data.get('total', 0)} prompts"
        if bookmark_count > 0:
            status += f"  [★ {bookmark_count} bookmarked]"
        if bookmark_filter:
            status += "  [FILTERED]"
        if autoplay:
            status += f"  [▶ AUTO {interval}s]"
        if blend_mode:
            status += "  [🌀 BLEND]"
        if hard_cut:
            status += "  [✂ HARD CUT]"
        if soft_cut:
            status += f"  [~ SOFT b={soft_cut_bias:.2f} c={soft_cut_chunks}]"
        if transition:
            method_label = "SLERP" if transition_method == "slerp" else "LERP"
            status += f"  [⟷ {method_label} c={transition_chunks}]"
        click.echo(status)
        click.echo("=" * term_width)

        # Calculate prompt display width
        prompt_width = term_width - 12  # Extra space for bookmark marker

        # Get prompts to show
        prompts_to_show = data.get("prompts", [])

        # Cap display (only when not in bookmark filter mode - we want to see all bookmarks)
        if not bookmark_filter:
            max_display = context_lines * 2 + 1
            if len(prompts_to_show) > max_display:
                prompts_to_show = prompts_to_show[:max_display]

        for item in prompts_to_show:
            is_current = item.get("current")
            is_bookmarked = item.get("bookmarked", False)
            marker = "▶ " if is_current else "  "
            bookmark = "★" if is_bookmarked else " "
            idx = item.get("index", 0)
            prompt = item.get("prompt", "")[:prompt_width]
            if is_current:
                click.echo(
                    click.style(f"{marker}{bookmark}[{idx:3d}] {prompt}", fg="green", bold=True)
                )
            elif is_bookmarked:
                click.echo(
                    click.style(f"{marker}{bookmark}[{idx:3d}] {prompt}", fg="yellow")
                )
            else:
                click.echo(f"{marker}{bookmark}[{idx:3d}] {prompt}")

        click.echo("=" * term_width)
        click.echo(
            "  ←/→ nav | N/P jump★ | m mark | B filter★ | g goto | a apply | o auto | +/- speed"
        )
        click.echo(
            "  h hard | s soft | t trans | T slerp | x cut! | b blend | z seed | j jiggle | D dir | q quit"
        )
        click.echo("=" * term_width + "\n")
        return data

    STASH_CAPTIONS_PATH = CONTENT_DIR / "_stash" / "Captioning" / "stash_captions.txt"

    def _normalize_stash_prompt(text: str) -> str:
        """Stash file is strictly one prompt per line."""
        return " ".join(text.replace("\r", "\n").replace("\n", " ").split()).strip()

    def append_to_jiggle_stash(prompt_text: str) -> None:
        """Append an applied jiggle variation to the stash playlist."""
        line = _normalize_stash_prompt(prompt_text)
        if not line:
            return
        STASH_CAPTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with STASH_CAPTIONS_PATH.open("a") as f:
            f.write(line + "\n")

    def fetch_jiggle_candidates(
        client: httpx.Client,
        prompt_text: str,
        *,
        count: int = 4,
        direction: str | None = None,
        mode: str = "attentional",
        intensity: float = 0.3,
        timeout_s: float = 10.0,
    ) -> dict[str, object]:
        """Fetch jiggle candidates from API using a single count=N call."""
        body: dict[str, object] = {
            "prompt": prompt_text,
            "count": count,
            "intensity": intensity,
            "mode": mode,
        }
        if direction:
            body["direction"] = direction
        try:
            r = client.post("/api/v1/prompt/jiggle", json=body, timeout=timeout_s)
        except Exception as e:
            return {"ok": False, "error": str(e)}

        if r.status_code != 200:
            return {"ok": False, "error": r.text, "status_code": r.status_code}

        try:
            data = r.json()
        except Exception:
            return {"ok": False, "error": r.text}

        status = data.get("status")
        original_prompt = data.get("original_prompt")
        variations = data.get("variations")
        if not isinstance(original_prompt, str):
            original_prompt = prompt_text
        if not isinstance(variations, list):
            variations = []
        variations = [v for v in variations if isinstance(v, str) and v.strip()]
        return {
            "ok": True,
            "status": status,
            "original_prompt": original_prompt,
            "variations": variations,
        }

    def display_jiggle_view(
        original: str,
        candidates: list[str],
        *,
        direction: str | None,
        intensity: float,
        mode: str,
        preview_index: int | None = None,  # Currently previewed candidate (0-3)
    ) -> None:
        """Render a full-screen jiggle candidate view."""
        term_width = shutil.get_terminal_size().columns
        click.clear()

        header = f"JIGGLE MODE [{mode}] [intensity={intensity:.1f}]"
        if direction:
            header += f' [dir="{direction[:20]}"]'
        click.echo(click.style(header, fg="cyan", bold=True))
        click.echo("=" * min(term_width, 70))

        click.echo("Original:")
        for line in textwrap.wrap(original, width=max(20, min(term_width - 4, 66)))[:2]:
            click.echo(f"  {line}")
        click.echo("-" * min(term_width, 70))

        for i, candidate in enumerate(candidates[:4], 1):
            is_previewing = (preview_index == i - 1)
            marker = " ◀──" if is_previewing else ""
            style = {"fg": "green", "bold": True} if is_previewing else {}
            click.echo(click.style(f"[{i}]{marker}", **style))
            for line in textwrap.wrap(candidate, width=max(20, min(term_width - 4, 66)))[:2]:
                click.echo(click.style(f"  {line}", **style))

        click.echo("=" * min(term_width, 70))
        click.echo("1-4: preview | Enter: confirm | j: regen | d: direction | J: semantic")
        click.echo("ESC: cancel | q: quit")

    def _apply_jiggle_prompt(
        client: httpx.Client,
        prompt_text: str,
        *,
        hard_cut: bool,
        soft_cut: bool,
        soft_cut_bias: float,
        soft_cut_chunks: int,
        transition: bool,
        transition_chunks: int,
        transition_method: str,
    ) -> httpx.Response:
        """Apply a prompt respecting current transition mode toggles."""
        if hard_cut:
            return client.post("/api/v1/realtime/hard-cut", json={"prompt": prompt_text})

        if transition:
            prompt_payload = [{"text": prompt_text, "weight": 1.0}]
            method = transition_method if transition_method in ("linear", "slerp") else "linear"
            msg: dict[str, object] = {
                "transition": {
                    "target_prompts": prompt_payload,
                    "num_steps": transition_chunks,
                    "temporal_interpolation_method": method,
                }
            }
            if soft_cut:
                msg["_rcp_soft_transition"] = {
                    "temp_bias": soft_cut_bias,
                    "num_chunks": soft_cut_chunks,
                }
            return client.post("/api/v1/realtime/parameters", json=msg)

        if soft_cut:
            return client.post(
                "/api/v1/realtime/soft-cut",
                json={
                    "prompt": prompt_text,
                    "temp_bias": soft_cut_bias,
                    "num_chunks": soft_cut_chunks,
                },
            )

        return client.put("/api/v1/realtime/prompt", json={"prompt": prompt_text})

    click.echo("\nPlaylist Navigation Mode")
    click.echo("Press q or ESC to quit\n")

    # Autoplay state
    autoplay = False
    autoplay_interval = 5.0
    last_advance = time.time()

    # Hard cut state - when enabled, all transitions reset the KV cache
    hard_cut = False

    # Soft cut state - when enabled, transitions temporarily lower KV cache bias
    soft_cut = False
    soft_cut_bias = 0.20  # Default temp bias
    soft_cut_chunks = 4  # Default duration

    # Transition (embedding interpolation) state
    transition = False
    transition_chunks = 4  # Default number of chunks to interpolate over
    transition_method = "slerp"  # linear or slerp

    # Blend mode state - when enabled, style switches don't reset cache (artistic artifacts)
    blend_mode = True

    # Bookmark filter - when enabled, only show bookmarked prompts in display
    bookmark_filter = False

    # Jiggle mode state (prompt variations)
    jiggle_mode = False
    jiggle_candidates: list[str] = []
    jiggle_original = ""
    jiggle_direction: str | None = None  # Persists across jiggle sessions
    jiggle_intensity = 0.3
    jiggle_view_mode = "attentional"
    jiggle_preview_index: int | None = None  # Currently previewed candidate (0-3)
    was_autoplay = False  # Restore autoplay on exit

    with get_client(ctx) as client:
        # Fetch current blend mode from server
        try:
            r = client.get("/api/v1/realtime/style/blend-mode")
            if r.status_code == 200:
                blend_mode = r.json().get("blend_mode", False)
        except Exception:
            pass  # Use default if server not ready

        if display_preview(
            client, autoplay, autoplay_interval, hard_cut,
            soft_cut, soft_cut_bias, soft_cut_chunks,
            transition, transition_chunks, transition_method, blend_mode
        ) is None:
            return

        while True:
            try:
                ch = get_char_nonblocking(timeout=0.2)

                if ch is not None:
                    # Quit
                    if ch in ("q", "Q", "\x03"):
                        click.echo("\nExiting navigation mode.\n")
                        break
                    elif ch == "\x1b" and not jiggle_mode:
                        click.echo("\nExiting navigation mode.\n")
                        break

                    # === JIGGLE MODE HANDLERS (must run before other mode handlers) ===
                    elif jiggle_mode:
                        # Cancel jiggle (ESC only; keep q as quit)
                        # If previewing, restore original prompt before exiting
                        if ch == "\x1b":
                            cancel_msg = "  Cancelled"
                            if jiggle_preview_index is not None:
                                # Restore original prompt
                                try:
                                    _apply_jiggle_prompt(
                                        client,
                                        jiggle_original,
                                        hard_cut=hard_cut,
                                        soft_cut=soft_cut,
                                        soft_cut_bias=soft_cut_bias,
                                        soft_cut_chunks=soft_cut_chunks,
                                        transition=transition,
                                        transition_chunks=transition_chunks,
                                        transition_method=transition_method,
                                    )
                                    cancel_msg = "  Cancelled (restored original)"
                                except Exception:
                                    cancel_msg = "  Cancelled (failed to restore original)"
                            jiggle_mode = False
                            jiggle_candidates = []
                            jiggle_preview_index = None
                            autoplay = was_autoplay
                            click.clear()
                            display_preview(
                                client,
                                autoplay,
                                autoplay_interval,
                                hard_cut,
                                soft_cut,
                                soft_cut_bias,
                                soft_cut_chunks,
                                transition,
                                transition_chunks,
                                transition_method,
                                blend_mode,
                                context,
                                bookmark_filter,
                            )
                            click.echo(cancel_msg)
                            continue

                        # Preview candidate 1-4 (apply but stay in jiggle mode)
                        if ch in "1234":
                            idx = int(ch) - 1
                            if idx < len(jiggle_candidates):
                                selected = jiggle_candidates[idx]
                                try:
                                    r2 = _apply_jiggle_prompt(
                                        client,
                                        selected,
                                        hard_cut=hard_cut,
                                        soft_cut=soft_cut,
                                        soft_cut_bias=soft_cut_bias,
                                        soft_cut_chunks=soft_cut_chunks,
                                        transition=transition,
                                        transition_chunks=transition_chunks,
                                        transition_method=transition_method,
                                    )
                                    if r2.status_code != 200:
                                        click.echo(f"  Failed to preview [{ch}]: {r2.text}")
                                    else:
                                        jiggle_preview_index = idx
                                        display_jiggle_view(
                                            jiggle_original,
                                            jiggle_candidates,
                                            direction=jiggle_direction,
                                            intensity=jiggle_intensity,
                                            mode=jiggle_view_mode,
                                            preview_index=jiggle_preview_index,
                                        )
                                        click.echo(f"  Previewing [{ch}] - press Enter to confirm, or try another")
                                except Exception as e:
                                    click.echo(f"  Failed to preview [{ch}]: {e}")
                            else:
                                click.echo(f"  No candidate [{ch}]")
                            continue

                        # Confirm current preview (Enter)
                        if ch in ("\r", "\n"):
                            if jiggle_preview_index is not None:
                                selected = jiggle_candidates[jiggle_preview_index]
                                # Save to stash for later recall
                                stash_msg = ""
                                try:
                                    append_to_jiggle_stash(selected)
                                except Exception as e:
                                    stash_msg = f" (stash save failed: {e})"
                                else:
                                    stash_msg = " (saved to stash)"

                                jiggle_mode = False
                                jiggle_candidates = []
                                jiggle_preview_index = None
                                autoplay = was_autoplay
                                click.clear()
                                display_preview(
                                    client,
                                    autoplay,
                                    autoplay_interval,
                                    hard_cut,
                                    soft_cut,
                                    soft_cut_bias,
                                    soft_cut_chunks,
                                    transition,
                                    transition_chunks,
                                    transition_method,
                                    blend_mode,
                                    context,
                                    bookmark_filter,
                                )
                                click.echo(f"  ✓ Confirmed{stash_msg}")
                            else:
                                click.echo("  No preview selected - press 1-4 to preview first")
                            continue

                        # Regenerate (attentional) - clears preview
                        if ch == "j":
                            jiggle_preview_index = None
                            jiggle_view_mode = "attentional"
                            click.echo("  Regenerating...")
                            result = fetch_jiggle_candidates(
                                client,
                                jiggle_original,
                                count=4,
                                direction=jiggle_direction,
                                mode=jiggle_view_mode,
                                intensity=jiggle_intensity,
                            )
                            if not result.get("ok"):
                                click.echo(f"  Failed to regenerate: {result.get('error', 'unknown error')}")
                            else:
                                jiggle_candidates = list(result.get("variations", []))  # type: ignore[list-item]
                                display_jiggle_view(
                                    str(result.get("original_prompt", jiggle_original)),
                                    jiggle_candidates,
                                    direction=jiggle_direction,
                                    intensity=jiggle_intensity,
                                    mode=jiggle_view_mode,
                                    preview_index=jiggle_preview_index,
                                )
                            continue

                        # Regenerate (semantic) - clears preview
                        if ch == "J":
                            if not jiggle_direction:
                                click.echo("  Semantic mode requires direction - press 'd' first")
                                continue
                            jiggle_preview_index = None
                            jiggle_view_mode = "semantic"
                            click.echo("  Regenerating (semantic)...")
                            result = fetch_jiggle_candidates(
                                client,
                                jiggle_original,
                                count=4,
                                direction=jiggle_direction,
                                mode=jiggle_view_mode,
                                intensity=jiggle_intensity,
                            )
                            if not result.get("ok"):
                                click.echo(f"  Failed to regenerate: {result.get('error', 'unknown error')}")
                            else:
                                jiggle_candidates = list(result.get("variations", []))  # type: ignore[list-item]
                                display_jiggle_view(
                                    str(result.get("original_prompt", jiggle_original)),
                                    jiggle_candidates,
                                    direction=jiggle_direction,
                                    intensity=jiggle_intensity,
                                    mode=jiggle_view_mode,
                                    preview_index=jiggle_preview_index,
                                )
                            continue

                        # Direction input + regen (attentional) - clears preview
                        if ch == "d":
                            click.echo("\nDirection (blank to clear): ", nl=False)
                            try:
                                direction_input = builtins.input().strip()
                            except EOFError:
                                direction_input = ""
                            jiggle_direction = direction_input or None
                            jiggle_preview_index = None
                            jiggle_view_mode = "attentional"
                            click.echo("  Regenerating...")
                            result = fetch_jiggle_candidates(
                                client,
                                jiggle_original,
                                count=4,
                                direction=jiggle_direction,
                                mode=jiggle_view_mode,
                                intensity=jiggle_intensity,
                            )
                            if not result.get("ok"):
                                click.echo(f"  Failed to regenerate: {result.get('error', 'unknown error')}")
                            else:
                                jiggle_candidates = list(result.get("variations", []))  # type: ignore[list-item]
                                display_jiggle_view(
                                    str(result.get("original_prompt", jiggle_original)),
                                    jiggle_candidates,
                                    direction=jiggle_direction,
                                    intensity=jiggle_intensity,
                                    mode=jiggle_view_mode,
                                    preview_index=jiggle_preview_index,
                                )
                            continue

                        click.echo("  1-4: preview | Enter: confirm | j: regen | d: direction | J: semantic | ESC: cancel")
                        continue

                    # Enter jiggle mode
                    elif ch == "j":
                        # Jiggle the active (currently generating) prompt, not the playlist line.
                        try:
                            state_r = client.get("/api/v1/realtime/state")
                        except Exception as e:
                            click.echo(f"  Failed to fetch active prompt: {e}")
                            continue
                        if state_r.status_code != 200:
                            click.echo(f"  Failed to fetch active prompt: {state_r.text}")
                            continue
                        try:
                            state = state_r.json()
                        except Exception:
                            click.echo("  Failed to parse active prompt state")
                            continue

                        active_prompt = state.get("compiled_prompt") or state.get("prompt")
                        if not isinstance(active_prompt, str) or not active_prompt.strip():
                            click.echo("  No active prompt to jiggle")
                            continue

                        click.echo("  Generating variations...")
                        result = fetch_jiggle_candidates(
                            client,
                            active_prompt,
                            count=4,
                            direction=jiggle_direction,
                            mode="attentional",
                            intensity=jiggle_intensity,
                        )
                        if not result.get("ok"):
                            click.echo(f"  Failed to generate variations: {result.get('error', 'unknown error')}")
                            continue

                        if result.get("status") == "unchanged":
                            click.echo("  [Jiggle unavailable - GEMINI_API_KEY not set]")
                            continue

                        candidates = list(result.get("variations", []))  # type: ignore[list-item]
                        if not candidates:
                            click.echo("  No variations generated")
                            continue

                        was_autoplay = autoplay
                        autoplay = False
                        jiggle_mode = True
                        jiggle_original = str(result.get("original_prompt", active_prompt))
                        jiggle_candidates = candidates
                        jiggle_view_mode = "attentional"
                        jiggle_preview_index = None  # No preview yet
                        display_jiggle_view(
                            jiggle_original,
                            jiggle_candidates,
                            direction=jiggle_direction,
                            intensity=jiggle_intensity,
                            mode=jiggle_view_mode,
                            preview_index=jiggle_preview_index,
                        )
                        continue

                    # Set jiggle direction before entering jiggle mode
                    elif ch == "D":
                        click.echo("\nJiggle direction (blank to clear): ", nl=False)
                        try:
                            direction_input = builtins.input().strip()
                        except EOFError:
                            direction_input = ""
                        jiggle_direction = direction_input or None
                        if jiggle_direction:
                            click.echo(f'  Direction set: "{jiggle_direction}" - press j to jiggle')
                        else:
                            click.echo("  Direction cleared")
                        continue

                    # Toggle autoplay
                    elif ch == "o":
                        autoplay = not autoplay
                        last_advance = time.time()
                        display_preview(
                            client, autoplay, autoplay_interval, hard_cut,
                            soft_cut, soft_cut_bias, soft_cut_chunks,
                            transition, transition_chunks, transition_method, blend_mode, context,
                        bookmark_filter
                        )

                    # Toggle hard cut mode (mutually exclusive with soft cut)
                    elif ch in ("H", "h"):
                        hard_cut = not hard_cut
                        if hard_cut:
                            soft_cut = False  # Mutually exclusive
                            transition = False  # Transition requires continuity
                        status = "ON - transitions will reset cache" if hard_cut else "OFF"
                        click.echo(f"  ✂ Hard cut: {status}")
                        display_preview(
                            client, autoplay, autoplay_interval, hard_cut,
                            soft_cut, soft_cut_bias, soft_cut_chunks,
                            transition, transition_chunks, transition_method, blend_mode, context,
                        bookmark_filter
                        )

                    # Toggle soft cut mode (mutually exclusive with hard cut)
                    elif ch == "s":
                        soft_cut = not soft_cut
                        if soft_cut:
                            hard_cut = False  # Mutually exclusive
                        if soft_cut:
                            status = f"ON (bias={soft_cut_bias}, chunks={soft_cut_chunks})"
                        else:
                            status = "OFF"
                        click.echo(f"  ~ Soft cut: {status}")
                        display_preview(
                            client, autoplay, autoplay_interval, hard_cut,
                            soft_cut, soft_cut_bias, soft_cut_chunks,
                            transition, transition_chunks, transition_method, blend_mode, context,
                        bookmark_filter
                        )

                    # Bias adjustment (1-5 keys when soft_cut active)
                    elif soft_cut and ch in "12345":
                        bias_map = {"1": 0.05, "2": 0.1, "3": 0.15, "4": 0.2, "5": 0.25}
                        soft_cut_bias = bias_map[ch]
                        click.echo(f"  ~ Soft cut bias: {soft_cut_bias}")
                        display_preview(
                            client, autoplay, autoplay_interval, hard_cut,
                            soft_cut, soft_cut_bias, soft_cut_chunks,
                            transition, transition_chunks, transition_method, blend_mode, context,
                        bookmark_filter
                        )

                    # Chunk adjustment (Shift+1-5 = !, @, #, $, % when soft_cut active)
                    elif soft_cut and ch in "!@#$%":
                        chunk_map = {"!": 1, "@": 2, "#": 3, "$": 4, "%": 5}
                        soft_cut_chunks = chunk_map[ch]
                        click.echo(f"  ~ Soft cut chunks: {soft_cut_chunks}")
                        display_preview(
                            client, autoplay, autoplay_interval, hard_cut,
                            soft_cut, soft_cut_bias, soft_cut_chunks,
                            transition, transition_chunks, transition_method, blend_mode, context,
                        bookmark_filter
                        )

                    # Toggle transition (embedding interpolation) mode
                    elif ch == "t":
                        transition = not transition
                        if transition:
                            hard_cut = False  # Transition requires continuity
                            status = f"ON (chunks={transition_chunks}, method={transition_method})"
                        else:
                            status = "OFF"
                        click.echo(f"  ⟷ Transition: {status}")
                        display_preview(
                            client, autoplay, autoplay_interval, hard_cut,
                            soft_cut, soft_cut_bias, soft_cut_chunks,
                            transition, transition_chunks, transition_method, blend_mode, context,
                        bookmark_filter
                        )

                    # Toggle transition method (linear/slerp)
                    elif ch == "T":
                        transition_method = "slerp" if transition_method == "linear" else "linear"
                        click.echo(f"  ⟷ Transition method: {transition_method}")
                        display_preview(
                            client, autoplay, autoplay_interval, hard_cut,
                            soft_cut, soft_cut_bias, soft_cut_chunks,
                            transition, transition_chunks, transition_method, blend_mode, context,
                        bookmark_filter
                        )

                    # Transition chunks adjustment (6-0 when transition active)
                    elif transition and ch in "67890":
                        chunk_map = {"6": 1, "7": 2, "8": 3, "9": 4, "0": 5}
                        transition_chunks = chunk_map[ch]
                        click.echo(f"  ⟷ Transition chunks: {transition_chunks}")
                        display_preview(
                            client, autoplay, autoplay_interval, hard_cut,
                            soft_cut, soft_cut_bias, soft_cut_chunks,
                            transition, transition_chunks, transition_method, blend_mode, context,
                        bookmark_filter
                        )

                    # Toggle blend mode (style switch without cache reset)
                    elif ch == "b":
                        blend_mode = not blend_mode
                        # Sync with server
                        try:
                            r = client.put(
                                "/api/v1/realtime/style/blend-mode",
                                json={"enabled": blend_mode}
                            )
                            if r.status_code == 200:
                                status = "ON - style switches will create blend artifacts" if blend_mode else "OFF - clean transitions"
                                click.echo(f"  🌀 Blend mode: {status}")
                            else:
                                blend_mode = not blend_mode  # Revert on failure
                                click.echo("  Failed to set blend mode")
                        except Exception as e:
                            blend_mode = not blend_mode
                            click.echo(f"  Error setting blend mode: {e}")
                        display_preview(
                            client, autoplay, autoplay_interval, hard_cut,
                            soft_cut, soft_cut_bias, soft_cut_chunks,
                            transition, transition_chunks, transition_method, blend_mode, context,
                        bookmark_filter
                        )

                    # Adjust speed
                    elif ch in ("+", "=", "]"):
                        autoplay_interval = max(1.0, autoplay_interval - 1.0)
                        click.echo(f"  Interval: {autoplay_interval}s")
                    elif ch in ("-", "_", "["):
                        autoplay_interval = min(30.0, autoplay_interval + 1.0)
                        click.echo(f"  Interval: {autoplay_interval}s")

                    # Next (uses bookmark nav when filter is ON)
                    elif ch in ("\x1b[C", "n", "l", " "):
                        params = {"apply": True}
                        if hard_cut:
                            params["hard_cut"] = True
                        if soft_cut:
                            params["soft_cut"] = True
                            params["soft_cut_bias"] = soft_cut_bias
                            params["soft_cut_chunks"] = soft_cut_chunks
                        if transition:
                            params["transition"] = True
                            params["transition_chunks"] = transition_chunks
                            params["transition_method"] = transition_method
                        endpoint = "/api/v1/realtime/playlist/bookmark/next" if bookmark_filter else "/api/v1/realtime/playlist/next"
                        r = client.post(endpoint, params=params)
                        if r.status_code == 200:
                            display_preview(
                                client, autoplay, autoplay_interval, hard_cut,
                                soft_cut, soft_cut_bias, soft_cut_chunks,
                                transition, transition_chunks, transition_method, blend_mode, context,
                            bookmark_filter
                            )
                        elif r.status_code == 400 and "No bookmarks" in r.text:
                            click.echo("  ⚠ No bookmarks set. Press 'm' to bookmark prompts.")
                        last_advance = time.time()

                    # Previous (stops autoplay, uses bookmark nav when filter is ON)
                    elif ch in ("\x1b[D", "p"):
                        params = {"apply": True}
                        if hard_cut:
                            params["hard_cut"] = True
                        if soft_cut:
                            params["soft_cut"] = True
                            params["soft_cut_bias"] = soft_cut_bias
                            params["soft_cut_chunks"] = soft_cut_chunks
                        if transition:
                            params["transition"] = True
                            params["transition_chunks"] = transition_chunks
                            params["transition_method"] = transition_method
                        endpoint = "/api/v1/realtime/playlist/bookmark/prev" if bookmark_filter else "/api/v1/realtime/playlist/prev"
                        r = client.post(endpoint, params=params)
                        if r.status_code == 200:
                            display_preview(
                                client, autoplay, autoplay_interval, hard_cut,
                                soft_cut, soft_cut_bias, soft_cut_chunks,
                                transition, transition_chunks, transition_method, blend_mode, context,
                            bookmark_filter
                            )
                        elif r.status_code == 400 and "No bookmarks" in r.text:
                            click.echo("  ⚠ No bookmarks set. Press 'm' to bookmark prompts.")
                        last_advance = time.time()
                        if autoplay:
                            autoplay = False
                            click.echo("  ⏸ Autoplay stopped")

                    # Goto
                    elif ch == "g":
                        click.echo("\nGoto index: ", nl=False)
                        fd = sys.stdin.fileno()
                        old_settings = termios.tcgetattr(fd)
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                        try:
                            idx_str = builtins.input()
                            idx = int(idx_str)
                            params = {"apply": True}
                            if hard_cut:
                                params["hard_cut"] = True
                            if soft_cut:
                                params["soft_cut"] = True
                                params["soft_cut_bias"] = soft_cut_bias
                                params["soft_cut_chunks"] = soft_cut_chunks
                            if transition:
                                params["transition"] = True
                                params["transition_chunks"] = transition_chunks
                                params["transition_method"] = transition_method
                            r = client.post(
                                "/api/v1/realtime/playlist/goto",
                                json={"index": idx},
                                params=params,
                            )
                            if r.status_code == 200:
                                display_preview(
                                    client, autoplay, autoplay_interval, hard_cut,
                                    soft_cut, soft_cut_bias, soft_cut_chunks,
                                    transition, transition_chunks, transition_method
                                )
                            last_advance = time.time()
                        except ValueError:
                            click.echo("Invalid index")
                        except EOFError:
                            pass

                    # Apply (with hard/soft cut/transition if enabled)
                    elif ch == "a":
                        params = {}
                        if hard_cut:
                            params["hard_cut"] = True
                        if soft_cut:
                            params["soft_cut"] = True
                            params["soft_cut_bias"] = soft_cut_bias
                            params["soft_cut_chunks"] = soft_cut_chunks
                        if transition:
                            params["transition"] = True
                            params["transition_chunks"] = transition_chunks
                            params["transition_method"] = transition_method
                        r = client.post("/api/v1/realtime/playlist/apply", params=params)
                        if r.status_code == 200:
                            msg = "✓ Prompt applied"
                            if hard_cut:
                                msg += " (hard cut)"
                            if soft_cut:
                                msg += f" (soft cut b={soft_cut_bias})"
                            if transition:
                                msg += f" (transition c={transition_chunks})"
                            click.echo(f"  {msg}")

                    # One-shot hard cut (doesn't change mode)
                    elif ch == "x":
                        r = client.post("/api/v1/realtime/hard-cut")
                        if r.status_code == 200:
                            click.echo("  ✂ One-shot hard cut applied")

                    # Randomize seed
                    elif ch == "z":
                        import random
                        new_seed = random.randint(0, 2**32 - 1)
                        r = client.post("/api/v1/realtime/parameters", json={"base_seed": new_seed})
                        if r.status_code == 200:
                            click.echo(f"  🎲 New seed: {new_seed}")
                        else:
                            click.echo(f"  Error setting seed: {r.text}")

                    # Soft cut with new seed (same prompt, new variation)
                    elif ch == "Z":
                        import random
                        new_seed = random.randint(0, 2**32 - 1)
                        # Set new seed
                        r1 = client.post("/api/v1/realtime/parameters", json={"base_seed": new_seed})
                        # Apply current prompt with soft cut
                        params = {
                            "soft_cut": True,
                            "soft_cut_bias": soft_cut_bias,
                            "soft_cut_chunks": soft_cut_chunks,
                        }
                        r2 = client.post("/api/v1/realtime/playlist/apply", params=params)
                        if r1.status_code == 200 and r2.status_code == 200:
                            click.echo(f"  🎲~ Soft cut with new seed: {new_seed}")
                        else:
                            click.echo(f"  Error: seed={r1.status_code}, apply={r2.status_code}")

                    # Set specific seed
                    elif ch == "S":
                        click.echo("  Enter seed: ", nl=False)
                        # Temporarily restore terminal for input
                        import termios
                        fd = sys.stdin.fileno()
                        old_settings = termios.tcgetattr(fd)
                        try:
                            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                            seed_str = builtins.input()
                            if seed_str.strip():
                                try:
                                    new_seed = int(seed_str.strip())
                                    r = client.post("/api/v1/realtime/parameters", json={"base_seed": new_seed})
                                    if r.status_code == 200:
                                        click.echo(f"  🎲 Seed set: {new_seed}")
                                    else:
                                        click.echo(f"  Error: {r.text}")
                                except ValueError:
                                    click.echo("  Invalid seed (must be integer)")
                        finally:
                            pass  # Will re-enter raw mode on next iteration

                    # Bookmark current seed
                    elif ch == "*":
                        r = client.post("/api/v1/realtime/seed/bookmark")
                        if r.status_code == 200:
                            data = r.json()
                            click.echo(f"  ⭐ Bookmarked seed: {data.get('bookmarked')}")
                        else:
                            click.echo(f"  Error: {r.text}")

                    # Show seed info (current, history, bookmarks)
                    elif ch == "#":
                        r = client.get("/api/v1/realtime/seed")
                        if r.status_code == 200:
                            data = r.json()
                            click.echo(f"  Current: {data.get('current')}")
                            click.echo(f"  History: {data.get('history', [])}")
                            click.echo(f"  Bookmarks: {data.get('bookmarks', [])}")
                        else:
                            click.echo(f"  Error: {r.text}")

                    # Toggle prompt bookmark
                    elif ch == "m":
                        r = client.post("/api/v1/realtime/playlist/bookmark")
                        if r.status_code == 200:
                            data = r.json()
                            if data.get("status") == "bookmarked":
                                click.echo(f"  ★ Bookmarked prompt {data.get('current_index')}")
                            else:
                                click.echo(f"  ☆ Unbookmarked prompt {data.get('current_index')}")
                            display_preview(
                                client, autoplay, autoplay_interval, hard_cut,
                                soft_cut, soft_cut_bias, soft_cut_chunks,
                                transition, transition_chunks, transition_method, blend_mode, context,
                            bookmark_filter
                            )
                        else:
                            click.echo(f"  Error: {r.text}")

                    # Next bookmarked prompt
                    elif ch == "N":
                        params = {"apply": True}
                        if hard_cut:
                            params["hard_cut"] = True
                        if soft_cut:
                            params["soft_cut"] = True
                            params["soft_cut_bias"] = soft_cut_bias
                            params["soft_cut_chunks"] = soft_cut_chunks
                        if transition:
                            params["transition"] = True
                            params["transition_chunks"] = transition_chunks
                            params["transition_method"] = transition_method
                        r = client.post("/api/v1/realtime/playlist/bookmark/next", params=params)
                        if r.status_code == 200:
                            display_preview(
                                client, autoplay, autoplay_interval, hard_cut,
                                soft_cut, soft_cut_bias, soft_cut_chunks,
                                transition, transition_chunks, transition_method, blend_mode, context,
                            bookmark_filter
                            )
                        elif r.status_code == 400 and "No bookmarks" in r.text:
                            click.echo("  ⚠ No bookmarks set. Press 'm' to bookmark prompts.")
                        else:
                            click.echo(f"  Error: {r.text}")

                    # Previous bookmarked prompt
                    elif ch == "P":
                        params = {"apply": True}
                        if hard_cut:
                            params["hard_cut"] = True
                        if soft_cut:
                            params["soft_cut"] = True
                            params["soft_cut_bias"] = soft_cut_bias
                            params["soft_cut_chunks"] = soft_cut_chunks
                        if transition:
                            params["transition"] = True
                            params["transition_chunks"] = transition_chunks
                            params["transition_method"] = transition_method
                        r = client.post("/api/v1/realtime/playlist/bookmark/prev", params=params)
                        if r.status_code == 200:
                            display_preview(
                                client, autoplay, autoplay_interval, hard_cut,
                                soft_cut, soft_cut_bias, soft_cut_chunks,
                                transition, transition_chunks, transition_method, blend_mode, context,
                                bookmark_filter
                            )
                        elif r.status_code == 400 and "No bookmarks" in r.text:
                            click.echo("  ⚠ No bookmarks set. Press 'm' to bookmark prompts.")
                        else:
                            click.echo(f"  Error: {r.text}")

                    # Toggle bookmark filter view
                    elif ch == "B":
                        bookmark_filter = not bookmark_filter
                        if bookmark_filter:
                            click.echo("  ★ Bookmark filter ON - showing only bookmarked prompts")
                        else:
                            click.echo("  ★ Bookmark filter OFF - showing all prompts")
                        display_preview(
                            client, autoplay, autoplay_interval, hard_cut,
                            soft_cut, soft_cut_bias, soft_cut_chunks,
                            transition, transition_chunks, transition_method, blend_mode, context,
                            bookmark_filter
                        )

                    # Refresh
                    elif ch == "r":
                        display_preview(
                            client, autoplay, autoplay_interval, hard_cut,
                            soft_cut, soft_cut_bias, soft_cut_chunks,
                            transition, transition_chunks, transition_method, blend_mode, context,
                        bookmark_filter
                        )

                # Autoplay advance
                if autoplay and (time.time() - last_advance) >= autoplay_interval:
                    params = {"apply": True}
                    if hard_cut:
                        params["hard_cut"] = True
                    if soft_cut:
                        params["soft_cut"] = True
                        params["soft_cut_bias"] = soft_cut_bias
                        params["soft_cut_chunks"] = soft_cut_chunks
                    if transition:
                        params["transition"] = True
                        params["transition_chunks"] = transition_chunks
                        params["transition_method"] = transition_method
                    r = client.post("/api/v1/realtime/playlist/next", params=params)
                    if r.status_code == 200:
                        data = r.json()
                        if not data.get("has_next", False):
                            autoplay = False
                            click.echo("  ⏹ End of playlist")
                        display_preview(
                            client, autoplay, autoplay_interval, hard_cut,
                            soft_cut, soft_cut_bias, soft_cut_chunks,
                            transition, transition_chunks, transition_method, blend_mode, context,
                        bookmark_filter
                        )
                    last_advance = time.time()

            except KeyboardInterrupt:
                click.echo("\nExiting navigation mode.\n")
                break


# --- Input Sources ---


@cli.group()
@click.pass_context
def input(ctx):
    """Manage video input sources (NDI, Spout, WebRTC)."""
    pass


@input.group()
@click.pass_context
def ndi(ctx):
    """NDI input control."""
    pass


@ndi.command("enable")
@click.argument("source", default="")
@click.option("--extra-ips", "-e", multiple=True, help="Extra IPs to probe (e.g., Tailscale IPs)")
@click.pass_context
def ndi_enable(ctx, source, extra_ips):
    """Enable NDI receiver.

    SOURCE is a substring to match against available NDI sources.
    Use --extra-ips for Tailscale or cross-subnet discovery.

    Examples:
        video-cli input ndi enable DepthOutput -e 100.70.189.4
        video-cli input ndi enable "QUIXOTRON"
    """
    with get_client(ctx) as client:
        payload = {
            "ndi_receiver": {
                "enabled": True,
                "source": source,
            }
        }
        if extra_ips:
            payload["ndi_receiver"]["extra_ips"] = list(extra_ips)
        r = client.post("/api/v1/realtime/parameters", json=payload)
        handle_error(r)
        result = {"status": "ndi_enabled", "source": source or "(any)"}
        if extra_ips:
            result["extra_ips"] = list(extra_ips)
        output(result, ctx)


@ndi.command("disable")
@click.pass_context
def ndi_disable(ctx):
    """Disable NDI receiver."""
    with get_client(ctx) as client:
        payload = {"ndi_receiver": {"enabled": False}}
        r = client.post("/api/v1/realtime/parameters", json=payload)
        handle_error(r)
        output({"status": "ndi_disabled"}, ctx)


@ndi.command("list")
@click.option("--extra-ips", "-e", multiple=True, help="Extra IPs to probe")
@click.option("--timeout", "-t", default=3000, help="Discovery timeout in ms")
@click.pass_context
def ndi_list(ctx, extra_ips, timeout):
    """List available NDI sources."""
    try:
        from scope.server.ndi.finder import list_sources
    except ImportError:
        output({"error": "NDI module not available"}, ctx)
        return

    sources = list_sources(
        timeout_ms=timeout,
        extra_ips=list(extra_ips) if extra_ips else None,
        show_local_sources=True,
    )
    result = {
        "sources": [{"name": s.name, "url": s.url_address} for s in sources],
        "count": len(sources),
    }
    output(result, ctx)


@ndi.command("probe")
@click.argument("source", default="")
@click.option("--extra-ips", "-e", multiple=True, help="Extra IPs to probe (e.g., Tailscale IPs)")
@click.option("--discover-timeout", default=3000, help="Discovery timeout in ms")
@click.option("--capture-timeout", default=2000, help="Capture timeout in ms")
@click.pass_context
def ndi_probe(ctx, source, extra_ips, discover_timeout, capture_timeout):
    """Connect to an NDI source and attempt to capture a single video frame.

    This is useful for debugging "source shows up in discovery but no frames arrive".
    """
    try:
        from scope.server.ndi.receiver import NDIReceiver
    except ImportError:
        output({"error": "NDI module not available"}, ctx)
        return

    receiver = NDIReceiver(recv_name="ScopeNDIProbe")
    if not receiver.create():
        output({"error": "NDI receiver create() failed"}, ctx)
        return

    extra_ips_list = list(extra_ips) if extra_ips else None
    try:
        src = receiver.connect_discovered(
            source_substring=source,
            extra_ips=extra_ips_list,
            timeout_ms=int(discover_timeout),
        )

        start = time.monotonic()
        deadline = start + max(0.0, float(capture_timeout)) / 1000.0
        frame = None
        while time.monotonic() < deadline:
            remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
            frame = receiver.receive_latest_rgb24(timeout_ms=min(250, remaining_ms))
            if frame is not None:
                break

        elapsed_ms = int((time.monotonic() - start) * 1000)
        result: dict[str, Any] = {
            "source": {"name": src.name, "url": src.url_address},
            "connections": receiver.get_no_connections(),
            "elapsed_ms": elapsed_ms,
            "frame": None,
        }
        if frame is not None:
            result["frame"] = {"shape": list(frame.shape), "dtype": str(frame.dtype)}

        output(result, ctx)
    except Exception as e:
        output({"error": str(e)}, ctx)
    finally:
        try:
            receiver.release()
        except Exception:
            pass


@input.command("status")
@click.pass_context
def input_status(ctx):
    """Get current input source status."""
    with get_client(ctx) as client:
        # Try to get state (requires WebRTC session)
        r = client.get("/api/v1/realtime/state")
        if r.status_code == 200:
            data = r.json()
            result = {
                "active_source": data.get("active_input_source", "unknown"),
                "ndi_frames_received": data.get("ndi_frames_received", 0),
                "ndi_frames_dropped": data.get("ndi_frames_dropped", 0),
            }
            output(result, ctx)
        else:
            # Fallback: just report that we can't get status without session
            output({"status": "no_session", "message": "Connect via WebRTC to see input status"}, ctx)


def main():
    cli()


if __name__ == "__main__":
    main()
