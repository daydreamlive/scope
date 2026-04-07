"""MCP (Model Context Protocol) server for Daydream Scope.

Exposes Scope's API as MCP tools, allowing AI assistants like Claude to
interact with a running Scope instance programmatically.

Usage:
    daydream-scope --mcp [--port PORT]

The MCP server communicates with a running Scope HTTP server via localhost.
When started without --port, it waits for a connect_to_scope tool call.
"""

import json
import logging
import sys

import httpx
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


def _fmt(data: dict | list) -> str:
    """Format JSON response data as a readable string."""
    return json.dumps(data, indent=2)


async def _json(resp: "httpx.Response") -> str:
    """Raise on HTTP error, then return formatted JSON."""
    resp.raise_for_status()
    return _fmt(resp.json())


def create_mcp_server(base_url: str | None = None) -> FastMCP:
    """Create and configure the MCP server with all Scope tools.

    Args:
        base_url: If provided, auto-connect to this Scope instance on startup.
                  If None, the server starts disconnected and waits for a
                  connect_to_scope tool call.
    """
    from contextlib import asynccontextmanager

    client: httpx.AsyncClient | None = None

    @asynccontextmanager
    async def _lifespan(_server: FastMCP):
        nonlocal client
        if base_url is not None:
            client = httpx.AsyncClient(base_url=base_url, timeout=300.0)
        try:
            yield
        finally:
            if client is not None:
                await client.aclose()
                client = None

    def _client() -> httpx.AsyncClient:
        if client is None:
            raise ValueError(
                "Not connected to a Scope instance. "
                "Use the connect_to_scope tool first with the port your Scope server is running on."
            )
        return client

    mcp = FastMCP(
        "daydream-scope",
        instructions=(
            "You are connected to a running Daydream Scope instance, a tool for "
            "real-time interactive generative AI video pipelines. Use the available "
            "tools to manage pipelines, assets, LoRAs, plugins, and monitor the system.\n\n"
            "Typical workflows:\n"
            "- Setup: connect_to_scope(port) -> get_pipeline_status -> load_pipeline -> start_stream (headless) -> update_parameters\n"
            "- Observe: capture_frame (see output), get_stream_url (MPEG-TS output URL), get_parameters (read state), get_session_metrics (fps/VRAM)\n"
            "- Cleanup: stop_stream (frees session resources)\n\n"
            "Key constraints:\n"
            "- You must call connect_to_scope first. The user will tell you which port Scope is running on.\n"
            "- Models must be downloaded before a pipeline can load. Use get_models_status to check, download_models if needed.\n"
            "- A pipeline must be loaded before starting a stream.\n"
            "- Use start_stream to begin a headless session, or wait for the user to click Start in the UI for WebRTC.\n"
            "- capture_frame returns a file_path to a JPEG you can read to see the pipeline's visual output.\n"
            "- If something fails, check get_logs with log_level='ERROR' to diagnose."
        ),
        lifespan=_lifespan,
    )

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def connect_to_scope(port: int) -> str:
        """Connect to a running Scope instance on the given port.
        Call this before using any other tools. Can be called again to
        switch to a different Scope instance.

        Args:
            port: The port the Scope HTTP server is running on (e.g. 8000)
        """
        nonlocal client

        if client is not None:
            await client.aclose()
            client = None

        new_base_url = f"http://localhost:{port}"
        new_client = httpx.AsyncClient(base_url=new_base_url, timeout=300.0)

        try:
            resp = await new_client.get("/health")
            resp.raise_for_status()
            health = resp.json()
            client = new_client
            return json.dumps(
                {
                    "status": "connected",
                    "base_url": new_base_url,
                    "server_version": health.get("version", "unknown"),
                },
                indent=2,
            )
        except Exception as e:
            await new_client.aclose()
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Could not connect to Scope at {new_base_url}: {e}",
                },
                indent=2,
            )

    # -------------------------------------------------------------------------
    # Cloud Connection
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def connect_to_cloud(
        app_id: str | None = None,
        api_key: str | None = None,
    ) -> str:
        """Connect to cloud for remote GPU inference.

        For local dev testing, start two Scope instances:
          Terminal 1 (cloud):  SCOPE_CLOUD_WS=1 uv run daydream-scope --port 8002
          Terminal 2 (local):  SCOPE_CLOUD_WS_URL=ws://localhost:8002/ws SCOPE_CLOUD_APP_ID=local uv run daydream-scope --port 8022

        Then connect MCP to the local instance (port 8022) and call this tool.
        When env vars SCOPE_CLOUD_APP_ID and SCOPE_CLOUD_WS_URL are set,
        app_id and api_key can be omitted.

        Args:
            app_id: Cloud app ID (optional if set via env var SCOPE_CLOUD_APP_ID)
            api_key: Cloud API key (optional if set via env var or for local dev testing)
        """
        body: dict = {}
        if app_id is not None:
            body["app_id"] = app_id
        if api_key is not None:
            body["api_key"] = api_key
        resp = await _client().post("/api/v1/cloud/connect", json=body)
        resp.raise_for_status()
        status = resp.json()

        # Poll until connected (background connection)
        if status.get("connecting"):
            import asyncio

            for _ in range(120):  # up to 120s
                await asyncio.sleep(1)
                poll = await _client().get("/api/v1/cloud/status")
                poll.raise_for_status()
                status = poll.json()
                if status.get("connected"):
                    return _fmt(status)
                if status.get("error"):
                    return _fmt(status)
                if not status.get("connecting"):
                    break

        return _fmt(status)

    @mcp.tool()
    async def disconnect_from_cloud() -> str:
        """Disconnect from cloud. Stops the WebSocket and WebRTC connections."""
        resp = await _client().post("/api/v1/cloud/disconnect", json={})
        return await _json(resp)

    @mcp.tool()
    async def get_cloud_status() -> str:
        """Get current cloud connection status including WebRTC and stats."""
        resp = await _client().get("/api/v1/cloud/status")
        return await _json(resp)

    # -------------------------------------------------------------------------
    # Pipeline Management
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def list_pipelines() -> str:
        """List all available pipelines with their schemas, supported modes,
        configuration options, and parameter definitions."""
        resp = await _client().get("/api/v1/pipelines/schemas")
        return await _json(resp)

    @mcp.tool()
    async def get_pipeline_status() -> str:
        """Get the current pipeline status: whether a pipeline is loaded, loading,
        or not loaded, along with load parameters and any loaded LoRA adapters."""
        resp = await _client().get("/api/v1/pipeline/status")
        return await _json(resp)

    @mcp.tool()
    async def load_pipeline(
        pipeline_id: str,
        load_params: dict | None = None,
    ) -> str:
        """Load a pipeline for video generation.

        Use list_pipelines to discover available pipelines and their accepted
        load_params. Common load_params include height, width, base_seed,
        quantization, vace_enabled, and vae_type, but each pipeline may
        define its own.

        Args:
            pipeline_id: Pipeline ID (e.g. "streamdiffusionv2", "longlive", "krea-realtime-video")
            load_params: Pipeline-specific load parameters as a dict (e.g. {"height": 512, "width": 512, "base_seed": 42})
        """
        body: dict = {"pipeline_ids": [pipeline_id]}
        if load_params:
            body["load_params"] = load_params

        resp = await _client().post("/api/v1/pipeline/load", json=body)
        return await _json(resp)

    @mcp.tool()
    async def get_models_status(pipeline_id: str) -> str:
        """Check whether models for a pipeline are downloaded and get download progress.

        Args:
            pipeline_id: Pipeline ID to check model status for
        """
        resp = await _client().get(
            "/api/v1/models/status", params={"pipeline_id": pipeline_id}
        )
        return await _json(resp)

    @mcp.tool()
    async def download_models(pipeline_id: str) -> str:
        """Start downloading the required models for a pipeline.

        Args:
            pipeline_id: Pipeline ID whose models to download
        """
        resp = await _client().post(
            "/api/v1/models/download", json={"pipeline_id": pipeline_id}
        )
        return await _json(resp)

    # -------------------------------------------------------------------------
    # Runtime Parameter Control
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def update_parameters(parameters: dict) -> str:
        """Update runtime parameters on the live stream. Changes are applied
        immediately and the frontend UI updates to reflect the new values.

        Requires an active stream (start one with start_stream or via the UI).

        The parameters dict accepts any combination of:

        Prompts:
        - prompts: list of {"text": str, "weight": float (0-100)}, e.g. [{"text": "a forest", "weight": 100}]
        - prompt_interpolation_method: "linear" or "slerp" (spatial blending of multiple prompts)
        - transition: {"target_prompts": [{"text": str, "weight": float}], "num_steps": int, "temporal_interpolation_method": "linear"|"slerp"}
          Smoothly interpolate from current prompts to target prompts over num_steps frames.

        Generation:
        - noise_scale: float 0.0-1.0 (video mode only, controls noise injection)
        - noise_controller: bool, automatic noise scale adjustment based on motion detection
        - denoising_step_list: list of ints, e.g. [1000, 750, 500, 250]

        Cache:
        - manage_cache: bool, automatic cache management on parameter changes
        - reset_cache: bool, trigger a one-shot cache reset (cleared after use)
        - kv_cache_attention_bias: float 0.01-1.0 (lower = less reliance on past frames, reduces repetition)

        LoRA:
        - lora_scales: list of {"path": str, "scale": float (-10.0 to 10.0)}, update loaded adapter strengths

        Input:
        - input_mode: "text" or "video"
        - input_source: {"enabled": bool, "source_type": "<type>", "source_name": "<name>"}
          Use list_input_source_types and list_input_sources to discover available types and names.

        Output:
        - output_sinks: dict of output configs keyed by type, e.g.
          {"spout": {"enabled": true, "name": "DaydreamScope"}}
          Available types: "spout" (Windows), "ndi" (all platforms), "syphon" (macOS).
          Use get_hardware_info to check which are available on this machine.

        VACE (reference image conditioning):
        - vace_ref_images: list of file paths (one-shot, cleared after use)
        - vace_use_input_video: bool, when true input video is used for VACE conditioning instead of latent init
        - vace_context_scale: float 0.0-2.0 (higher = reference images more influential)

        Frame references:
        - first_frame_image: str file path (one-shot, enables firstframe extension mode)
        - last_frame_image: str file path (one-shot, enables lastframe extension mode)
        - images: list of file paths (one-shot, non-VACE visual conditioning)

        Session:
        - paused: bool, pause/resume processing
        - recording: bool, enable/disable recording
        - pipeline_ids: list of str, pipeline chain to execute

        Pipeline-specific parameters are also accepted and passed through to the loaded pipeline.

        Args:
            parameters: Dict of parameter names to values
        """
        resp = await _client().post("/api/v1/session/parameters", json=parameters)
        return await _json(resp)

    # -------------------------------------------------------------------------
    # Session Observation
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def get_parameters() -> str:
        """Get the current runtime parameters from all active sessions.
        Returns the merged parameter state (prompts, noise, denoising, etc.)."""
        resp = await _client().get("/api/v1/session/parameters")
        return await _json(resp)

    @mcp.tool()
    async def capture_frame(quality: int = 85) -> str:
        """Capture the current pipeline output frame as a JPEG screenshot.
        Saves the image to a temp file and returns the file path so you can
        read it. Requires an active stream (WebRTC or headless).

        Args:
            quality: JPEG quality (1-100, default 85)
        """
        import tempfile

        resp = await _client().get("/api/v1/session/frame", params={"quality": quality})
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(
            suffix=".jpg", prefix="scope_frame_", delete=False
        ) as f:
            f.write(resp.content)
            file_path = f.name

        return json.dumps(
            {
                "file_path": file_path,
                "size_bytes": len(resp.content),
            },
            indent=2,
        )

    @mcp.tool()
    async def get_session_metrics() -> str:
        """Get performance metrics from all active sessions.
        Returns per-session frame stats (fps_in, fps_out, pipeline_fps,
        frames_in, frames_out, elapsed_seconds) and GPU VRAM usage."""
        resp = await _client().get("/api/v1/session/metrics")
        return await _json(resp)

    # -------------------------------------------------------------------------
    # Pipeline Lifecycle
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def start_stream(
        pipeline_id: str,
        input_mode: str = "text",
        prompts: list[dict] | None = None,
        input_source: dict | None = None,
    ) -> str:
        """Start a headless pipeline session (no browser needed).
        The pipeline must already be loaded via load_pipeline.
        Once started, use capture_frame, update_parameters,
        and stop_stream to control it.

        Args:
            pipeline_id: Pipeline ID to run (must already be loaded)
            input_mode: "text" for prompt-only generation, "video" for input source processing
            prompts: Initial prompts, e.g. [{"text": "a forest", "weight": 100}]
            input_source: Server-side input source config for video mode. Format: {"enabled": true, "source_type": "<type>", "source_name": "<name>"}. For video_file, source_name can be a full file path or an asset name.
        """
        body: dict = {"pipeline_id": pipeline_id, "input_mode": input_mode}
        if prompts is not None:
            body["prompts"] = prompts
        if input_source is not None:
            body["input_source"] = input_source
        resp = await _client().post("/api/v1/session/start", json=body)
        return await _json(resp)

    @mcp.tool()
    async def stop_stream() -> str:
        """Stop the active headless pipeline session and free its resources."""
        resp = await _client().post("/api/v1/session/stop")
        return await _json(resp)

    # -------------------------------------------------------------------------
    # Streaming Output
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def get_stream_url() -> str:
        """Get the MPEG-TS streaming URL for the active headless session.
        The URL streams H.264 video (and AAC audio when the pipeline
        produces audio) as video/mp2t. Requires an active headless
        stream started via start_stream.

        Returns the full URL that can be opened with ffplay, VLC, or
        any player that supports MPEG-TS over HTTP.
        """
        base_url = _client().base_url
        return json.dumps(
            {"stream_url": f"{base_url}/api/v1/session/output.ts"},
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Recording
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def start_recording() -> str:
        """Start recording the active headless session output to an MP4 file.
        Requires an active headless stream (started via start_stream).
        """
        resp = await _client().post("/api/v1/recordings/headless/start")
        return await _json(resp)

    @mcp.tool()
    async def stop_recording() -> str:
        """Stop recording the active headless session.
        Returns the path to the recording file.
        """
        resp = await _client().post("/api/v1/recordings/headless/stop")
        return await _json(resp)

    @mcp.tool()
    async def download_recording() -> str:
        """Download the recording from the active headless session as an MP4 file.
        Stops recording if still active, then saves the MP4 to a temp file
        and returns the file path so you can read/verify it.
        """
        import tempfile

        resp = await _client().get("/api/v1/recordings/headless")
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(
            suffix=".mp4", prefix="scope_recording_", delete=False
        ) as f:
            f.write(resp.content)
            file_path = f.name

        return json.dumps(
            {
                "file_path": file_path,
                "size_bytes": len(resp.content),
            },
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Asset Management
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def list_assets() -> str:
        """List all available assets (images and videos) in the assets directory.
        Returns name, path, size, type, and creation time for each asset."""
        resp = await _client().get("/api/v1/assets")
        return await _json(resp)

    # -------------------------------------------------------------------------
    # LoRA Management
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def list_loras() -> str:
        """List all installed LoRA adapter files with their metadata
        (name, path, size, SHA256, provenance)."""
        resp = await _client().get("/api/v1/loras")
        return await _json(resp)

    @mcp.tool()
    async def install_lora(url: str, filename: str | None = None) -> str:
        """Install a LoRA adapter from a URL (HuggingFace or CivitAI).

        Args:
            url: URL to download the LoRA from
            filename: Optional filename to save as (auto-detected if not provided)
        """
        body: dict = {"url": url}
        if filename:
            body["filename"] = filename
        resp = await _client().post("/api/v1/loras", json=body)
        return await _json(resp)

    @mcp.tool()
    async def download_lora(
        source: str,
        repo_id: str | None = None,
        hf_filename: str | None = None,
        model_id: str | None = None,
        version_id: str | None = None,
        url: str | None = None,
        subfolder: str | None = None,
    ) -> str:
        """Download a LoRA adapter from HuggingFace, CivitAI, or a direct URL.

        Args:
            source: Download source ("huggingface", "civitai", or "url")
            repo_id: HuggingFace repo ID (for source="huggingface")
            hf_filename: Filename within the HF repo (for source="huggingface")
            model_id: CivitAI model ID (for source="civitai")
            version_id: CivitAI version ID (for source="civitai")
            url: Direct download URL (for source="url")
            subfolder: Subfolder within the LoRA directory to save to
        """
        body = {
            k: v
            for k, v in {
                "source": source,
                "repo_id": repo_id,
                "hf_filename": hf_filename,
                "model_id": model_id,
                "version_id": version_id,
                "url": url,
                "subfolder": subfolder,
            }.items()
            if v is not None
        }
        resp = await _client().post("/api/v1/lora/download", json=body)
        return await _json(resp)

    @mcp.tool()
    async def delete_lora(name: str) -> str:
        """Delete a LoRA adapter file.

        Args:
            name: Filename of the LoRA to delete
        """
        resp = await _client().delete(f"/api/v1/loras/{name}")
        return await _json(resp)

    # -------------------------------------------------------------------------
    # Plugin Management
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def list_plugins() -> str:
        """List all installed plugins with metadata, pipeline info,
        and available updates."""
        resp = await _client().get("/api/v1/plugins")
        return await _json(resp)

    @mcp.tool()
    async def install_plugin(
        package: str,
        editable: bool = False,
        upgrade: bool = False,
        force: bool = False,
        pre: bool = False,
    ) -> str:
        """Install a Scope plugin.

        Args:
            package: Package specifier (PyPI name, git URL, or local path)
            editable: Install in editable/development mode
            upgrade: Upgrade if already installed
            force: Skip dependency validation
            pre: Include pre-release versions
        """
        body = {
            "package": package,
            "editable": editable,
            "upgrade": upgrade,
            "force": force,
            "pre": pre,
        }
        resp = await _client().post("/api/v1/plugins", json=body)
        return await _json(resp)

    @mcp.tool()
    async def uninstall_plugin(name: str) -> str:
        """Uninstall a Scope plugin.

        Args:
            name: Plugin package name to uninstall
        """
        resp = await _client().delete(f"/api/v1/plugins/{name}")
        return await _json(resp)

    @mcp.tool()
    async def reload_plugin(name: str, force: bool = False) -> str:
        """Reload an editable plugin to pick up code changes without restarting.

        Args:
            name: Plugin package name to reload
            force: Force reload even if plugin pipelines are currently loaded
        """
        resp = await _client().post(
            f"/api/v1/plugins/{name}/reload", json={"force": force}
        )
        return await _json(resp)

    # -------------------------------------------------------------------------
    # System / Monitoring
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def get_health() -> str:
        """Check server health, version, git commit, and uptime."""
        resp = await _client().get("/health")
        return await _json(resp)

    @mcp.tool()
    async def get_hardware_info() -> str:
        """Get hardware information: GPU VRAM, Spout/NDI/Syphon availability."""
        resp = await _client().get("/api/v1/hardware/info")
        return await _json(resp)

    @mcp.tool()
    async def get_logs(lines: int = 200, log_level: str | None = None) -> str:
        """Get recent Scope server log lines for debugging and monitoring.
        Returns logs from the main Scope process (not the MCP process).
        Useful for diagnosing pipeline errors, checking model loading, and
        monitoring WebRTC session activity. Supports up to 1000 lines.

        Args:
            lines: Number of recent log lines to return (1-1000, default 200)
            log_level: Optional minimum log level filter ("DEBUG", "INFO", "WARNING", "ERROR"). When set, only lines containing this level or higher are returned.
        """
        resp = await _client().get("/api/v1/logs/tail", params={"lines": lines})
        resp.raise_for_status()
        data = resp.json()
        log_lines = data.get("lines", [])

        if log_level:
            level_order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            level_upper = log_level.upper()
            if level_upper in level_order:
                min_idx = level_order.index(level_upper)
                allowed = set(level_order[min_idx:])
                log_lines = [
                    line
                    for line in log_lines
                    if any(f" {lvl} " in line or f" {lvl}:" in line for lvl in allowed)
                ]

        return "\n".join(log_lines)

    # -------------------------------------------------------------------------
    # Input Sources
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def list_input_source_types() -> str:
        """List available input source types (webcam, screen capture, NDI, Spout, Syphon, etc.)."""
        resp = await _client().get("/api/v1/input-sources")
        return await _json(resp)

    @mcp.tool()
    async def list_input_sources(source_type: str) -> str:
        """List available sources for a given input type.

        Args:
            source_type: Input source type (e.g. "webcam", "screen", "ndi", "spout", "syphon")
        """
        resp = await _client().get(f"/api/v1/input-sources/{source_type}/sources")
        return await _json(resp)

    # -------------------------------------------------------------------------
    # OSC (Open Sound Control)
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def get_osc_status() -> str:
        """Get OSC server status (running, host, port)."""
        resp = await _client().get("/api/v1/osc/status")
        return await _json(resp)

    @mcp.tool()
    async def get_osc_paths() -> str:
        """List available OSC control paths for the currently loaded pipeline."""
        resp = await _client().get("/api/v1/osc/paths")
        return await _json(resp)

    # -------------------------------------------------------------------------
    # Workflow
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def resolve_workflow(workflow_json: str) -> str:
        """Resolve dependencies for a workflow import (checks pipelines, LoRAs, plugins).

        Args:
            workflow_json: The workflow JSON string to resolve dependencies for
        """
        try:
            workflow = json.loads(workflow_json)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON: {e}"})
        resp = await _client().post("/api/v1/workflow/resolve", json=workflow)
        return await _json(resp)

    # -------------------------------------------------------------------------
    # API Keys
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def list_api_keys() -> str:
        """List configured API key services and their status (set/unset)."""
        resp = await _client().get("/api/v1/keys")
        return await _json(resp)

    @mcp.tool()
    async def set_api_key(service_id: str, value: str) -> str:
        """Set an API key for a service (e.g. HuggingFace token).

        Args:
            service_id: Service identifier (e.g. "hf_token", "civitai_token")
            value: The API key value
        """
        resp = await _client().put(f"/api/v1/keys/{service_id}", json={"value": value})
        return await _json(resp)

    @mcp.tool()
    async def delete_api_key(service_id: str) -> str:
        """Delete a stored API key for a service.

        Args:
            service_id: Service identifier to delete the key for
        """
        resp = await _client().delete(f"/api/v1/keys/{service_id}")
        return await _json(resp)

    # -------------------------------------------------------------------------
    # Logs as a Resource
    # -------------------------------------------------------------------------

    @mcp.resource("logs://current")
    async def current_log_file() -> str:
        """The full contents of the current server log file."""
        resp = await _client().get("/api/v1/logs/current")
        resp.raise_for_status()
        return resp.text

    return mcp


def run_mcp_server(port: int | None = None):
    """Run the MCP server over stdio.

    Args:
        port: If provided, auto-connect to a Scope instance on this port.
              If None, the server starts disconnected and waits for a
              connect_to_scope tool call.
    """
    base_url = f"http://localhost:{port}" if port is not None else None

    # Redirect all logging to stderr so stdout stays clean for MCP stdio transport
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    if base_url:
        logger.info(f"Starting Daydream Scope MCP server (Scope API at {base_url})")
    else:
        logger.info(
            "Starting Daydream Scope MCP server (disconnected, waiting for connect_to_scope)"
        )

    mcp_server = create_mcp_server(base_url)
    mcp_server.run(transport="stdio")
