"""MCP (Model Context Protocol) server for Daydream Scope.

Exposes Scope's API as MCP tools, allowing AI assistants like Claude to
interact with a running Scope instance programmatically.

Usage:
    daydream-scope --mcp [--port PORT]

The MCP server communicates with a running Scope HTTP server via localhost.
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


def create_mcp_server(base_url: str = "http://localhost:8000") -> FastMCP:
    """Create and configure the MCP server with all Scope tools."""
    from contextlib import asynccontextmanager

    client: httpx.AsyncClient | None = None

    @asynccontextmanager
    async def _lifespan(_server: FastMCP):
        nonlocal client
        client = httpx.AsyncClient(base_url=base_url, timeout=300.0)
        try:
            yield
        finally:
            await client.aclose()
            client = None

    mcp = FastMCP(
        "daydream-scope",
        instructions=(
            "You are connected to a running Daydream Scope instance, a tool for "
            "real-time interactive generative AI video pipelines. Use the available "
            "tools to manage pipelines, assets, LoRAs, plugins, and monitor the system.\n\n"
            "Typical workflows:\n"
            "- Setup: get_pipeline_status -> load_pipeline -> start_stream (headless) -> update_parameters\n"
            "- Observe: capture_frame (see output), get_parameters (read state), get_session_metrics (fps/VRAM)\n"
            "- Record: get_session_metrics (get session_id) -> start_recording -> stop_recording\n"
            "- Cleanup: stop_stream -> unload_pipeline (frees VRAM before loading a different pipeline)\n\n"
            "Key constraints:\n"
            "- Use start_stream to begin a headless session, or wait for the user to click Start in the UI for WebRTC.\n"
            "- capture_frame returns a file_path to a JPEG you can read to see the pipeline's visual output.\n"
            "- get_logs with log_level='ERROR' is useful for diagnosing failures."
        ),
        lifespan=_lifespan,
    )

    # -------------------------------------------------------------------------
    # Pipeline Management
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def list_pipelines() -> str:
        """List all available pipelines with their schemas, supported modes,
        configuration options, and parameter definitions."""
        resp = await client.get("/api/v1/pipelines/schemas")
        return await _json(resp)

    @mcp.tool()
    async def get_pipeline_status() -> str:
        """Get the current pipeline status: whether a pipeline is loaded, loading,
        or not loaded, along with load parameters and any loaded LoRA adapters."""
        resp = await client.get("/api/v1/pipeline/status")
        return await _json(resp)

    @mcp.tool()
    async def load_pipeline(
        pipeline_id: str,
        height: int | None = None,
        width: int | None = None,
        base_seed: int | None = None,
        quantization: str | None = None,
        vace_enabled: bool | None = None,
        vae_type: str | None = None,
    ) -> str:
        """Load a pipeline for video generation.

        Args:
            pipeline_id: Pipeline ID (e.g. "streamdiffusionv2", "longlive", "krea-realtime-video")
            height: Output video height in pixels
            width: Output video width in pixels
            base_seed: Random seed for reproducible generation
            quantization: Quantization method ("fp8_e4m3fn", "fp8_e5m2", or null for none)
            vace_enabled: Enable VACE for reference image conditioning
            vae_type: VAE type ("wan", "lightvae", "tae", "lighttae")
        """
        load_params = {
            k: v
            for k, v in {
                "height": height,
                "width": width,
                "base_seed": base_seed,
                "quantization": quantization,
                "vace_enabled": vace_enabled,
                "vae_type": vae_type,
            }.items()
            if v is not None
        }
        body: dict = {"pipeline_ids": [pipeline_id]}
        if load_params:
            body["load_params"] = load_params

        resp = await client.post("/api/v1/pipeline/load", json=body)
        return await _json(resp)

    @mcp.tool()
    async def get_models_status(pipeline_id: str) -> str:
        """Check whether models for a pipeline are downloaded and get download progress.

        Args:
            pipeline_id: Pipeline ID to check model status for
        """
        resp = await client.get(
            "/api/v1/models/status", params={"pipeline_id": pipeline_id}
        )
        return await _json(resp)

    @mcp.tool()
    async def download_models(pipeline_id: str) -> str:
        """Start downloading the required models for a pipeline.

        Args:
            pipeline_id: Pipeline ID whose models to download
        """
        resp = await client.post(
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
        - prompts: list of {"text": str, "weight": float (0-100)}, e.g. [{"text": "a forest", "weight": 100}]
        - prompt_interpolation_method: "linear" or "slerp"
        - noise_scale: float 0.0-1.0 (video mode only)
        - noise_controller: bool, automatic noise adjustment based on motion
        - denoising_step_list: list of ints, e.g. [1000, 750, 500, 250]
        - manage_cache: bool, automatic cache management
        - reset_cache: bool, trigger a one-shot cache reset
        - kv_cache_attention_bias: float 0.01-1.0 (lower = less repetition)
        - lora_scales: list of {"path": str, "scale": float}
        - input_mode: "text" or "video"
        - input_source: {"enabled": true, "source_type": "<type>", "source_name": "<name>"}
        - paused: bool
        - vace_ref_images: list of file paths
        - vace_use_input_video: bool
        - vace_context_scale: float 0.0-2.0
        - first_frame_image: str file path
        - last_frame_image: str file path
        - images: list of file paths
        - Plus any pipeline-specific parameters (passed through to the pipeline)

        Args:
            parameters: Dict of parameter names to values
        """
        resp = await client.post("/api/v1/session/parameters", json=parameters)
        return await _json(resp)

    # -------------------------------------------------------------------------
    # Session Observation
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def get_parameters() -> str:
        """Get the current runtime parameters from all active sessions.
        Returns the merged parameter state (prompts, noise, denoising, etc.)."""
        resp = await client.get("/api/v1/session/parameters")
        return await _json(resp)

    @mcp.tool()
    async def capture_frame(quality: int = 85) -> str:
        """Capture the current pipeline output frame as a JPEG screenshot.
        Saves the image to a temp file and returns the file path so you can
        view it. Requires an active WebRTC stream.

        Args:
            quality: JPEG quality (1-100, default 85)
        """
        import base64
        import tempfile

        resp = await client.get("/api/v1/session/frame", params={"quality": quality})
        resp.raise_for_status()

        # Save to a temp file so Claude can read the image
        with tempfile.NamedTemporaryFile(
            suffix=".jpg", prefix="scope_frame_", delete=False
        ) as f:
            f.write(resp.content)
            file_path = f.name

        # Also return base64 for inline viewing
        b64 = base64.b64encode(resp.content).decode("ascii")
        return json.dumps(
            {
                "file_path": file_path,
                "base64_jpeg": b64,
                "size_bytes": len(resp.content),
            },
            indent=2,
        )

    @mcp.tool()
    async def get_session_metrics() -> str:
        """Get performance metrics from all active sessions.
        Returns per-session frame stats (fps_in, fps_out, pipeline_fps,
        frames_in, frames_out, elapsed_seconds) and GPU VRAM usage."""
        resp = await client.get("/api/v1/session/metrics")
        return await _json(resp)

    # -------------------------------------------------------------------------
    # Pipeline Lifecycle
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def unload_pipeline() -> str:
        """Unload all currently loaded pipelines and free GPU memory.
        Use this before loading a different pipeline or to reclaim VRAM."""
        resp = await client.post("/api/v1/pipeline/unload")
        return await _json(resp)

    @mcp.tool()
    async def start_stream(
        pipeline_id: str,
        input_mode: str = "text",
        prompts: list[dict] | None = None,
        input_source: dict | None = None,
    ) -> str:
        """Start a headless pipeline session (no browser needed).
        The pipeline must already be loaded via load_pipeline.
        Returns a session_id for use with capture_frame, update_parameters,
        start_recording, stop_recording, and stop_stream.

        Args:
            pipeline_id: Pipeline ID to run (must already be loaded)
            input_mode: "text" for prompt-only generation, "video" for input source processing
            prompts: Initial prompts, e.g. [{"text": "a forest", "weight": 100}]
            input_source: Server-side input source config for video mode. Format: {"enabled": true, "source_type": "<type>", "source_name": "<name>"}
        """
        body: dict = {"pipeline_id": pipeline_id, "input_mode": input_mode}
        if prompts is not None:
            body["prompts"] = prompts
        if input_source is not None:
            body["input_source"] = input_source
        resp = await client.post("/api/v1/session/start", json=body)
        return await _json(resp)

    @mcp.tool()
    async def stop_stream(session_id: str) -> str:
        """Stop a headless pipeline session and free its resources.

        Args:
            session_id: The session ID returned by start_stream
        """
        resp = await client.post(f"/api/v1/session/{session_id}/stop")
        return await _json(resp)

    # -------------------------------------------------------------------------
    # Recording Control
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def start_recording(session_id: str) -> str:
        """Start recording the output of a WebRTC session to an MP4 file.

        Args:
            session_id: The session ID to start recording for (get from get_session_metrics)
        """
        resp = await client.post(f"/api/v1/session/{session_id}/recording/start")
        return await _json(resp)

    @mcp.tool()
    async def stop_recording(session_id: str) -> str:
        """Stop recording the output of a WebRTC session.

        Args:
            session_id: The session ID to stop recording for
        """
        resp = await client.post(f"/api/v1/session/{session_id}/recording/stop")
        return await _json(resp)

    # -------------------------------------------------------------------------
    # Asset Management
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def list_assets() -> str:
        """List all available assets (images and videos) in the assets directory.
        Returns name, path, size, type, and creation time for each asset."""
        resp = await client.get("/api/v1/assets")
        return await _json(resp)

    # -------------------------------------------------------------------------
    # LoRA Management
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def list_loras() -> str:
        """List all installed LoRA adapter files with their metadata
        (name, path, size, SHA256, provenance)."""
        resp = await client.get("/api/v1/loras")
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
        resp = await client.post("/api/v1/loras", json=body)
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
        resp = await client.post("/api/v1/lora/download", json=body)
        return await _json(resp)

    @mcp.tool()
    async def delete_lora(name: str) -> str:
        """Delete a LoRA adapter file.

        Args:
            name: Filename of the LoRA to delete
        """
        resp = await client.delete(f"/api/v1/loras/{name}")
        return await _json(resp)

    # -------------------------------------------------------------------------
    # Plugin Management
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def list_plugins() -> str:
        """List all installed plugins with metadata, pipeline info,
        and available updates."""
        resp = await client.get("/api/v1/plugins")
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
        resp = await client.post("/api/v1/plugins", json=body)
        return await _json(resp)

    @mcp.tool()
    async def uninstall_plugin(name: str) -> str:
        """Uninstall a Scope plugin.

        Args:
            name: Plugin package name to uninstall
        """
        resp = await client.delete(f"/api/v1/plugins/{name}")
        return await _json(resp)

    @mcp.tool()
    async def reload_plugin(name: str, force: bool = False) -> str:
        """Reload an editable plugin to pick up code changes without restarting.

        Args:
            name: Plugin package name to reload
            force: Force reload even if plugin pipelines are currently loaded
        """
        resp = await client.post(
            f"/api/v1/plugins/{name}/reload", json={"force": force}
        )
        return await _json(resp)

    # -------------------------------------------------------------------------
    # System / Monitoring
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def get_health() -> str:
        """Check server health, version, git commit, and uptime."""
        resp = await client.get("/health")
        return await _json(resp)

    @mcp.tool()
    async def get_hardware_info() -> str:
        """Get hardware information: GPU VRAM, Spout/NDI/Syphon availability."""
        resp = await client.get("/api/v1/hardware/info")
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
        resp = await client.get("/api/v1/logs/tail", params={"lines": lines})
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
        resp = await client.get("/api/v1/input-sources")
        return await _json(resp)

    @mcp.tool()
    async def list_input_sources(source_type: str) -> str:
        """List available sources for a given input type.

        Args:
            source_type: Input source type (e.g. "webcam", "screen", "ndi", "spout", "syphon")
        """
        resp = await client.get(f"/api/v1/input-sources/{source_type}/sources")
        return await _json(resp)

    # -------------------------------------------------------------------------
    # OSC (Open Sound Control)
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def get_osc_status() -> str:
        """Get OSC server status (running, host, port)."""
        resp = await client.get("/api/v1/osc/status")
        return await _json(resp)

    @mcp.tool()
    async def get_osc_paths() -> str:
        """List available OSC control paths for the currently loaded pipeline."""
        resp = await client.get("/api/v1/osc/paths")
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
        resp = await client.post("/api/v1/workflow/resolve", json=workflow)
        return await _json(resp)

    # -------------------------------------------------------------------------
    # API Keys
    # -------------------------------------------------------------------------

    @mcp.tool()
    async def list_api_keys() -> str:
        """List configured API key services and their status (set/unset)."""
        resp = await client.get("/api/v1/keys")
        return await _json(resp)

    @mcp.tool()
    async def set_api_key(service_id: str, value: str) -> str:
        """Set an API key for a service (e.g. HuggingFace token).

        Args:
            service_id: Service identifier (e.g. "hf_token", "civitai_token")
            value: The API key value
        """
        resp = await client.put(f"/api/v1/keys/{service_id}", json={"value": value})
        return await _json(resp)

    @mcp.tool()
    async def delete_api_key(service_id: str) -> str:
        """Delete a stored API key for a service.

        Args:
            service_id: Service identifier to delete the key for
        """
        resp = await client.delete(f"/api/v1/keys/{service_id}")
        return await _json(resp)

    # -------------------------------------------------------------------------
    # Logs as a Resource
    # -------------------------------------------------------------------------

    @mcp.resource("logs://current")
    async def current_log_file() -> str:
        """The full contents of the current server log file."""
        resp = await client.get("/api/v1/logs/current")
        resp.raise_for_status()
        return resp.text

    return mcp


def run_mcp_server(port: int = 8000):
    """Run the MCP server over stdio, connecting to a Scope instance on the given port."""
    base_url = f"http://localhost:{port}"

    # Redirect all logging to stderr so stdout stays clean for MCP stdio transport
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    logger.info(f"Starting Daydream Scope MCP server (Scope API at {base_url})")

    mcp_server = create_mcp_server(base_url)
    mcp_server.run(transport="stdio")
