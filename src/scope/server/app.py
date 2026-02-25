import asyncio
import contextlib
import io
import logging
import os
import subprocess
import sys
import threading
import time
import warnings
import webbrowser
from contextlib import asynccontextmanager
from datetime import datetime
from functools import wraps
from importlib.metadata import version
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

import click
import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

if TYPE_CHECKING:
    from .cloud_connection import CloudConnectionManager
    from .pipeline_manager import PipelineManager
    from .schema import PluginInfo
    from .webrtc import WebRTCManager

from .cloud_proxy import (
    cloud_proxy,
    get_hardware_info_from_cloud,
    recording_download_cloud_path,
    upload_asset_to_cloud,
)
from .download_models import download_models
from .download_progress_manager import download_progress_manager
from .file_utils import (
    IMAGE_EXTENSIONS,
    LORA_EXTENSIONS,
    VIDEO_EXTENSIONS,
    iter_files,
)
from .kafka_publisher import (
    KafkaPublisher,
    is_kafka_enabled,
    set_kafka_publisher,
)
from .logs_config import (
    cleanup_old_logs,
    ensure_logs_dir,
    get_current_log_file,
    get_logs_dir,
    get_most_recent_log_file,
)
from .models_config import (
    ensure_models_dir,
    get_assets_dir,
    get_lora_dir,
    models_are_downloaded,
)
from .pipeline_manager import PipelineManager
from .recording import (
    cleanup_recording_files,
    cleanup_temp_file,
)
from .schema import (
    ApiKeyDeleteResponse,
    ApiKeyInfo,
    ApiKeySetRequest,
    ApiKeySetResponse,
    ApiKeysListResponse,
    AssetFileInfo,
    AssetsResponse,
    CloudConnectRequest,
    CloudStatusResponse,
    HardwareInfoResponse,
    HealthResponse,
    IceCandidateRequest,
    IceServerConfig,
    IceServersResponse,
    PipelineLoadRequest,
    PipelineSchemasResponse,
    PipelineStatusResponse,
    WebRTCOfferRequest,
    WebRTCOfferResponse,
)


class STUNErrorFilter(logging.Filter):
    """Filter to suppress STUN/TURN connection errors that are not critical."""

    def filter(self, record):
        # Suppress STUN  exeception that occurrs always during the stream restart
        if "Task exception was never retrieved" in record.getMessage():
            return False
        return True


# Ensure logs directory exists and clean up old logs
logs_dir = ensure_logs_dir()
cleanup_old_logs(max_age_days=1)  # Delete logs older than 1 day
log_file = get_current_log_file()

# Configure logging - set root to WARNING to keep non-app libraries quiet by default
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Console handler handles INFO
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    if isinstance(handler, logging.StreamHandler) and not isinstance(
        handler, RotatingFileHandler
    ):
        handler.setLevel(logging.INFO)

# Add rotating file handler
file_handler = RotatingFileHandler(
    log_file,
    maxBytes=5 * 1024 * 1024,  # 5 MB per file
    backupCount=5,  # Keep 5 backup files
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
root_logger.addHandler(file_handler)

# Add the filter to suppress STUN/TURN errors
stun_filter = STUNErrorFilter()
logging.getLogger("asyncio").addFilter(stun_filter)

# Set INFO level for your app modules
logging.getLogger("scope.server").setLevel(logging.INFO)
logging.getLogger("scope.core").setLevel(logging.INFO)

# Set INFO level for uvicorn
logging.getLogger("uvicorn.error").setLevel(logging.INFO)

# Enable verbose logging for other libraries when needed
if os.getenv("VERBOSE_LOGGING"):
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("aiortc").setLevel(logging.INFO)

# Select pipeline depending on the "PIPELINE" environment variable
PIPELINE = os.getenv("PIPELINE", None)

logger = logging.getLogger(__name__)


def suppress_init_output(func):
    """Decorator to suppress all initialization output (logging, warnings, stdout/stderr)."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore")
            # Temporarily disable all logging
            logging.disable(logging.CRITICAL)
            try:
                return func(*args, **kwargs)
            finally:
                # Re-enable logging
                logging.disable(logging.NOTSET)

    return wrapper


def get_git_commit_hash() -> str:
    """
    Get the current git commit hash.

    Returns:
        Git commit hash if available, otherwise a fallback message.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,  # 5 second timeout
            cwd=Path(__file__).parent,  # Run in the project directory
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return "unknown (not a git repository)"
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return "unknown (git error)"
    except FileNotFoundError:
        return "unknown (git not installed)"
    except Exception:
        return "unknown"


def print_version_info():
    """Print version information and exit."""
    try:
        pkg_version = version("daydream-scope")
    except Exception:
        pkg_version = "unknown"

    git_hash = get_git_commit_hash()

    print(f"daydream-scope: {pkg_version}")
    print(f"git commit: {git_hash}")


def configure_static_files():
    """Configure static file serving for production."""
    frontend_dist = Path(__file__).parent.parent.parent.parent / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount(
            "/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets"
        )
        logger.info(f"Serving static assets from {frontend_dist / 'assets'}")
    else:
        logger.info("Frontend dist directory not found - running in development mode")


# Global WebRTC manager instance
webrtc_manager = None
# Global pipeline manager instance
pipeline_manager = None
# Server startup timestamp for detecting restarts
server_start_time = time.time()
# Global cloud connection manager instance
cloud_connection_manager = None
# Global Kafka publisher instance (optional, initialized if credentials are present)
kafka_publisher = None


async def prewarm_pipeline(pipeline_id: str):
    """Background task to pre-warm the pipeline without blocking startup."""
    try:
        await asyncio.wait_for(
            pipeline_manager.load_pipelines([pipeline_id]),
            timeout=300,  # 5 minute timeout for pipeline loading
        )
    except Exception as e:
        logger.error(f"Error pre-warming pipeline {pipeline_id} in background: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for startup and shutdown events."""
    # Lazy imports to avoid loading torch at CLI startup (fixes Windows DLL locking)
    import torch

    from .cloud_connection import CloudConnectionManager
    from .pipeline_manager import PipelineManager
    from .webrtc import WebRTCManager

    # Startup
    global webrtc_manager, pipeline_manager, cloud_connection_manager, kafka_publisher

    # Check CUDA availability and warn if not available
    if not torch.cuda.is_available():
        warning_msg = (
            "CUDA is not available on this system. "
            "Some pipelines may not work without a CUDA-compatible GPU. "
            "The application will start, but pipeline functionality may be limited."
        )
        logger.warning(warning_msg)

    # Clean up recording files from previous sessions (in case of crashes)
    cleanup_recording_files()

    # Log logs directory
    logs_dir = get_logs_dir()
    logger.info(f"Logs directory: {logs_dir}")

    # Ensure models directory and subdirectories exist
    models_dir = ensure_models_dir()
    logger.info(f"Models directory: {models_dir}")

    # Ensure assets directory exists for VACE reference images and other media (at same level as models)
    assets_dir = get_assets_dir()
    assets_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Assets directory: {assets_dir}")

    # Initialize pipeline manager (but don't load pipeline yet)
    pipeline_manager = PipelineManager()
    logger.info("Pipeline manager initialized")

    # Pre-warm the default pipeline
    if PIPELINE is not None:
        asyncio.create_task(prewarm_pipeline(PIPELINE))

    webrtc_manager = WebRTCManager()
    logger.info("WebRTC manager initialized")

    cloud_connection_manager = CloudConnectionManager()
    logger.info("Cloud connection manager initialized")

    # Initialize Kafka publisher if credentials are configured
    if is_kafka_enabled():
        kafka_publisher = KafkaPublisher()
        if await kafka_publisher.start():
            set_kafka_publisher(kafka_publisher)
            logger.info("Kafka publisher initialized")
        else:
            kafka_publisher = None
            logger.warning("Kafka publisher failed to start")

    yield

    # Shutdown
    if webrtc_manager:
        logger.info("Shutting down WebRTC manager...")
        await webrtc_manager.stop()
        logger.info("WebRTC manager shutdown complete")

    if pipeline_manager:
        logger.info("Shutting down pipeline manager...")
        pipeline_manager.unload_all_pipelines()
        logger.info("Pipeline manager shutdown complete")

    if cloud_connection_manager and cloud_connection_manager.is_connected:
        logger.info("Shutting down cloud connection...")
        await cloud_connection_manager.disconnect()
        logger.info("Cloud connection shutdown complete")

    if kafka_publisher:
        logger.info("Shutting down Kafka publisher...")
        await kafka_publisher.stop()
        set_kafka_publisher(None)
        logger.info("Kafka publisher shutdown complete")


def get_webrtc_manager() -> "WebRTCManager":
    """Dependency to get WebRTC manager instance."""
    return webrtc_manager


def get_pipeline_manager() -> "PipelineManager":
    """Dependency to get pipeline manager instance."""
    return pipeline_manager


def get_cloud_connection_manager() -> "CloudConnectionManager":
    """Dependency to get cloud connection manager instance."""
    return cloud_connection_manager


app = FastAPI(
    lifespan=lifespan,
    title="Scope",
    description="A tool for running and customizing real-time, interactive generative AI pipelines and models",
    version=version("daydream-scope"),
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        server_start_time=server_start_time,
        version=version("daydream-scope"),
        git_commit=get_git_commit_hash(),
    )


@app.post("/api/v1/restart")
async def restart_server():
    """Restart the server process.

    This endpoint is called after plugin install/uninstall to ensure
    Python's module cache is refreshed. The server restarts by re-executing
    the entry point, which replaces the current process and keeps terminal
    output working.

    When running under the Electron app (DAYDREAM_SCOPE_MANAGED=1), the server
    exits with code 42 to signal the managing process to respawn it. This
    ensures proper PID tracking and prevents orphaned processes on Windows.

    Known limitation (Windows/Git Bash): After the server restarts and you
    press Ctrl+C, the terminal may appear to hang. The process exits correctly,
    but MinTTY's input buffer gets stuck. Press any key to get your prompt back.
    This is a quirk of how MinTTY handles process replacement and doesn't affect
    CMD, PowerShell, or the Electron app.
    """
    # If managed by an external process (e.g., Electron), exit and let it respawn us
    # This prevents orphaned processes on Windows where os.execv spawns a new PID
    if os.environ.get("DAYDREAM_SCOPE_MANAGED"):
        logger.info("Server restart requested (managed mode) - exiting for respawn...")

        def do_managed_exit():
            time.sleep(0.5)  # Give time for response to be sent
            logger.info("Exiting with code 42 for managed respawn...")
            os._exit(42)  # Use os._exit to terminate entire process from thread

        thread = threading.Thread(target=do_managed_exit, daemon=True)
        thread.start()
        return {"message": "Server exiting for respawn..."}

    # Standalone mode: self-restart via subprocess/os.execv
    def do_restart():
        time.sleep(0.5)  # Give time for response to be sent

        # Close all logging handlers to avoid file descriptor warnings
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)

        # On Windows, entry points are .exe files but sys.argv[0] may not have extension
        executable = sys.argv[0]
        if sys.platform == "win32" and not executable.endswith(".exe"):
            executable += ".exe"

        if sys.platform == "win32":
            # On Windows, we can't use os.execv() because it spawns a child process
            # instead of replacing in-place (unlike Unix). So we spawn with Popen
            # and exit. Known issue: In Git Bash/MinTTY, after Ctrl+C the terminal
            # may require an extra keypress due to MinTTY's input buffer handling.
            subprocess.Popen(
                [executable] + sys.argv[1:],
                stdin=subprocess.DEVNULL,
                stdout=None,  # Inherit parent's stdout
                stderr=None,  # Inherit parent's stderr
            )
            sys.stdout.flush()
            sys.stderr.flush()
            try:
                sys.stdin.close()
            except Exception:
                pass
            os._exit(0)
        else:
            # On Unix, execv works correctly (replaces process in-place)
            os.execv(executable, sys.argv)

    # Run in a thread to allow response to be sent first
    thread = threading.Thread(target=do_restart, daemon=True)
    thread.start()
    return {"message": "Server restarting..."}


@app.get("/")
async def root():
    """Serve the frontend at the root URL."""
    frontend_dist = Path(__file__).parent.parent.parent.parent / "frontend" / "dist"

    # Only serve SPA if frontend dist exists (production mode)
    if not frontend_dist.exists():
        return {"message": "Scope API - Frontend not built"}

    # Serve the frontend index.html with no-cache headers
    # This ensures clients like Electron alway fetch the latest HTML (which references hashed assets)
    index_file = frontend_dist / "index.html"
    if index_file.exists():
        return FileResponse(
            index_file,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    return {"message": "Scope API - Frontend index.html not found"}


@app.post("/api/v1/pipeline/load")
@cloud_proxy(timeout=60.0)
async def load_pipeline(
    request: PipelineLoadRequest,
    http_request: Request,
    pipeline_manager: "PipelineManager" = Depends(get_pipeline_manager),
    cloud_manager: "CloudConnectionManager" = Depends(get_cloud_connection_manager),
):
    """Load one or more pipelines.

    In cloud mode (when connected to cloud), this proxies the request to the
    cloud-hosted scope backend.
    """
    try:
        # Get pipeline IDs to load
        pipeline_ids = request.pipeline_ids
        if not pipeline_ids:
            raise HTTPException(
                status_code=400,
                detail="pipeline_ids must be provided and cannot be empty",
            )

        # load_params is already a dict (or None)
        load_params_dict = request.load_params

        # Local mode: start loading in background without blocking
        asyncio.create_task(
            pipeline_manager.load_pipelines(
                pipeline_ids,
                load_params_dict,
                connection_id=request.connection_id,
                connection_info=request.connection_info,
                user_id=request.user_id,
            )
        )
        return {"message": "Pipeline loading initiated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/pipeline/status", response_model=PipelineStatusResponse)
@cloud_proxy()
async def get_pipeline_status(
    http_request: Request,
    pipeline_manager: "PipelineManager" = Depends(get_pipeline_manager),
    cloud_manager: "CloudConnectionManager" = Depends(get_cloud_connection_manager),
):
    """Get current pipeline status.

    In cloud mode (when connected to cloud), this proxies the request to the
    cloud-hosted scope backend.
    """
    try:
        status_info = await pipeline_manager.get_status_info_async()
        return PipelineStatusResponse(**status_info)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/pipelines/schemas", response_model=PipelineSchemasResponse)
@cloud_proxy()
async def get_pipeline_schemas(
    http_request: Request,
    cloud_manager: "CloudConnectionManager" = Depends(get_cloud_connection_manager),
):
    """Get configuration schemas and defaults for all available pipelines.

    Returns the output of each pipeline's get_schema_with_metadata() method,
    which includes:
    - Pipeline metadata (id, name, description, version)
    - supported_modes: List of supported input modes ("text", "video")
    - default_mode: Default input mode for this pipeline
    - mode_defaults: Mode-specific default overrides (if any)
    - config_schema: Full JSON schema with defaults

    The frontend should use this as the source of truth for parameter defaults.

    In cloud mode (when connected to cloud), this proxies the request to the
    cloud-hosted scope backend to get the available pipelines there.
    """
    from scope.core.pipelines.registry import PipelineRegistry
    from scope.core.plugins import get_plugin_manager

    plugin_manager = get_plugin_manager()
    pipelines: dict = {}

    for pipeline_id in PipelineRegistry.list_pipelines():
        config_class = PipelineRegistry.get_config_class(pipeline_id)
        if config_class:
            # get_schema_with_metadata() includes supported_modes, default_mode,
            # and mode_defaults directly from the config class
            schema_data = config_class.get_schema_with_metadata()
            schema_data["plugin_name"] = plugin_manager.get_plugin_for_pipeline(
                pipeline_id
            )
            pipelines[pipeline_id] = schema_data

    return PipelineSchemasResponse(pipelines=pipelines)


@app.get("/api/v1/webrtc/ice-servers", response_model=IceServersResponse)
async def get_ice_servers(
    webrtc_manager: "WebRTCManager" = Depends(get_webrtc_manager),
    cloud_manager: "CloudConnectionManager" = Depends(get_cloud_connection_manager),
):
    """Return ICE server configuration for frontend WebRTC connection.

    In cloud mode, this returns the ICE servers from the cloud-hosted scope backend.
    """
    # If connected to cloud, get ICE servers from cloud
    if cloud_manager.is_connected:
        try:
            cloud_ice_servers = await cloud_manager.webrtc_get_ice_servers()
            return IceServersResponse(
                iceServers=[
                    IceServerConfig(**server)
                    for server in cloud_ice_servers.get("iceServers", [])
                ]
            )
        except Exception as e:
            logger.warning(f"Failed to get ICE servers from cloud, using local: {e}")

    # Local mode or fallback
    ice_servers = []
    for server in webrtc_manager.rtc_config.iceServers:
        ice_servers.append(
            IceServerConfig(
                urls=server.urls,
                username=server.username if hasattr(server, "username") else None,
                credential=server.credential if hasattr(server, "credential") else None,
            )
        )

    return IceServersResponse(iceServers=ice_servers)


@app.post("/api/v1/webrtc/offer", response_model=WebRTCOfferResponse)
async def handle_webrtc_offer(
    request: WebRTCOfferRequest,
    webrtc_manager: "WebRTCManager" = Depends(get_webrtc_manager),
    pipeline_manager: "PipelineManager" = Depends(get_pipeline_manager),
    cloud_manager: "CloudConnectionManager" = Depends(get_cloud_connection_manager),
):
    """Handle WebRTC offer and return answer.

    In cloud mode, video flows through the backend to cloud:
        Browser → Backend (WebRTC) → cloud (WebRTC) → Backend → Browser

    This enables:
    - Spout input to be forwarded to cloud
    - Full control over the video pipeline on the backend
    - Local backend can record/manipulate frames
    """
    try:
        # If connected to cloud, use cloud mode (video flows through backend)
        if cloud_manager.is_connected:
            logger.info(
                "[CLOUD] Using relay mode - video will flow through backend to cloud"
            )
            return await webrtc_manager.handle_offer_with_relay(request, cloud_manager)

        # Local mode: ensure pipeline is loaded before proceeding
        status_info = await pipeline_manager.get_status_info_async()
        if status_info["status"] != "loaded":
            raise HTTPException(
                status_code=400,
                detail="Pipeline not loaded. Please load pipeline first.",
            )

        return await webrtc_manager.handle_offer(request, pipeline_manager)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling WebRTC offer: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.patch(
    "/api/v1/webrtc/offer/{session_id}", status_code=204, response_class=Response
)
async def add_ice_candidate(
    session_id: str,
    candidate_request: IceCandidateRequest,
    webrtc_manager: "WebRTCManager" = Depends(get_webrtc_manager),
):
    """Add ICE candidate(s) to an existing WebRTC session (Trickle ICE).

    This endpoint follows the Trickle ICE pattern, allowing clients to send
    ICE candidates as they are discovered.

    Note: In cloud mode, the browser still connects to the LOCAL backend via WebRTC.
    The backend then relays frames to/from cloud via a separate WebRTC connection.
    So browser ICE candidates always go to the local WebRTC session.
    """
    # TODO: Validate that the Content-Type is 'application/trickle-ice-sdpfrag'
    # At the moment FastAPI defaults to validating that it is 'application/json'
    try:
        # Always add ICE candidates to the local session
        # (In cloud mode, browser connects to local backend, not directly to cloud)
        for candidate_init in candidate_request.candidates:
            await webrtc_manager.add_ice_candidate(
                session_id=session_id,
                candidate=candidate_init.candidate,
                sdp_mid=candidate_init.sdpMid,
                sdp_mline_index=candidate_init.sdpMLineIndex,
            )

        logger.debug(
            f"Added {len(candidate_request.candidates)} ICE candidates to session {session_id}"
        )

        # Return 204 No Content on success
        return Response(status_code=204)

    except ValueError as e:
        # Session not found or invalid candidate
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Error adding ICE candidate to session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/recordings/{session_id}")
@cloud_proxy(recording_download_cloud_path, timeout=120.0)
async def download_recording(
    http_request: Request,
    session_id: str,
    background_tasks: BackgroundTasks,
    webrtc_manager: "WebRTCManager" = Depends(get_webrtc_manager),
    cloud_manager: "CloudConnectionManager" = Depends(get_cloud_connection_manager),
):
    """Download the recording file for the specified session.
    This will finalize the current recording and create a copy for download,
    then continue recording with a new file.

    In cloud mode, this proxies the download request to cloud.
    """
    try:
        session = webrtc_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found",
            )

        # Check if session has a recording manager
        if not session.recording_manager:
            raise HTTPException(
                status_code=404,
                detail=f"Recording not available for session {session_id}",
            )

        # Finalize the recording and get the download file
        download_file = await session.recording_manager.finalize_and_get_recording()
        if not download_file or not Path(download_file).exists():
            raise HTTPException(
                status_code=404,
                detail="Recording file not available",
            )

        # Schedule cleanup of the temp file after download
        background_tasks.add_task(cleanup_temp_file, download_file)

        # Generate filename with datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording-{timestamp}.mp4"

        # Return the file for download
        return FileResponse(
            download_file,
            media_type="video/mp4",
            filename=filename,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading recording: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


class ModelStatusResponse(BaseModel):
    downloaded: bool


class DownloadModelsRequest(BaseModel):
    pipeline_id: str


class LoRAFileInfo(BaseModel):
    """Metadata for an available LoRA file on disk."""

    name: str
    path: str
    size_mb: float
    folder: str | None = None


class LoRAFilesResponse(BaseModel):
    """Response containing all discoverable LoRA files."""

    lora_files: list[LoRAFileInfo]


@app.get("/api/v1/loras", response_model=LoRAFilesResponse)
@cloud_proxy()
async def list_lora_files(
    http_request: Request,
    cloud_manager: "CloudConnectionManager" = Depends(get_cloud_connection_manager),
):
    """List available LoRA files in the models/lora directory and its subdirectories.

    When cloud mode is active, lists LoRA files from the cloud server instead.
    """

    def process_lora_file(file_path: Path, lora_dir: Path) -> LoRAFileInfo:
        """Extract LoRA file metadata."""
        size_mb = file_path.stat().st_size / (1024 * 1024)
        relative_path = file_path.relative_to(lora_dir)
        folder = (
            str(relative_path.parent) if relative_path.parent != Path(".") else None
        )
        return LoRAFileInfo(
            name=file_path.stem,
            path=str(file_path),
            size_mb=round(size_mb, 2),
            folder=folder,
        )

    try:
        lora_dir = get_lora_dir()
        lora_files: list[LoRAFileInfo] = []

        for file_path in iter_files(lora_dir, LORA_EXTENSIONS):
            lora_files.append(process_lora_file(file_path, lora_dir))

        lora_files.sort(key=lambda x: (x.folder or "", x.name))
        return LoRAFilesResponse(lora_files=lora_files)

    except Exception as e:  # pragma: no cover - defensive logging
        logger.error(f"list_lora_files: Error listing LoRA files: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


class LoRAInstallRequest(BaseModel):
    url: str
    filename: str | None = None


class LoRAInstallResponse(BaseModel):
    message: str
    file: LoRAFileInfo


ALLOWED_LORA_HOSTS = {"civitai.com", "huggingface.co"}


@app.post("/api/v1/loras", response_model=LoRAInstallResponse)
async def install_lora_file(
    request: LoRAInstallRequest,
    http_request: Request,
    cloud_manager: "CloudConnectionManager" = Depends(get_cloud_connection_manager),
):
    """Install a LoRA file from a URL (e.g. HuggingFace, CivitAI).

    When cloud mode is active, the install happens on the cloud machine.
    Token injection for CivitAI URLs happens locally before proxying.
    """
    from urllib.parse import parse_qs, urlparse

    from .models_config import get_civitai_token

    # Inject CivitAI token if needed (before cloud proxying)
    url = request.url
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    is_civitai = hostname == "civitai.com" or hostname.endswith(".civitai.com")

    if is_civitai:
        query_params = parse_qs(parsed.query)
        if "token" not in query_params:
            stored_token = get_civitai_token()
            if stored_token:
                separator = "&" if parsed.query else "?"
                url = f"{url}{separator}token={stored_token}"
            else:
                raise HTTPException(
                    status_code=400,
                    detail="CivitAI requires an API token for programmatic downloads. "
                    "Add your CivitAI API key in Settings > API Keys. "
                    "Get your API key at https://civitai.com/user/account",
                )

    # If connected to cloud, proxy with the (potentially modified) URL
    if cloud_manager.is_connected:
        logger.info("Proxying LoRA install to cloud")
        body = {"url": url, "filename": request.filename}
        try:
            response = await cloud_manager.api_request(
                method="POST",
                path="/api/v1/loras",
                body=body,
                timeout=300.0,
            )
        except Exception as e:
            logger.error(f"Cloud proxy request failed: {e}")
            raise HTTPException(
                status_code=502,
                detail=f"Cloud request failed: {e}",
            ) from e

        status = response.get("status", 200)
        if status >= 400:
            raise HTTPException(
                status_code=status,
                detail=response.get("error", "Cloud request failed"),
            )
        return response.get("data", {})

    # Local installation
    import re
    from urllib.parse import unquote

    import httpx

    from .download_models import http_get

    try:
        # Re-parse URL (may have been modified with token)
        parsed = urlparse(url)
        hostname = parsed.hostname or ""

        # Validate hostname is from allowed sources
        is_allowed = any(
            hostname == allowed or hostname.endswith(f".{allowed}")
            for allowed in ALLOWED_LORA_HOSTS
        )
        if not is_allowed:
            raise HTTPException(
                status_code=400,
                detail=f"URL must be from {' or '.join(sorted(ALLOWED_LORA_HOSTS))}",
            )

        # Determine filename from URL if not provided
        filename = request.filename
        if not filename:
            filename = unquote(parsed.path.split("/")[-1])
        # If still no filename (or it doesn't look like a file), try Content-Disposition
        if not filename or "." not in filename:
            # Use streaming GET instead of HEAD (some servers return 403 for HEAD)
            with httpx.Client(follow_redirects=True, timeout=10.0) as client:
                with client.stream("GET", url) as response:
                    if response.status_code == 401 or response.status_code == 403:
                        raise HTTPException(
                            status_code=response.status_code,
                            detail="Access denied. Check that the URL is correct and includes any required authentication.",
                        )
                    if response.status_code == 404:
                        raise HTTPException(
                            status_code=404,
                            detail="File not found. Check that the URL is correct.",
                        )
                    if response.status_code >= 400:
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"Failed to fetch URL: HTTP {response.status_code}",
                        )
                    content_disp = response.headers.get("content-disposition", "")
                    # Parse filename from Content-Disposition header
                    # e.g., 'attachment; filename="model.safetensors"'
                    match = re.search(
                        r'filename[*]?=["\']?([^"\';]+)["\']?', content_disp
                    )
                    if match:
                        filename = unquote(match.group(1).strip())
                    # Don't read the body - just close the connection

        if not filename or "." not in filename:
            raise HTTPException(
                status_code=400,
                detail="Could not determine filename from URL. Please provide a filename.",
            )

        # Validate file extension
        ext = Path(filename).suffix.lower()
        if ext not in LORA_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file extension '{ext}'. Allowed: {', '.join(sorted(LORA_EXTENSIONS))}",
            )

        lora_dir = get_lora_dir()
        dest_path = lora_dir / filename

        if dest_path.exists():
            raise HTTPException(
                status_code=409,
                detail=f"File '{filename}' already exists.",
            )

        # Install in a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, http_get, url, dest_path)

        size_mb = dest_path.stat().st_size / (1024 * 1024)
        file_info = LoRAFileInfo(
            name=dest_path.stem,
            path=str(dest_path),
            size_mb=round(size_mb, 2),
            folder=None,
        )

        return LoRAInstallResponse(
            message=f"Successfully installed '{filename}'",
            file=file_info,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"install_lora_file: Error installing LoRA: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


class LoRADeleteResponse(BaseModel):
    success: bool
    message: str


@app.delete("/api/v1/loras/{name}")
@cloud_proxy()
async def delete_lora_file(
    name: str,
    http_request: Request,
    cloud_manager: "CloudConnectionManager" = Depends(get_cloud_connection_manager),
):
    """Delete a LoRA file by name."""
    try:
        lora_dir = get_lora_dir()

        # Search for the file with any supported extension
        found_path = None
        for ext in LORA_EXTENSIONS:
            candidate = lora_dir / f"{name}{ext}"
            if candidate.exists():
                found_path = candidate
                break

        # Also check subdirectories
        if not found_path:
            for subdir in lora_dir.iterdir():
                if subdir.is_dir():
                    for ext in LORA_EXTENSIONS:
                        candidate = subdir / f"{name}{ext}"
                        if candidate.exists():
                            found_path = candidate
                            break
                    if found_path:
                        break

        if not found_path:
            raise HTTPException(
                status_code=404,
                detail=f"LoRA file '{name}' not found",
            )

        # Delete the file
        found_path.unlink()
        logger.info(f"Deleted LoRA file: {found_path}")

        return LoRADeleteResponse(
            success=True,
            message=f"Successfully deleted '{name}'",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"delete_lora_file: Error deleting LoRA: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/assets", response_model=AssetsResponse)
@cloud_proxy()
async def list_assets(
    http_request: Request,
    type: str | None = Query(None, description="Filter by asset type (image, video)"),
    cloud_manager: "CloudConnectionManager" = Depends(get_cloud_connection_manager),
):
    """List available asset files in the assets directory and its subdirectories.

    When cloud mode is active, lists assets from the cloud server instead.
    """

    def process_asset_file(
        file_path: Path, assets_dir: Path, asset_type: str
    ) -> AssetFileInfo:
        """Extract asset file metadata."""
        size_mb = file_path.stat().st_size / (1024 * 1024)
        created_at = file_path.stat().st_ctime
        relative_path = file_path.relative_to(assets_dir)
        folder = (
            str(relative_path.parent) if relative_path.parent != Path(".") else None
        )
        return AssetFileInfo(
            name=file_path.stem,
            path=str(file_path),
            size_mb=round(size_mb, 2),
            folder=folder,
            type=asset_type,
            created_at=created_at,
        )

    try:
        assets_dir = get_assets_dir()
        asset_files: list[AssetFileInfo] = []

        if type == "image":
            extensions = IMAGE_EXTENSIONS
        elif type == "video":
            extensions = VIDEO_EXTENSIONS
        else:
            extensions = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

        for file_path in iter_files(assets_dir, extensions):
            ext = file_path.suffix.lower()
            asset_type = "image" if ext in IMAGE_EXTENSIONS else "video"
            asset_files.append(process_asset_file(file_path, assets_dir, asset_type))

        # Sort by created_at (most recent first), then by folder and name
        asset_files.sort(key=lambda x: (-x.created_at, x.folder or "", x.name))
        return AssetsResponse(assets=asset_files)

    except Exception as e:  # pragma: no cover - defensive logging
        logger.error(f"list_assets: Error listing asset files: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/assets", response_model=AssetFileInfo)
async def upload_asset(
    request: Request,
    filename: str = Query(...),
    cloud_manager: "CloudConnectionManager" = Depends(get_cloud_connection_manager),
):
    """Upload an asset file (image or video) to the assets directory.

    When cloud mode is active, the file is uploaded to the cloud server instead.
    """

    try:
        # Validate file type - support both images and videos
        allowed_extensions = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

        file_extension = Path(filename).suffix.lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}",
            )

        # Determine asset type
        if file_extension in IMAGE_EXTENSIONS:
            asset_type = "image"
            content_type_map = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".bmp": "image/bmp",
            }
        else:
            asset_type = "video"
            content_type_map = {
                ".mp4": "video/mp4",
                ".avi": "video/x-msvideo",
                ".mov": "video/quicktime",
                ".mkv": "video/x-matroska",
                ".webm": "video/webm",
            }
        content_type = content_type_map.get(file_extension, "application/octet-stream")

        # Read file content from request body
        content = await request.body()

        # Validate file size (50MB limit)
        max_size = 50 * 1024 * 1024  # 50MB
        if len(content) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum of {max_size / (1024 * 1024):.0f}MB",
            )

        # If cloud mode is active, upload to cloud AND save locally for thumbnails
        if cloud_manager.is_connected:
            return await upload_asset_to_cloud(
                cloud_manager,
                content,
                filename,
                content_type,
                asset_type,
            )

        # Local mode: save to local assets directory
        assets_dir = get_assets_dir()
        assets_dir.mkdir(parents=True, exist_ok=True)

        # Save file to assets directory
        file_path = assets_dir / filename
        file_path.write_bytes(content)

        # Return file info matching AssetFileInfo structure
        size_mb = len(content) / (1024 * 1024)
        created_at = file_path.stat().st_ctime
        relative_path = file_path.relative_to(assets_dir)
        folder = (
            str(relative_path.parent) if relative_path.parent != Path(".") else None
        )

        logger.info(f"upload_asset: Uploaded {asset_type} file: {file_path}")
        return AssetFileInfo(
            name=file_path.stem,
            path=str(file_path),
            size_mb=round(size_mb, 2),
            folder=folder,
            type=asset_type,
            created_at=created_at,
        )

    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - defensive logging
        logger.error(f"upload_asset: Error uploading asset file: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/assets/{asset_path:path}")
async def serve_asset(asset_path: str):
    """Serve an asset file (for thumbnails/previews).

    Handles both relative paths and absolute paths (e.g., from cloud).
    For absolute paths, extracts the filename and serves from local assets.
    """
    try:
        assets_dir = get_assets_dir()

        # Handle absolute paths (e.g., from cloud: /root/.daydream-scope/assets/filename.png)
        # Extract just the filename to serve from local cache
        if asset_path.startswith("/") or asset_path.startswith("root/"):
            # Extract just the filename
            filename = Path(asset_path).name
            file_path = assets_dir / filename
            logger.debug(
                f"serve_asset: Extracted filename '{filename}' from absolute path"
            )
        else:
            file_path = assets_dir / asset_path

        # Security check: ensure the path is within assets directory
        try:
            file_path = file_path.resolve()
            assets_dir_resolved = assets_dir.resolve()
            if not str(file_path).startswith(str(assets_dir_resolved)):
                raise HTTPException(status_code=403, detail="Access denied")
        except Exception:
            raise HTTPException(status_code=403, detail="Invalid path") from None

        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="Asset not found")

        # Determine media type based on extension
        file_extension = file_path.suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".mp4": "video/mp4",
            ".avi": "video/x-msvideo",
            ".mov": "video/quicktime",
            ".mkv": "video/x-matroska",
            ".webm": "video/webm",
        }
        media_type = media_types.get(file_extension, "application/octet-stream")

        return FileResponse(file_path, media_type=media_type)

    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - defensive logging
        logger.error(f"serve_asset: Error serving asset file: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/models/status")
@cloud_proxy()
async def get_model_status(
    http_request: Request,
    pipeline_id: str,
    cloud_manager: "CloudConnectionManager" = Depends(get_cloud_connection_manager),
):
    """Check if models for a pipeline are downloaded and get download progress."""
    try:
        progress = download_progress_manager.get_progress(pipeline_id)

        # If download is in progress, always report as not downloaded
        if progress and progress.get("is_downloading"):
            return {"downloaded": False, "progress": progress}

        # Check if files actually exist
        downloaded = models_are_downloaded(pipeline_id)

        # Clean up progress if download is complete
        if downloaded and progress:
            download_progress_manager.clear_progress(pipeline_id)
            progress = None

        return {"downloaded": downloaded, "progress": progress}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking model status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/models/download")
@cloud_proxy(timeout=60.0)
async def download_pipeline_models(
    request: DownloadModelsRequest,
    http_request: Request,
    cloud_manager: "CloudConnectionManager" = Depends(get_cloud_connection_manager),
):
    """Download models for a specific pipeline."""
    try:
        if not request.pipeline_id:
            raise HTTPException(status_code=400, detail="pipeline_id is required")

        pipeline_id = request.pipeline_id

        # Local mode: check if download already in progress
        existing_progress = download_progress_manager.get_progress(pipeline_id)
        if existing_progress and existing_progress.get("is_downloading"):
            raise HTTPException(
                status_code=409,
                detail=f"Download already in progress for {pipeline_id}",
            )

        # Clear any previous error state before starting a new download
        download_progress_manager.clear_progress(pipeline_id)

        # Download in a background thread to avoid blocking
        import threading

        def _is_auth_error(error: Exception) -> bool:
            """Check if a download error is authentication-related."""
            msg = str(error)
            return "401" in msg or "403" in msg or "Unauthorized" in msg

        def download_in_background():
            """Run download in background thread."""
            try:
                download_models(pipeline_id)
                download_progress_manager.mark_complete(pipeline_id)
            except Exception as e:
                logger.error(f"Error downloading models for {pipeline_id}: {e}")
                if _is_auth_error(e):
                    user_msg = "Download failed due to authentication error. For HuggingFace models, make sure your HuggingFace key is configured in Settings > API Keys."
                else:
                    user_msg = "Download failed. Check the server logs for details."
                download_progress_manager.mark_error(pipeline_id, user_msg)

        thread = threading.Thread(target=download_in_background)
        thread.daemon = True
        thread.start()

        return {"message": f"Model download started for {pipeline_id}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting model download: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


def is_spout_available() -> bool:
    """Check if Spout is available (native Windows only, not WSL)."""
    return sys.platform == "win32"


def is_ndi_output_available() -> bool:
    """Check if NDI SDK is available for output."""
    from scope.core.ndi import is_available

    return is_available()


_source_discovery_cache: dict[str, tuple[float, list]] = {}
_SOURCE_DISCOVERY_TTL = 10  # seconds


def _resolve_input_source_class(source_type: str):
    """Resolve a source_type string to its InputSource class, or raise HTTPException."""
    from scope.core.inputs import get_input_source_classes

    source_classes = get_input_source_classes()
    source_class = source_classes.get(source_type)
    if source_class is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown input source type '{source_type}'. "
            f"Available types: {list(source_classes.keys())}",
        )
    if not source_class.is_available():
        raise HTTPException(
            status_code=400,
            detail=f"Input source '{source_type}' is not available on this system.",
        )
    return source_class


@app.get("/api/v1/input-sources")
async def list_input_source_types():
    """List available input source types with their availability status."""
    from scope.core.inputs import get_input_source_classes

    sources = []
    for cls in get_input_source_classes().values():
        sources.append(
            {
                "source_id": cls.source_id,
                "source_name": cls.source_name,
                "source_description": cls.source_description,
                "available": cls.is_available(),
            }
        )

    return {"input_sources": sources}


@app.get("/api/v1/input-sources/{source_type}/sources")
def list_input_sources(source_type: str, timeout_ms: int = Query(5000)):
    """List discovered sources for a given input source type."""
    source_class = _resolve_input_source_class(source_type)

    # Return cached results if still fresh
    cached = _source_discovery_cache.get(source_type)
    if cached is not None:
        ts, sources = cached
        if time.monotonic() - ts < _SOURCE_DISCOVERY_TTL:
            return {"source_type": source_type, "sources": sources}

    try:
        instance = source_class()
        try:
            discovered = instance.list_sources(timeout_ms=timeout_ms)
            sources = [
                {
                    "name": s.name,
                    "identifier": s.identifier,
                    "metadata": s.metadata,
                }
                for s in discovered
            ]
            _source_discovery_cache[source_type] = (time.monotonic(), sources)
            return {"source_type": source_type, "sources": sources}
        finally:
            instance.close()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing sources for '{source_type}': {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/input-sources/{source_type}/sources/{identifier:path}/resolution")
def get_input_source_resolution(
    source_type: str, identifier: str, timeout_ms: int = Query(5000)
):
    """Probe the native resolution of a specific input source."""
    source_class = _resolve_input_source_class(source_type)

    try:
        instance = source_class()
        try:
            resolution = instance.get_source_resolution(
                identifier, timeout_ms=timeout_ms
            )
            if resolution is None:
                raise HTTPException(
                    status_code=408,
                    detail=f"Could not determine resolution for '{identifier}' "
                    f"within {timeout_ms}ms.",
                )
            width, height = resolution
            return {"width": width, "height": height}
        finally:
            instance.close()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error probing resolution for '{source_type}/{identifier}': {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/input-sources/{source_type}/sources/{identifier:path}/stream")
async def stream_input_source_preview(
    source_type: str, identifier: str, fps: int = Query(2, ge=1, le=30)
):
    """MJPEG stream of an input source for live preview."""
    source_class = _resolve_input_source_class(source_type)

    async def _generate():
        from PIL import Image

        max_w = 320
        interval = 1.0 / fps
        receiver = None
        loop = asyncio.get_event_loop()

        try:
            # Create persistent receiver
            receiver = source_class()
            connected = await loop.run_in_executor(None, receiver.connect, identifier)
            if not connected:
                logger.warning(f"Preview stream: could not connect to '{identifier}'")
                return

            while True:
                frame = await loop.run_in_executor(None, receiver.receive_frame, 200)

                if frame is not None:
                    h, w = frame.shape[:2]
                    if w > max_w:
                        scale = max_w / w
                        new_w = max_w
                        new_h = int(h * scale)
                        img = Image.fromarray(frame).resize(
                            (new_w, new_h), Image.NEAREST
                        )
                    else:
                        img = Image.fromarray(frame)

                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=70)
                    jpeg = buf.getvalue()

                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
                        b"\r\n" + jpeg + b"\r\n"
                    )

                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass
        finally:
            if receiver is not None:
                receiver.close()

    return StreamingResponse(
        _generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/v1/hardware/info", response_model=HardwareInfoResponse)
async def get_hardware_info(
    cloud_manager: "CloudConnectionManager" = Depends(get_cloud_connection_manager),
):
    """Get hardware information including available VRAM and Spout availability.

    In cloud mode (when connected to cloud), this proxies the request to the
    cloud-hosted scope backend to get the cloud GPU's hardware info.
    """
    try:
        # If connected to cloud, proxy the request to get cloud's hardware info
        if cloud_manager.is_connected:
            return await get_hardware_info_from_cloud(
                cloud_manager,
                is_spout_available(),
                is_ndi_output_available(),
            )

        # Local mode: get local hardware info
        import torch  # Lazy import to avoid loading at CLI startup

        vram_gb = None

        if torch.cuda.is_available():
            # Get total VRAM from the first GPU (in bytes), convert to GB
            _, total_mem = torch.cuda.mem_get_info(0)
            vram_gb = total_mem / (1024**3)

        return HardwareInfoResponse(
            vram_gb=vram_gb,
            spout_available=is_spout_available(),
            ndi_available=is_ndi_output_available(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting hardware info: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/keys", response_model=ApiKeysListResponse)
async def list_api_keys():
    """List all registered API key services with their status."""
    import os

    from huggingface_hub import get_token

    from .models_config import get_civitai_token, get_civitai_token_source

    # HuggingFace
    hf_token = get_token()
    hf_env_var_set = bool(os.environ.get("HF_TOKEN"))
    if hf_token:
        hf_source = "env_var" if hf_env_var_set else "stored"
    else:
        hf_source = None

    hf_key = ApiKeyInfo(
        id="huggingface",
        name="HuggingFace",
        description="Required for downloading gated models",
        is_set=hf_token is not None,
        source=hf_source,
        env_var="HF_TOKEN",
        key_url="https://huggingface.co/settings/tokens",
    )

    # CivitAI
    civitai_token = get_civitai_token()

    civitai_key = ApiKeyInfo(
        id="civitai",
        name="CivitAI",
        description="Required for downloading LoRAs from CivitAI",
        is_set=civitai_token is not None,
        source=get_civitai_token_source(),
        env_var="CIVITAI_API_TOKEN",
        key_url="https://civitai.com/user/account",
    )

    return ApiKeysListResponse(keys=[hf_key, civitai_key])


@app.put("/api/v1/keys/{service_id}", response_model=ApiKeySetResponse)
async def set_api_key(service_id: str, request: ApiKeySetRequest):
    """Set/save an API key for a service."""
    import os

    from .models_config import CIVITAI_TOKEN_ENV_VAR, set_civitai_token

    if service_id == "huggingface":
        if os.environ.get("HF_TOKEN"):
            raise HTTPException(
                status_code=409,
                detail="HF_TOKEN environment variable is already set. Remove it to manage this key from the UI.",
            )

        from huggingface_hub import login

        login(token=request.value, add_to_git_credential=False)
        return ApiKeySetResponse(success=True, message="HuggingFace token saved")

    elif service_id == "civitai":
        if os.environ.get(CIVITAI_TOKEN_ENV_VAR):
            raise HTTPException(
                status_code=409,
                detail="CIVITAI_API_TOKEN environment variable is already set. Remove it to manage this key from the UI.",
            )

        set_civitai_token(request.value)
        return ApiKeySetResponse(success=True, message="CivitAI token saved")

    else:
        raise HTTPException(status_code=404, detail=f"Unknown service: {service_id}")


@app.delete("/api/v1/keys/{service_id}", response_model=ApiKeyDeleteResponse)
async def delete_api_key(service_id: str):
    """Remove a stored API key for a service."""
    import os

    from .models_config import (
        clear_civitai_token,
        get_civitai_token_source,
    )

    if service_id == "huggingface":
        env_var_set = bool(os.environ.get("HF_TOKEN"))
        if env_var_set:
            raise HTTPException(
                status_code=409,
                detail="Cannot remove token set via HF_TOKEN environment variable. Unset the environment variable instead.",
            )

        from huggingface_hub import logout

        logout()
        return ApiKeyDeleteResponse(success=True, message="HuggingFace token removed")

    elif service_id == "civitai":
        source = get_civitai_token_source()
        if source == "env_var":
            raise HTTPException(
                status_code=409,
                detail="Cannot remove token set via CIVITAI_API_TOKEN environment variable. Unset the environment variable instead.",
            )
        if source != "stored":
            raise HTTPException(status_code=404, detail="No CivitAI token to remove")

        clear_civitai_token()
        return ApiKeyDeleteResponse(success=True, message="CivitAI token removed")

    else:
        raise HTTPException(status_code=404, detail=f"Unknown service: {service_id}")


@app.get("/api/v1/logs/current")
async def get_current_logs():
    """Get the most recent application log file for bug reporting."""
    try:
        log_file_path = get_most_recent_log_file()

        if log_file_path is None or not log_file_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Log file not found. The application may not have logged anything yet.",
            )

        # Read the entire file into memory to avoid Content-Length issues
        # with actively written log files.
        # Use errors='replace' to handle non-UTF-8 bytes gracefully (e.g., Windows-1252
        # characters from subprocess output or exception messages on Windows).
        log_content = log_file_path.read_text(encoding="utf-8", errors="replace")

        # Return as a text response with proper headers for download
        return Response(
            content=log_content,
            media_type="text/plain",
            headers={
                "Content-Disposition": f'attachment; filename="{log_file_path.name.replace(".log", ".txt")}"'
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving log file: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# Plugin Management API Endpoints


def _convert_plugin_dict_to_info(plugin_dict: dict) -> "PluginInfo":
    """Convert a plugin dictionary from PluginManager to PluginInfo schema."""
    from .schema import PluginInfo, PluginPipelineInfo, PluginSource

    pipelines = [
        PluginPipelineInfo(
            pipeline_id=p["pipeline_id"],
            pipeline_name=p["pipeline_name"],
        )
        for p in plugin_dict.get("pipelines", [])
    ]

    return PluginInfo(
        name=plugin_dict["name"],
        version=plugin_dict.get("version"),
        author=plugin_dict.get("author"),
        description=plugin_dict.get("description"),
        source=PluginSource(plugin_dict.get("source", "pypi")),
        editable=plugin_dict.get("editable", False),
        editable_path=plugin_dict.get("editable_path"),
        pipelines=pipelines,
        latest_version=plugin_dict.get("latest_version"),
        update_available=plugin_dict.get("update_available"),
        package_spec=plugin_dict.get("package_spec"),
    )


@app.get("/api/v1/plugins")
async def list_plugins():
    """List all installed plugins with metadata."""
    from scope.core.plugins import get_plugin_manager

    from .schema import FailedPluginInfoSchema, PluginListResponse

    try:
        plugin_manager = get_plugin_manager()
        plugins_data = await plugin_manager.list_plugins_async()

        plugins = [_convert_plugin_dict_to_info(p) for p in plugins_data]

        failed = [
            FailedPluginInfoSchema(
                package_name=f.package_name,
                entry_point_name=f.entry_point_name,
                error_type=f.error_type,
                error_message=f.error_message,
            )
            for f in plugin_manager.get_failed_plugins()
        ]

        return PluginListResponse(
            plugins=plugins, total=len(plugins), failed_plugins=failed
        )
    except Exception as e:
        logger.error(f"Error listing plugins: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/plugins")
async def install_plugin(
    request: Request,
    pipeline_manager: "PipelineManager" = Depends(get_pipeline_manager),
):
    """Install a plugin from PyPI, git URL, or local path."""
    from scope.core.plugins import (
        PluginDependencyError,
        PluginInstallError,
        PluginNameCollisionError,
        get_plugin_manager,
    )

    from .schema import PluginInstallRequest, PluginInstallResponse

    # Parse request body
    body = await request.json()
    install_request = PluginInstallRequest(**body)

    logger.info(f"Installing plugin: {install_request.package}")
    try:
        plugin_manager = get_plugin_manager()

        result = await plugin_manager.install_plugin_async(
            package=install_request.package,
            editable=install_request.editable,
            upgrade=install_request.upgrade,
            pre=install_request.pre,
            force=install_request.force,
        )

        plugin_info = None
        plugin_name = install_request.package
        if result.get("plugin"):
            plugin_info = _convert_plugin_dict_to_info(result["plugin"])
            plugin_name = plugin_info.name

        logger.info(f"Plugin installed: {plugin_name}")
        return PluginInstallResponse(
            success=result["success"],
            message=result["message"],
            plugin=plugin_info,
        )

    except PluginDependencyError as e:
        logger.error(
            f"Plugin install failed (dependency error): {install_request.package} - {e}"
        )
        raise HTTPException(
            status_code=422,
            detail=(
                f"Failed to install {install_request.package}: "
                "dependency conflict. Check server logs for details."
            ),
        ) from e
    except PluginNameCollisionError as e:
        logger.error(
            f"Plugin install failed (name collision): {install_request.package} - {e}"
        )
        raise HTTPException(
            status_code=409,
            detail=f"Plugin name collision: {install_request.package}",
        ) from e
    except PluginInstallError as e:
        logger.error(f"Plugin install failed: {install_request.package} - {e}")
        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to install {install_request.package}. "
                "Check server logs for details."
            ),
        ) from e
    except Exception as e:
        logger.error(f"Plugin install failed: {install_request.package} - {e}")
        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to install {install_request.package}. "
                "Check server logs for details."
            ),
        ) from e


@app.delete("/api/v1/plugins/{name}")
async def uninstall_plugin(
    name: str,
    pipeline_manager: "PipelineManager" = Depends(get_pipeline_manager),
):
    """Uninstall a plugin, cleaning up loaded pipelines."""
    from scope.core.plugins import (
        PluginInstallError,
        PluginNotFoundError,
        get_plugin_manager,
    )

    from .schema import PluginUninstallResponse

    logger.info(f"Uninstalling plugin: {name}")
    try:
        plugin_manager = get_plugin_manager()

        result = await plugin_manager.uninstall_plugin_async(
            name=name,
            pipeline_manager=pipeline_manager,
        )

        logger.info(f"Plugin uninstalled: {name}")
        return PluginUninstallResponse(
            success=result["success"],
            message=result["message"],
            unloaded_pipelines=result.get("unloaded_pipelines", []),
        )

    except PluginNotFoundError as e:
        logger.error(f"Plugin uninstall failed (not found): {name} - {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Plugin '{name}' not found",
        ) from e
    except PluginInstallError as e:
        logger.error(f"Plugin uninstall failed: {name} - {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to uninstall {name}. Check server logs for details.",
        ) from e
    except Exception as e:
        logger.error(f"Plugin uninstall failed: {name} - {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to uninstall {name}. Check server logs for details.",
        ) from e


@app.post("/api/v1/plugins/{name}/reload")
async def reload_plugin(
    name: str,
    request: Request,
    pipeline_manager: "PipelineManager" = Depends(get_pipeline_manager),
):
    """Reload an editable plugin for development (without server restart)."""
    from scope.core.plugins import (
        PluginInUseError,
        PluginNotEditableError,
        PluginNotFoundError,
        get_plugin_manager,
    )

    from .schema import PluginReloadRequest, PluginReloadResponse

    # Parse request body
    body = await request.json()
    reload_request = PluginReloadRequest(**body)

    try:
        plugin_manager = get_plugin_manager()

        result = await plugin_manager.reload_plugin_async(
            name=name,
            force=reload_request.force,
            pipeline_manager=pipeline_manager,
        )

        return PluginReloadResponse(
            success=result["success"],
            message=result["message"],
            reloaded_pipelines=result.get("reloaded_pipelines", []),
            added_pipelines=result.get("added_pipelines", []),
            removed_pipelines=result.get("removed_pipelines", []),
        )

    except PluginNotFoundError as e:
        logger.error(f"Plugin reload failed (not found): {name} - {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Plugin '{name}' not found",
        ) from e
    except PluginNotEditableError as e:
        logger.error(f"Plugin reload failed (not editable): {name} - {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Plugin '{name}' is not installed in editable mode",
        ) from e
    except PluginInUseError as e:
        logger.error(f"Plugin reload failed (in use): {name} - {e}")
        raise HTTPException(
            status_code=409,
            detail={
                "message": f"Plugin '{name}' has loaded pipelines. Use force=true to unload them.",
                "loaded_pipelines": e.loaded_pipelines,
            },
        ) from e
    except Exception as e:
        logger.error(f"Plugin reload failed: {name} - {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload {name}. Check server logs for details.",
        ) from e


# =============================================================================
# Cloud Integration Endpoints
# =============================================================================


@app.post("/api/v1/cloud/connect", response_model=CloudStatusResponse)
async def connect_to_cloud(
    request: CloudConnectRequest,
    cloud_manager: "CloudConnectionManager" = Depends(get_cloud_connection_manager),
):
    """Connect to cloud for remote GPU inference.

    This establishes a WebSocket connection to the cloud runner,
    which stays open until disconnect is called. Once connected:
    - Pipeline loading is proxied to the cloud-hosted scope backend
    - Video from the browser flows through the backend to cloud
    - The cloud runner stays warm and ready for video processing

    Credentials can be provided in the request body or via CLI args:
    --cloud-app-id and --cloud-api-key (or SCOPE_CLOUD_APP_ID/SCOPE_CLOUD_API_KEY env vars).

    Note: The connection may take 1-2 minutes on cold start while the
    cloud runner initializes.

    Architecture:
        Browser → Backend (WebRTC) → cloud (WebRTC) → Backend → Browser
        Spout → Backend → cloud (WebRTC) → Backend → Spout/Browser
    """
    try:
        # Use request body credentials if provided, otherwise fall back to CLI/env
        app_id = request.app_id or os.environ.get("SCOPE_CLOUD_APP_ID")
        api_key = request.api_key or os.environ.get("SCOPE_CLOUD_API_KEY")

        if not app_id:
            raise HTTPException(
                status_code=400,
                detail="cloud credentials not configured. Use --cloud-app-id and --cloud-api-key CLI args, "
                "or SCOPE_CLOUD_APP_ID and SCOPE_CLOUD_API_KEY environment variables.",
            )

        logger.info(
            f"Connecting to cloud (background): {app_id} (user_id: {request.user_id})"
        )
        await cloud_manager.connect_background(app_id, api_key, request.user_id)

        credentials_configured = bool(os.environ.get("SCOPE_CLOUD_APP_ID"))
        return CloudStatusResponse(
            connected=False,
            connecting=True,
            webrtc_connected=False,
            app_id=app_id,
            credentials_configured=credentials_configured,
        )
    except Exception as e:
        logger.error(f"Error connecting to cloud: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/cloud/disconnect", response_model=CloudStatusResponse)
async def disconnect_from_cloud(
    cloud_manager: "CloudConnectionManager" = Depends(get_cloud_connection_manager),
):
    """Disconnect from cloud.

    This closes the WebSocket and WebRTC connections to cloud, returning
    to local GPU processing mode. Any in-progress operations will be interrupted.
    """
    try:
        await cloud_manager.disconnect()
        credentials_configured = bool(os.environ.get("SCOPE_CLOUD_APP_ID"))
        return CloudStatusResponse(
            connected=False,
            webrtc_connected=False,
            app_id=None,
            credentials_configured=credentials_configured,
        )
    except Exception as e:
        logger.error(f"Error disconnecting from cloud: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/cloud/status", response_model=CloudStatusResponse)
async def get_cloud_status(
    cloud_manager: "CloudConnectionManager" = Depends(get_cloud_connection_manager),
):
    """Get current cloud connection status."""
    status = cloud_manager.get_status()
    # Check if credentials are configured via CLI/env
    credentials_configured = bool(os.environ.get("SCOPE_CLOUD_APP_ID"))
    return CloudStatusResponse(**status, credentials_configured=credentials_configured)


@app.get("/api/v1/cloud/stats")
async def get_cloud_stats(
    cloud_manager: "CloudConnectionManager" = Depends(get_cloud_connection_manager),
):
    """Get detailed cloud connection statistics.

    Returns connection stats including:
    - Uptime
    - WebRTC offers sent/successful
    - ICE candidates sent
    - API requests sent/successful

    Also prints stats to the server log for debugging.
    """
    # Print stats to log
    cloud_manager.print_stats()

    # Return full status with stats
    return cloud_manager.get_status()


@app.get("/{path:path}")
async def serve_frontend(request: Request, path: str):
    """Serve the frontend for all non-API routes (fallback for client-side routing)."""
    frontend_dist = Path(__file__).parent.parent.parent.parent / "frontend" / "dist"

    # Only serve SPA if frontend dist exists (production mode)
    if not frontend_dist.exists():
        raise HTTPException(status_code=404, detail="Frontend not built")

    # Check if requesting a specific file that exists
    file_path = frontend_dist / path
    if file_path.exists() and file_path.is_file():
        # Determine media type based on extension to fix MIME type issues on Windows
        file_extension = file_path.suffix.lower()
        media_types = {
            ".js": "application/javascript",
            ".mjs": "application/javascript",
            ".css": "text/css",
            ".html": "text/html",
            ".json": "application/json",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
            ".ico": "image/x-icon",
            ".woff": "font/woff",
            ".woff2": "font/woff2",
            ".ttf": "font/ttf",
            ".eot": "application/vnd.ms-fontobject",
        }
        media_type = media_types.get(file_extension)
        return FileResponse(file_path, media_type=media_type)

    # Fallback to index.html for SPA routing
    # This ensures clients like Electron alway fetch the latest HTML (which references hashed assets)
    index_file = frontend_dist / "index.html"
    if index_file.exists():
        return FileResponse(
            index_file,
            media_type="text/html",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    raise HTTPException(status_code=404, detail="Frontend index.html not found")


def open_browser_when_ready(host: str, port: int, server):
    """Open browser when server is ready, with fallback to URL logging."""
    # Wait for server to be ready
    while not getattr(server, "started", False):
        time.sleep(0.1)

    # Determine the URL to open
    url = (
        f"http://localhost:{port}"
        if host in ["0.0.0.0", "127.0.0.1"]
        else f"http://{host}:{port}"
    )

    try:
        success = webbrowser.open(url)
        if success:
            logger.info(f"🌐 Opened browser at {url}")
    except Exception:
        success = False

    if not success:
        logger.info(f"🌐 UI is available at: {url}")


def run_server(reload: bool, host: str, port: int, no_browser: bool):
    """Run the Daydream Scope server."""

    from scope.core.pipelines.registry import (
        PipelineRegistry,  # noqa: F401 - imported for side effects (registry initialization)
    )

    # Configure static file serving
    configure_static_files()

    # Check if we're in production mode (frontend dist exists)
    frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"
    is_production = frontend_dist.exists()

    if is_production:
        # Create server instance for production mode
        config = uvicorn.Config(
            "scope.server.app:app",
            host=host,
            port=port,
            reload=reload,
            log_config=None,  # Use our logging config, don't override it
        )
        server = uvicorn.Server(config)

        # Start browser opening thread (unless disabled)
        if not no_browser:
            browser_thread = threading.Thread(
                target=open_browser_when_ready,
                args=(host, port, server),
                daemon=True,
            )
            browser_thread.start()
        else:
            logger.info("main: Skipping browser auto-launch due to --no-browser")

        # Run the server
        try:
            server.run()
        except KeyboardInterrupt:
            pass  # Clean shutdown on Ctrl+C
    else:
        # Development mode - just run normally
        uvicorn.run(
            "scope.server.app:app",
            host=host,
            port=port,
            reload=reload,
            log_config=None,  # Use our logging config, don't override it
        )


@click.group(invoke_without_command=True)
@click.option("--version", is_flag=True, help="Show version information and exit")
@click.option(
    "--reload", is_flag=True, help="Enable auto-reload for development (default: False)"
)
@click.option("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
@click.option("--port", default=8000, help="Port to bind to (default: 8000)")
@click.option(
    "-N",
    "--no-browser",
    is_flag=True,
    help="Do not automatically open a browser window after the server starts",
)
@click.option(
    "--cloud-app-id",
    default="Daydream/scope-app--prod/ws",
    envvar="SCOPE_CLOUD_APP_ID",
    help="Cloud app ID for cloud mode (e.g., 'username/scope-app')",
)
@click.option(
    "--cloud-api-key",
    default=None,
    envvar="SCOPE_CLOUD_API_KEY",
    help="Cloud API key for cloud mode",
)
@click.pass_context
def main(
    ctx,
    version: bool,
    reload: bool,
    host: str,
    port: int,
    no_browser: bool,
    cloud_app_id: str | None,
    cloud_api_key: str | None,
):
    # Handle version flag
    if version:
        print_version_info()
        sys.exit(0)

    # Store cloud credentials in environment for app access
    if cloud_app_id:
        os.environ["SCOPE_CLOUD_APP_ID"] = cloud_app_id
    if cloud_api_key:
        os.environ["SCOPE_CLOUD_API_KEY"] = cloud_api_key

    # If no subcommand was invoked, run the server
    if ctx.invoked_subcommand is None:
        run_server(reload, host, port, no_browser)


@main.command()
def plugins():
    """List all installed plugins."""
    import asyncio

    from scope.core.plugins.manager import get_plugin_manager

    @suppress_init_output
    def _list_plugins():
        pm = get_plugin_manager()
        pm.load_plugins()
        return asyncio.run(pm.list_plugins_async())

    plugin_list = _list_plugins()

    if not plugin_list:
        click.echo("No plugins installed.")
        return

    click.echo(f"{len(plugin_list)} plugin(s) installed:\n")

    for plugin in plugin_list:
        name = plugin["name"]
        version = plugin.get("version", "unknown")
        source = plugin.get("source", "unknown")
        pipelines = plugin.get("pipelines", [])

        click.echo(f"  {name} ({version})")
        click.echo(f"    Source: {source}")
        if pipelines:
            pipeline_ids = [p["pipeline_id"] for p in pipelines]
            click.echo(f"    Pipelines: {', '.join(pipeline_ids)}")


@main.command()
def pipelines():
    """List all available pipelines."""

    @suppress_init_output
    def _load_pipelines():
        from scope.core.pipelines.registry import PipelineRegistry

        return PipelineRegistry.list_pipelines()

    all_pipelines = _load_pipelines()

    if not all_pipelines:
        click.echo("No pipelines available.")
        return

    click.echo(f"{len(all_pipelines)} pipeline(s) available:\n")

    # List all pipelines
    for pipeline_id in all_pipelines:
        click.echo(f"  • {pipeline_id}")


@main.command()
@click.argument("package", required=False)
@click.option("--upgrade", is_flag=True, help="Upgrade package to latest version")
@click.option(
    "-e", "--editable", help="Install a project in editable mode from this path"
)
@click.option(
    "--pre", is_flag=True, help="Include pre-release and development versions"
)
@click.option("--force", is_flag=True, help="Skip dependency validation")
def install(package, upgrade, editable, pre, force):
    """Install a plugin."""
    import asyncio

    from scope.core.plugins.manager import (
        PluginDependencyError,
        PluginInstallError,
        PluginNameCollisionError,
        get_plugin_manager,
    )

    if not package and not editable:
        click.echo("Error: Must specify a package or use -e/--editable", err=True)
        sys.exit(1)

    # Determine what to install
    install_package = editable if editable else package
    is_editable = bool(editable)

    @suppress_init_output
    def _install():
        pm = get_plugin_manager()
        return asyncio.run(
            pm.install_plugin_async(
                package=install_package,
                editable=is_editable,
                upgrade=upgrade,
                pre=pre,
                force=force,
            )
        )

    try:
        result = _install()
        click.echo(result["message"])
    except PluginDependencyError as e:
        click.echo(f"Dependency error: {e}", err=True)
        click.echo("\nUse --force to install anyway (may break environment)", err=True)
        sys.exit(1)
    except PluginNameCollisionError as e:
        click.echo(f"Name collision: {e}", err=True)
        sys.exit(1)
    except PluginInstallError as e:
        click.echo(f"Installation failed: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("name", required=True)
def uninstall(name):
    """Uninstall a plugin."""
    import asyncio

    from scope.core.plugins.manager import (
        PluginInstallError,
        PluginNotFoundError,
        get_plugin_manager,
    )

    @suppress_init_output
    def _uninstall():
        pm = get_plugin_manager()
        pm.load_plugins()
        return asyncio.run(pm.uninstall_plugin_async(name=name))

    try:
        result = _uninstall()
        click.echo(result["message"])
        if result.get("unloaded_pipelines"):
            click.echo(f"Unloaded pipelines: {', '.join(result['unloaded_pipelines'])}")
    except PluginNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except PluginInstallError as e:
        click.echo(f"Uninstall failed: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
