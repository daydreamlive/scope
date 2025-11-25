import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .download_models import download_models
from .logs_config import (
    get_logs_dir,
    get_most_recent_log_file,
)
from .models_config import ensure_models_dir, get_models_dir, models_are_downloaded
from .pipeline_manager import PipelineManager
from .schema import (
    HardwareInfoResponse,
    HealthResponse,
    PipelineLoadRequest,
    PipelineStatusResponse,
    WebRTCOfferRequest,
    WebRTCOfferResponse,
)
from .webrtc import WebRTCManager

logger = logging.getLogger(__name__)

# Select pipeline depending on the "PIPELINE" environment variable
PIPELINE = os.getenv("PIPELINE", None)

# Global WebRTC manager instance
webrtc_manager = None
# Global pipeline manager instance
pipeline_manager = None


async def prewarm_pipeline(pipeline_id: str):
    """Background task to pre-warm the pipeline without blocking startup."""
    try:
        await asyncio.wait_for(
            pipeline_manager.load_pipeline(pipeline_id),
            timeout=300,  # 5 minute timeout for pipeline loading
        )
    except Exception as e:
        logger.error(f"Error pre-warming pipeline {pipeline_id} in background: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for startup and shutdown events."""
    # Startup
    global webrtc_manager, pipeline_manager

    # Check CUDA availability before proceeding
    if not torch.cuda.is_available():
        error_msg = (
            "CUDA is not available on this system. "
            "This application currently requires a CUDA-compatible GPU and "
            "other hardware will be supported in the future."
        )
        logger.error(error_msg)
        sys.exit(1)

    # Log logs directory
    logs_dir = get_logs_dir()
    logger.info(f"Logs directory: {logs_dir}")

    # Ensure models directory and lora subdirectory exist
    models_dir = ensure_models_dir()
    logger.info(f"Models directory: {models_dir}")

    # Initialize pipeline manager (but don't load pipeline yet)
    pipeline_manager = PipelineManager()
    logger.info("Pipeline manager initialized")

    # Pre-warm the default pipeline
    if PIPELINE is not None:
        asyncio.create_task(prewarm_pipeline(PIPELINE))

    webrtc_manager = WebRTCManager()
    logger.info("WebRTC manager initialized")

    yield

    # Shutdown
    if webrtc_manager:
        logger.info("Shutting down WebRTC manager...")
        await webrtc_manager.stop()
        logger.info("WebRTC manager shutdown complete")

    if pipeline_manager:
        logger.info("Shutting down pipeline manager...")
        pipeline_manager.unload_pipeline()
        logger.info("Pipeline manager shutdown complete")


def get_webrtc_manager() -> WebRTCManager:
    """Dependency to get WebRTC manager instance."""
    return webrtc_manager


def get_pipeline_manager() -> PipelineManager:
    """Dependency to get pipeline manager instance."""
    return pipeline_manager


def create_api_app() -> FastAPI:
    """Create and configure the API FastAPI application."""
    from importlib.metadata import version

    app = FastAPI(
        lifespan=lifespan,
        title="Scope API",
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
        return HealthResponse(status="healthy", timestamp=datetime.now().isoformat())

    @app.get("/")
    async def root():
        """Serve the frontend at the root URL."""
        frontend_dist = Path(__file__).parent.parent.parent.parent / "frontend" / "dist"

        # Only serve SPA if frontend dist exists (production mode)
        if not frontend_dist.exists():
            return {"message": "Scope API - Frontend not built"}

        # Serve the frontend index.html
        index_file = frontend_dist / "index.html"
        if index_file.exists():
            return FileResponse(index_file)

        return {"message": "Scope API - Frontend index.html not found"}

    @app.post("/api/v1/pipeline/load")
    async def load_pipeline(
        request: PipelineLoadRequest,
        pipeline_manager: PipelineManager = Depends(get_pipeline_manager),
    ):
        """Load a pipeline."""
        try:
            # Convert pydantic model to dict for pipeline manager
            load_params_dict = None
            if request.load_params:
                load_params_dict = request.load_params.model_dump()

            # Start loading in background without blocking
            asyncio.create_task(
                pipeline_manager.load_pipeline(request.pipeline_id, load_params_dict)
            )
            return {"message": "Pipeline loading initiated successfully"}
        except Exception as e:
            logger.error(f"Error loading pipeline: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/api/v1/pipeline/status", response_model=PipelineStatusResponse)
    async def get_pipeline_status(
        pipeline_manager: PipelineManager = Depends(get_pipeline_manager),
    ):
        """Get current pipeline status."""
        try:
            status_info = await pipeline_manager.get_status_info_async()
            return PipelineStatusResponse(**status_info)
        except Exception as e:
            logger.error(f"Error getting pipeline status: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/api/v1/webrtc/offer", response_model=WebRTCOfferResponse)
    async def handle_webrtc_offer(
        request: WebRTCOfferRequest,
        webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
        pipeline_manager: PipelineManager = Depends(get_pipeline_manager),
    ):
        """Handle WebRTC offer and return answer."""
        try:
            # Ensure pipeline is loaded before proceeding
            status_info = await pipeline_manager.get_status_info_async()
            if status_info["status"] != "loaded":
                raise HTTPException(
                    status_code=400,
                    detail="Pipeline not loaded. Please load pipeline first.",
                )

            return await webrtc_manager.handle_offer(request, pipeline_manager)

        except Exception as e:
            logger.error(f"Error handling WebRTC offer: {e}")
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

    @app.get("/api/v1/lora/list", response_model=LoRAFilesResponse)
    async def list_lora_files():
        """List available LoRA files in the models/lora directory and its subdirectories."""

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
            lora_dir = get_models_dir() / "lora"
            lora_files: list[LoRAFileInfo] = []

            if lora_dir.exists() and lora_dir.is_dir():
                for pattern in ("*.safetensors", "*.bin", "*.pt"):
                    for file_path in lora_dir.rglob(pattern):
                        if file_path.is_file():
                            lora_files.append(process_lora_file(file_path, lora_dir))

            lora_files.sort(key=lambda x: (x.folder or "", x.name))
            return LoRAFilesResponse(lora_files=lora_files)

        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(f"list_lora_files: Error listing LoRA files: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/api/v1/models/status")
    async def get_model_status(pipeline_id: str):
        """Check if models for a pipeline are downloaded."""
        try:
            downloaded = models_are_downloaded(pipeline_id)
            return ModelStatusResponse(downloaded=downloaded)
        except Exception as e:
            logger.error(f"Error checking model status: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/api/v1/models/download")
    async def download_pipeline_models(request: DownloadModelsRequest):
        """Download models for a specific pipeline."""
        try:
            if not request.pipeline_id:
                raise HTTPException(status_code=400, detail="pipeline_id is required")

            # Download in a background thread to avoid blocking
            import threading

            def download_in_background():
                download_models(pipeline_id=request.pipeline_id)

            thread = threading.Thread(target=download_in_background)
            thread.daemon = True
            thread.start()

            return {"message": f"Model download started for {request.pipeline_id}"}
        except Exception as e:
            logger.error(f"Error starting model download: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/api/v1/hardware/info", response_model=HardwareInfoResponse)
    async def get_hardware_info():
        """Get hardware information including available VRAM."""
        try:
            vram_gb = None

            if torch.cuda.is_available():
                # Get total VRAM from the first GPU (in bytes), convert to GB
                _, total_mem = torch.cuda.mem_get_info(0)
                vram_gb = total_mem / (1024**3)

            return HardwareInfoResponse(vram_gb=vram_gb)
        except Exception as e:
            logger.error(f"Error getting hardware info: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

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
            # with actively written log files
            log_content = log_file_path.read_text(encoding="utf-8")

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
            return FileResponse(file_path)

        # Fallback to index.html for SPA routing
        index_file = frontend_dist / "index.html"
        if index_file.exists():
            return FileResponse(index_file)

        raise HTTPException(status_code=404, detail="Frontend index.html not found")

    return app


def run_api_server(port: int = 8080, host: str = "0.0.0.0"):
    """Run the API server on the specified port."""
    api_app = create_api_app()

    # Configure static file serving
    frontend_dist = Path(__file__).parent.parent.parent.parent / "frontend" / "dist"
    if frontend_dist.exists():
        api_app.mount(
            "/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets"
        )
        logger.info(f"Serving static assets from {frontend_dist / 'assets'}")

    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(
        api_app,
        host=host,
        port=port,
        log_config=None,  # Use our logging config
    )
