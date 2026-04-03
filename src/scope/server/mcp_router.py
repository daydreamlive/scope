"""REST endpoints for headless session management.

Provides parameter control, frame capture, metrics, and session lifecycle
endpoints used by the MCP server and other programmatic clients.
"""

import io
import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

if TYPE_CHECKING:
    from .pipeline_manager import PipelineManager
    from .webrtc import WebRTCManager

from .schema import Parameters

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["session"])


# ---------------------------------------------------------------------------
# Dependencies (deferred imports to avoid circular import with app.py)
# ---------------------------------------------------------------------------


def _get_webrtc_manager() -> "WebRTCManager":
    from .app import webrtc_manager

    return webrtc_manager


def _get_pipeline_manager() -> "PipelineManager":
    from .app import pipeline_manager

    return pipeline_manager


# ---------------------------------------------------------------------------
# Parameter Control
# ---------------------------------------------------------------------------


@router.post("/session/parameters")
async def update_session_parameters(
    parameters: Parameters,
    webrtc_manager: "WebRTCManager" = Depends(_get_webrtc_manager),
):
    """Update runtime parameters for all active WebRTC sessions.

    Applies parameter changes to the pipeline (same path as the WebRTC data
    channel) and notifies connected frontends so their UI stays in sync.
    """
    params_dict = parameters.model_dump(exclude_none=True)
    if not params_dict:
        raise HTTPException(status_code=400, detail="No parameters provided")

    # Copy before broadcast_parameter_update which mutates params_dict
    # (frame_processor.update_parameters pops node_id).
    notification_params = dict(params_dict)

    webrtc_manager.broadcast_parameter_update(params_dict)
    webrtc_manager.broadcast_notification(
        {"type": "parameters_updated", "parameters": notification_params}
    )

    return {"status": "ok", "applied_parameters": notification_params}


@router.get("/session/parameters")
async def get_session_parameters(
    webrtc_manager: "WebRTCManager" = Depends(_get_webrtc_manager),
):
    """Get the current runtime parameters from the active session.

    Returns the parameter state from the session's frame processor.
    """
    result = webrtc_manager.get_frame_processor()
    params = result[1].parameters if result else {}
    return {"parameters": params}


# ---------------------------------------------------------------------------
# Frame Capture
# ---------------------------------------------------------------------------


@router.get("/session/frame")
async def capture_frame(
    webrtc_manager: "WebRTCManager" = Depends(_get_webrtc_manager),
    quality: int = Query(default=85, ge=1, le=100),
):
    """Capture the current pipeline output frame as a JPEG image.

    Returns the most recent rendered frame from the active session
    (WebRTC or headless).
    """
    frame = webrtc_manager.get_last_frame()
    if frame is None:
        raise HTTPException(
            status_code=404,
            detail="No frame available (no active session or pipeline not running)",
        )

    try:
        from PIL import Image

        frame_np = frame.to_ndarray(format="rgb24")
        img = Image.fromarray(frame_np)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Error capturing frame: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ---------------------------------------------------------------------------
# Session Metrics
# ---------------------------------------------------------------------------


@router.get("/session/metrics")
async def get_session_metrics(
    webrtc_manager: "WebRTCManager" = Depends(_get_webrtc_manager),
):
    """Get performance metrics from the active session.

    Returns frame stats from the session's frame processor and GPU VRAM
    usage when CUDA is available.
    """
    session_stats = {}
    result = webrtc_manager.get_frame_processor()
    if result:
        sid, fp, is_headless = result
        if not (is_headless and not fp.running):
            stats = fp.get_frame_stats()
            if is_headless:
                stats["headless"] = True
            session_stats[sid] = stats

    gpu_info = {}
    try:
        import torch

        if torch.cuda.is_available():
            gpu_info = {
                "vram_allocated_mb": round(
                    torch.cuda.memory_allocated() / (1024 * 1024), 1
                ),
                "vram_reserved_mb": round(
                    torch.cuda.memory_reserved() / (1024 * 1024), 1
                ),
                "vram_total_mb": round(
                    torch.cuda.get_device_properties(0).total_mem / (1024 * 1024), 1
                ),
            }
    except Exception:
        pass

    return {
        "sessions": session_stats,
        "gpu": gpu_info,
    }


# ---------------------------------------------------------------------------
# Session Lifecycle
# ---------------------------------------------------------------------------


class StartStreamRequest(BaseModel):
    pipeline_id: str
    input_mode: str = "text"
    prompts: list[dict] | None = None
    input_source: dict | None = None


@router.post("/session/start")
async def start_stream(
    request: StartStreamRequest,
    webrtc_manager: "WebRTCManager" = Depends(_get_webrtc_manager),
    pipeline_manager: "PipelineManager" = Depends(_get_pipeline_manager),
):
    """Start a headless pipeline session without WebRTC.

    Creates a FrameProcessor directly and begins generating frames.
    Use capture_frame to see output, update_parameters to control it,
    and POST /api/v1/session/stop to tear it down.
    """
    from .frame_processor import FrameProcessor
    from .headless import HeadlessSession

    # Build initial parameters
    initial_params: dict = {
        "pipeline_ids": [request.pipeline_id],
        "input_mode": request.input_mode,
    }
    if request.prompts is not None:
        initial_params["prompts"] = request.prompts
    if request.input_source is not None:
        initial_params["input_source"] = request.input_source

    try:
        frame_processor = FrameProcessor(
            pipeline_manager=pipeline_manager,
            initial_parameters=initial_params,
        )
        frame_processor.start()

        if not frame_processor.running:
            raise HTTPException(
                status_code=500,
                detail="FrameProcessor failed to start (check logs for details)",
            )

        session = HeadlessSession(
            frame_processor=frame_processor,
        )
        session.start_frame_consumer()
        webrtc_manager.add_headless_session(session)

        logger.info(f"Started headless session with pipeline {request.pipeline_id}")
        return {
            "status": "ok",
            "pipeline_id": request.pipeline_id,
            "input_mode": request.input_mode,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting headless session: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/session/stop")
async def stop_stream(
    webrtc_manager: "WebRTCManager" = Depends(_get_webrtc_manager),
):
    """Stop the active headless pipeline session."""
    if not webrtc_manager.headless_session:
        raise HTTPException(status_code=404, detail="No active headless session")
    try:
        await webrtc_manager.remove_headless_session()
        return {"status": "ok", "message": "Headless session stopped"}
    except Exception as e:
        logger.error(f"Error stopping headless session: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
