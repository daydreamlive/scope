"""REST endpoints for headless session management.

Provides parameter control, frame capture, metrics, and session lifecycle
endpoints used by the MCP server and other programmatic clients.
"""

import io
import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, model_validator

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
    sink_node_id: str | None = Query(default=None),
):
    """Capture the current pipeline output frame as a JPEG image.

    Returns the most recent rendered frame from the active session
    (WebRTC or headless). When sink_node_id is provided, captures from
    that specific sink node in a multi-sink graph. In multi-sink headless
    sessions, omitting sink_node_id returns the most recently consumed frame
    from any sink, so callers that need stable per-sink capture should pass
    sink_node_id explicitly.
    """
    frame = webrtc_manager.get_last_frame(sink_node_id=sink_node_id)
    if frame is None:
        detail = "No frame available"
        if sink_node_id:
            detail += f" for sink node '{sink_node_id}'"
        detail += " (no active session or pipeline not running)"
        raise HTTPException(status_code=404, detail=detail)

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


@router.get("/session/output.ts")
async def stream_headless_output_ts(
    webrtc_manager: "WebRTCManager" = Depends(_get_webrtc_manager),
):
    """Stream the active headless session as MPEG-TS."""
    session = webrtc_manager.headless_session
    if session is None or not session.frame_processor.running:
        raise HTTPException(
            status_code=404,
            detail="No active headless session",
        )

    streamer = session.create_ts_streamer()

    async def stream_generator():
        try:
            async for chunk in streamer.iter_bytes():
                yield chunk
        finally:
            session.remove_media_sink(streamer)
            streamer.close()

    return StreamingResponse(
        stream_generator(),
        media_type="video/mp2t",
        headers={
            "Cache-Control": "no-store",
            "Connection": "keep-alive",
        },
    )


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
    pipeline_id: str | None = None
    input_mode: str = "text"
    prompts: list[dict] | None = None
    input_source: dict | None = None
    graph: dict | None = None
    parameters: dict[str, Any] | None = None
    node_parameters: dict[str, dict[str, Any]] | None = None

    @model_validator(mode="after")
    def _require_pipeline_or_graph(self) -> "StartStreamRequest":
        if self.pipeline_id is None and self.graph is None:
            raise ValueError("Either pipeline_id or graph must be provided")
        return self


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

    Supports two graph configurations:
    - Simple: provide pipeline_id for a single-pipeline session
    - Graph: provide a graph dict with nodes/edges for multi-source/multi-sink
    """
    from scope.core.pipelines.registry import PipelineRegistry

    from .frame_processor import FrameProcessor
    from .headless import HeadlessSession

    if request.graph is not None:
        # Graph mode: extract pipeline_ids from graph nodes
        from .graph_schema import GraphConfig

        graph_config = GraphConfig.model_validate(request.graph)
        errors = graph_config.validate_structure()
        if errors:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid graph: {'; '.join(errors)}",
            )
        pipeline_ids = graph_config.get_pipeline_node_ids()
        backend_node_ids = graph_config.get_backend_node_ids()
        if not pipeline_ids and not backend_node_ids:
            raise HTTPException(
                status_code=400,
                detail="Graph must contain at least one pipeline or custom node",
            )

        pipeline_tuples = [
            (node.id, node.pipeline_id, None)
            for node in graph_config.nodes
            if node.type == "pipeline" and node.pipeline_id
        ]
        pipeline_id_list = [t[1] for t in pipeline_tuples]

        unknown = [p for p in pipeline_id_list if not PipelineRegistry.is_registered(p)]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown pipeline_id(s): {unknown}. "
                    f"Known: {PipelineRegistry.list_pipelines()}"
                ),
            )

        if pipeline_tuples:
            # Skip load for node-only graphs.
            await pipeline_manager.load_pipelines(pipeline_tuples)

        initial_params: dict = {
            "pipeline_ids": pipeline_id_list,
            "input_mode": request.input_mode,
            "graph": request.graph,
        }
    else:
        # Simple single-pipeline mode (pipeline_id guaranteed by model_validator)
        assert request.pipeline_id is not None
        pipeline_id_list = [request.pipeline_id]
        initial_params = {
            "pipeline_ids": pipeline_id_list,
            "input_mode": request.input_mode,
        }

    if request.prompts is not None:
        initial_params["prompts"] = request.prompts
    if request.input_source is not None:
        initial_params["input_source"] = request.input_source
    # Flat pipeline parameters (e.g. width/height, __prompt, noise_scale) merged
    # into initial_parameters so they reach the pipeline on the first call,
    # matching how the WebRTC frontend delivers them.
    if request.parameters:
        for key, value in request.parameters.items():
            if key not in initial_params:
                initial_params[key] = value

    try:
        frame_processor = FrameProcessor(
            pipeline_manager=pipeline_manager,
            initial_parameters=initial_params,
        )
        frame_processor.start()

        # Per-node parameters target a specific graph node (e.g. longlive vs
        # rife), distinct from the broadcast `parameters` above.
        # FrameProcessor.update_parameters routes by node_id (and buffers in
        # _pending_node_params if the graph isn't wired yet).
        if request.node_parameters:
            for node_id, node_params in request.node_parameters.items():
                payload = {"node_id": node_id, **node_params}
                frame_processor.update_parameters(payload)

        if not frame_processor.running:
            raise HTTPException(
                status_code=500,
                detail="FrameProcessor failed to start (check logs for details)",
            )

        session = HeadlessSession(
            frame_processor=frame_processor,
            expect_audio=PipelineRegistry.chain_produces_audio(pipeline_id_list),
        )
        session.start_frame_consumer()
        webrtc_manager.add_headless_session(session)

        pipeline_id = request.pipeline_id or ",".join(pipeline_id_list)
        logger.info(f"Started headless session (local) with pipeline(s) {pipeline_id}")
        response: dict = {
            "status": "ok",
            "input_mode": request.input_mode,
            "cloud_mode": False,
        }
        if request.graph is not None:
            response["graph"] = True
            response["pipeline_ids"] = pipeline_id_list
            sink_ids = graph_config.get_sink_node_ids()
            response["sink_node_ids"] = sink_ids
            source_ids = graph_config.get_source_node_ids()
            response["source_node_ids"] = source_ids
        else:
            response["pipeline_id"] = request.pipeline_id
        return response
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
