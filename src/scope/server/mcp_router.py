"""REST endpoints for headless session management.

Provides parameter control, frame capture, metrics, and session lifecycle
endpoints used by the MCP server and other programmatic clients.
"""

import io
import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, model_validator

if TYPE_CHECKING:
    from .cloud_connection import CloudConnectionManager
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


def _get_cloud_manager() -> "CloudConnectionManager":
    from .app import cloud_connection_manager

    return cloud_connection_manager


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
    that specific sink node in a multi-sink graph.
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

    @model_validator(mode="after")
    def _require_pipeline_or_graph(self) -> "StartStreamRequest":
        if self.pipeline_id is None and self.graph is None:
            raise ValueError("Either pipeline_id or graph must be provided")
        return self


def _wire_cloud_outputs(cloud_manager, frame_processor, graph_config) -> None:
    """Wire cloud WebRTC extra output handlers to FrameProcessor queues.

    In headless cloud mode there is no CloudTrack to do this wiring, so we
    do it here after start_webrtc + FrameProcessor.start().

    The cloud sends multiple video tracks:
    - Track 0: primary sink (goes to cloud_manager's main callback → FP._cloud_output_queue)
    - Track 1..N: extra sinks → FP.sink_manager._sink_queues_by_node
    - Track N+1..M: record nodes → FP.recording.record_queues
    """
    import queue

    from av import VideoFrame

    webrtc_client = cloud_manager._webrtc_client
    if webrtc_client is None:
        return

    sink_ids = graph_config.get_sink_node_ids()
    record_ids = graph_config.get_record_node_ids()

    # Create per-sink queues in the SinkManager so HeadlessSession
    # can read from them via get_from_sink()
    sink_queues = frame_processor.sink_manager._sink_queues_by_node
    for sid in sink_ids:
        if sid not in sink_queues:
            sink_queues[sid] = queue.Queue(maxsize=2)

    # The first sink is the primary (track 0) — its frames come through
    # the main cloud callback → _cloud_output_queue. We need to also
    # put them into the per-sink queue so HeadlessSession._consume_frames
    # can find them.
    if sink_ids:
        primary_sink_id = sink_ids[0]
        primary_q = sink_queues[primary_sink_id]

        import torch

        cloud_relay = frame_processor._cloud_relay
        original_callback = cloud_relay.on_frame_from_cloud

        def _primary_sink_callback(frame: VideoFrame) -> None:
            original_callback(frame)
            try:
                frame_np = frame.to_ndarray(format="rgb24")
                t = torch.as_tensor(frame_np, dtype=torch.uint8).unsqueeze(0)
                try:
                    primary_q.put_nowait(t)
                except queue.Full:
                    try:
                        primary_q.get_nowait()
                        primary_q.put_nowait(t)
                    except queue.Empty:
                        pass
            except Exception as e:
                logger.error(f"Error in primary sink callback: {e}")

        cloud_manager.remove_frame_callback(original_callback)
        cloud_manager.add_frame_callback(_primary_sink_callback)
        # Store ref so stop() can deregister
        cloud_relay.on_frame_from_cloud = _primary_sink_callback

    # Wire extra sink output handlers (track index 1+)
    for i, sid in enumerate(sink_ids[1:], start=1):
        if i >= len(webrtc_client.output_handlers):
            continue
        handler = webrtc_client.output_handlers[i]
        sink_q = sink_queues[sid]

        def _make_sink_cb(q, sink_id):
            def cb(frame: VideoFrame) -> None:
                try:
                    frame_np = frame.to_ndarray(format="rgb24")
                    t = torch.as_tensor(frame_np, dtype=torch.uint8).unsqueeze(0)
                    try:
                        q.put_nowait(t)
                    except queue.Full:
                        try:
                            q.get_nowait()
                            q.put_nowait(t)
                        except queue.Empty:
                            pass
                except Exception as e:
                    logger.error(f"Error in sink {sink_id} callback: {e}")

            return cb

        handler.add_callback(_make_sink_cb(sink_q, sid))
        logger.info(f"Wired cloud output track {i} to sink {sid}")

    # Wire record node output handlers
    num_extra_sinks = max(0, len(sink_ids) - 1)
    for i, rec_id in enumerate(record_ids):
        handler_index = num_extra_sinks + 1 + i
        if handler_index >= len(webrtc_client.output_handlers):
            continue
        handler = webrtc_client.output_handlers[handler_index]

        def _make_rec_cb(fp, rid):
            def cb(frame: VideoFrame) -> None:
                fp.sink_manager.put_to_record(rid, frame)

            return cb

        handler.add_callback(_make_rec_cb(frame_processor, rec_id))
        logger.info(f"Wired cloud output track {handler_index} to record {rec_id}")


@router.post("/session/start")
async def start_stream(
    request: StartStreamRequest,
    webrtc_manager: "WebRTCManager" = Depends(_get_webrtc_manager),
    pipeline_manager: "PipelineManager" = Depends(_get_pipeline_manager),
    cloud_manager: "CloudConnectionManager" = Depends(_get_cloud_manager),
):
    """Start a headless pipeline session without WebRTC.

    Creates a FrameProcessor directly and begins generating frames.
    Use capture_frame to see output, update_parameters to control it,
    and POST /api/v1/session/stop to tear it down.

    Supports two modes:
    - Simple: provide pipeline_id for a single-pipeline session
    - Graph: provide a graph dict with nodes/edges for multi-source/multi-sink

    When cloud is connected, runs in cloud relay mode (frames sent to cloud
    for processing).
    """
    from scope.core.pipelines.registry import PipelineRegistry

    from .frame_processor import FrameProcessor
    from .headless import HeadlessSession

    # Determine if we should use cloud mode
    use_cloud = cloud_manager is not None and cloud_manager.is_connected

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
        if not pipeline_ids:
            raise HTTPException(
                status_code=400,
                detail="Graph must contain at least one pipeline node",
            )

        pipeline_tuples = [
            (node.id, node.pipeline_id, None)
            for node in graph_config.nodes
            if node.type == "pipeline" and node.pipeline_id
        ]
        pipeline_id_list = [t[1] for t in pipeline_tuples]

        if not use_cloud:
            # Local mode: load pipelines locally
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

    try:
        if use_cloud:
            # Cloud mode: start WebRTC relay to cloud, then create
            # FrameProcessor in cloud mode (no local pipeline_manager)
            await cloud_manager.start_webrtc(initial_params)
            frame_processor = FrameProcessor(
                pipeline_manager=None,
                initial_parameters=initial_params,
                cloud_manager=cloud_manager,
            )
        else:
            frame_processor = FrameProcessor(
                pipeline_manager=pipeline_manager,
                initial_parameters=initial_params,
            )
        frame_processor.start()

        # In cloud graph mode, wire cloud extra output handlers to
        # FrameProcessor's sink/record queues so that HeadlessSession
        # can capture per-sink frames and per-record-node recordings.
        if use_cloud and request.graph is not None:
            _wire_cloud_outputs(cloud_manager, frame_processor, graph_config)

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
        mode = "cloud" if use_cloud else "local"
        logger.info(f"Started headless session ({mode}) with pipeline(s) {pipeline_id}")
        response: dict = {
            "status": "ok",
            "input_mode": request.input_mode,
            "cloud_mode": use_cloud,
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
