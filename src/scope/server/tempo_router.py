"""REST endpoints for tempo synchronization.

Provides tempo sync control endpoints for enabling/disabling tempo sources,
querying beat state, and adjusting BPM.
"""

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException

if TYPE_CHECKING:
    from .tempo_sync import TempoSync

from .schema import (
    TempoEnableRequest,
    TempoSetTempoRequest,
    TempoSourcesResponse,
    TempoStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tempo", tags=["tempo"])


# ---------------------------------------------------------------------------
# Dependencies (deferred imports to avoid circular import with app.py)
# ---------------------------------------------------------------------------


def _get_tempo_sync() -> "TempoSync | None":
    from .app import tempo_sync

    return tempo_sync


@router.get("/status", response_model=TempoStatusResponse)
async def get_tempo_status():
    """Get current tempo sync status including beat state."""
    tempo_sync = _get_tempo_sync()
    if tempo_sync is None:
        return TempoStatusResponse(enabled=False, beats_per_bar=4)
    status = tempo_sync.get_status()
    return TempoStatusResponse(**status)


@router.post("/enable", response_model=TempoStatusResponse)
async def enable_tempo(request: TempoEnableRequest):
    """Enable tempo synchronization with the specified source."""
    tempo_sync = _get_tempo_sync()
    if tempo_sync is None:
        raise HTTPException(status_code=500, detail="Tempo sync not initialized")

    try:
        await tempo_sync.enable(
            source_type=request.source,
            bpm=request.bpm,
            midi_device=request.midi_device,
            beats_per_bar=request.beats_per_bar,
        )
    except ImportError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Required dependency not installed: {e}",
        ) from e
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    status = tempo_sync.get_status()
    return TempoStatusResponse(**status)


@router.post("/disable", response_model=TempoStatusResponse)
async def disable_tempo():
    """Disable tempo synchronization."""
    tempo_sync = _get_tempo_sync()
    if tempo_sync is None:
        raise HTTPException(status_code=500, detail="Tempo sync not initialized")

    await tempo_sync.disable()
    status = tempo_sync.get_status()
    return TempoStatusResponse(**status)


@router.post("/set_tempo", response_model=TempoStatusResponse)
async def set_tempo(request: TempoSetTempoRequest):
    """Set the session tempo (BPM). Only supported by some sources (e.g. Link)."""
    tempo_sync = _get_tempo_sync()
    if tempo_sync is None:
        raise HTTPException(status_code=500, detail="Tempo sync not initialized")
    if not tempo_sync.enabled:
        raise HTTPException(status_code=400, detail="Tempo sync is not enabled")
    try:
        tempo_sync.set_tempo(request.bpm)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    # get_status reads from the poll-loop cache which may be stale.
    # Patch the BPM to reflect what we just set so the response is correct.
    status = tempo_sync.get_status()
    if status.get("beat_state"):
        status["beat_state"]["bpm"] = request.bpm
    return TempoStatusResponse(**status)


@router.get("/sources", response_model=TempoSourcesResponse)
async def get_tempo_sources():
    """Get available tempo sources and their capabilities."""
    tempo_sync = _get_tempo_sync()
    if tempo_sync is None:
        return TempoSourcesResponse(sources={})
    return TempoSourcesResponse(sources=tempo_sync.get_available_sources())
