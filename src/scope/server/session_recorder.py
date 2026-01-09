from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ControlEvent:
    """A single control event during recording."""

    # Primary timebase
    chunk_index: int

    # Secondary timebase: seconds since recording start
    wall_time: float

    # Prompt
    prompt: str | None = None
    prompt_weight: float = 1.0

    # Transition
    transition_steps: int | None = None
    transition_method: str | None = None  # "linear" or "slerp"

    # Cuts
    hard_cut: bool = False
    soft_cut_bias: float | None = None
    soft_cut_chunks: int | None = None
    soft_cut_restore_bias: float | None = None  # None means "was unset"
    soft_cut_restore_was_set: bool = False


@dataclass
class SessionRecording:
    """Container for a complete recording session."""

    events: list[ControlEvent] = field(default_factory=list)

    # Chunk timebase (primary)
    start_chunk: int = 0
    end_chunk: int | None = None

    # Wall-clock (secondary)
    start_wall_time: float | None = None
    end_wall_time: float | None = None

    # Pipeline info
    pipeline_id: str | None = None
    load_params: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.start_wall_time is not None and self.end_wall_time is None

    @property
    def duration_seconds(self) -> float:
        if self.start_wall_time is None:
            return 0.0
        end = self.end_wall_time if self.end_wall_time is not None else time.monotonic()
        return end - self.start_wall_time

    @property
    def duration_chunks(self) -> int:
        if self.end_chunk is None:
            return 0
        return self.end_chunk - self.start_chunk


class SessionRecorder:
    """Records control events during a streaming session.

    Intended usage: mutate only from the FrameProcessor worker thread.
    FastAPI threads should read status via get_status_snapshot().
    """

    def __init__(self) -> None:
        self._recording: SessionRecording | None = None
        self._last_prompt: str | None = None
        self._status_snapshot: dict[str, Any] = {"is_recording": False}

    @property
    def is_recording(self) -> bool:
        recording = self._recording
        return recording is not None and recording.is_active

    @property
    def last_prompt(self) -> str | None:
        return self._last_prompt

    def start(
        self,
        *,
        chunk_index: int,
        pipeline_id: str,
        load_params: dict[str, Any],
        baseline_prompt: str | None = None,
        baseline_weight: float = 1.0,
    ) -> None:
        if not pipeline_id:
            raise ValueError("pipeline_id is required for session recording")

        start_wall_time = time.monotonic()
        self._recording = SessionRecording(
            start_chunk=int(chunk_index),
            start_wall_time=start_wall_time,
            pipeline_id=pipeline_id,
            load_params=dict(load_params or {}),
        )

        self._last_prompt = None
        if baseline_prompt is not None:
            self._recording.events.append(
                ControlEvent(
                    chunk_index=int(chunk_index),
                    wall_time=0.0,
                    prompt=baseline_prompt,
                    prompt_weight=float(baseline_weight),
                )
            )
            self._last_prompt = baseline_prompt

        self._update_status_snapshot()

    def record_event(
        self,
        *,
        chunk_index: int,
        wall_time: float,
        prompt: str | None = None,
        prompt_weight: float = 1.0,
        transition_steps: int | None = None,
        transition_method: str | None = None,
        hard_cut: bool = False,
        soft_cut_bias: float | None = None,
        soft_cut_chunks: int | None = None,
        soft_cut_restore_bias: float | None = None,
        soft_cut_restore_was_set: bool = False,
    ) -> None:
        if not self.is_recording:
            return
        recording = self._recording
        if recording is None or recording.start_wall_time is None:
            return

        if prompt is not None:
            self._last_prompt = prompt

        effective_prompt = prompt
        if prompt is None and (hard_cut or soft_cut_bias is not None):
            effective_prompt = self._last_prompt

        relative_time = max(0.0, float(wall_time) - float(recording.start_wall_time))

        recording.events.append(
            ControlEvent(
                chunk_index=int(chunk_index),
                wall_time=relative_time,
                prompt=effective_prompt,
                prompt_weight=float(prompt_weight),
                transition_steps=transition_steps,
                transition_method=transition_method,
                hard_cut=bool(hard_cut),
                soft_cut_bias=soft_cut_bias,
                soft_cut_chunks=soft_cut_chunks,
                soft_cut_restore_bias=soft_cut_restore_bias,
                soft_cut_restore_was_set=bool(soft_cut_restore_was_set),
            )
        )
        self._update_status_snapshot()

    def stop(self, *, chunk_index: int) -> SessionRecording | None:
        if not self.is_recording:
            return None

        recording = self._recording
        if recording is None:
            return None

        recording.end_chunk = int(chunk_index)
        recording.end_wall_time = time.monotonic()

        self._recording = None
        self._last_prompt = None
        self._update_status_snapshot()
        return recording

    def _update_status_snapshot(self) -> None:
        recording = self._recording
        if recording is None:
            self._status_snapshot = {"is_recording": False}
            return

        self._status_snapshot = {
            "is_recording": recording.is_active,
            "start_chunk": recording.start_chunk,
            "duration_seconds": recording.duration_seconds,
            "events_count": len(recording.events),
        }

    def get_status_snapshot(self) -> dict[str, Any]:
        return self._status_snapshot

    def export_timeline(self, recording: SessionRecording) -> dict[str, Any]:
        segments: list[dict[str, Any]] = []

        for i, event in enumerate(recording.events):
            if event.prompt is None:
                continue

            end_chunk = recording.end_chunk if recording.end_chunk is not None else event.chunk_index
            end_time = recording.duration_seconds
            for next_event in recording.events[i + 1 :]:
                if next_event.prompt is None:
                    continue
                end_chunk = next_event.chunk_index
                end_time = next_event.wall_time
                break

            segment: dict[str, Any] = {
                "startTime": float(event.wall_time),
                "endTime": float(end_time),
                "startChunk": int(event.chunk_index - recording.start_chunk),
                "endChunk": int(end_chunk - recording.start_chunk),
                "prompts": [
                    {"text": event.prompt, "weight": float(event.prompt_weight)}
                ],
            }

            if event.transition_steps is not None and int(event.transition_steps) > 0:
                segment["transitionSteps"] = int(event.transition_steps)
            if event.transition_method:
                segment["temporalInterpolationMethod"] = event.transition_method

            if event.hard_cut:
                segment["initCache"] = True

            if event.soft_cut_bias is not None:
                segment["softCut"] = {
                    "bias": float(event.soft_cut_bias),
                    "chunks": int(event.soft_cut_chunks or 2),
                    "restoreBias": (
                        float(event.soft_cut_restore_bias)
                        if event.soft_cut_restore_bias is not None
                        else None
                    ),
                    "restoreWasSet": bool(event.soft_cut_restore_was_set),
                }

            segments.append(segment)

        load_params = recording.load_params or {}
        height = load_params.get("height")
        width = load_params.get("width")

        settings: dict[str, Any] = {"pipelineId": recording.pipeline_id}
        if height is not None and width is not None:
            settings["resolution"] = {"height": int(height), "width": int(width)}
        if "seed" in load_params:
            settings["seed"] = load_params.get("seed")
        if "kv_cache_attention_bias" in load_params:
            settings["kvCacheAttentionBias"] = load_params.get("kv_cache_attention_bias")

        # Export LoRA configuration for replay
        loras = load_params.get("loras")
        if loras and isinstance(loras, list):
            settings["loras"] = [
                {
                    "path": lora.get("path"),
                    "scale": float(lora.get("scale", 1.0)),
                    **({"mergeMode": lora.get("merge_mode")} if lora.get("merge_mode") else {}),
                }
                for lora in loras
                if isinstance(lora, dict) and lora.get("path")
            ]
        lora_merge_mode = load_params.get("lora_merge_mode")
        if lora_merge_mode:
            settings["loraMergeStrategy"] = lora_merge_mode

        return {
            "version": "1.1",
            "exportedAt": datetime.now(timezone.utc).isoformat(),
            "recording": {
                "durationSeconds": float(recording.duration_seconds),
                "durationChunks": int(recording.duration_chunks),
                "startChunk": int(recording.start_chunk),
                "endChunk": int(recording.end_chunk) if recording.end_chunk is not None else None,
            },
            "settings": settings,
            "prompts": segments,
        }

    def save(self, recording: SessionRecording, path: Path) -> Path:
        timeline = self.export_timeline(recording)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(timeline, indent=2))
        return path
