"""Modulation Oscilloscope pipeline.

Renders real-time oscilloscope traces for modulatable parameters so you
can see the exact wave shapes being applied by the ModulationEngine.
Pure PyTorch, no model dependencies, runs at high FPS.

Layout:
  ┌──────────────────────────────────────┐
  │  BPM display    beat-phase bar       │
  │  ┌─ noise_scale ──────── 0.45 ────┐  │
  │  │  ~~~~~~~~~~~~ trace ~~~~~~~~~~  │  │
  │  └────────────────────────────────┘  │
  │  ┌─ vace_context_scale ── 1.20 ──┐  │
  │  │  ~~~~~~~~~~~~ trace ~~~~~~~~~~  │  │
  │  └────────────────────────────────┘  │
  │  ┌─ kv_cache_bias ─────── 0.30 ──┐  │
  │  │  ~~~~~~~~~~~~ trace ~~~~~~~~~~  │  │
  │  └────────────────────────────────┘  │
  │  ▮▮▮   ▮▮▮   ▮▮▮  (level meters)    │
  │  [●][○][○][○]  (beat dots)          │
  └──────────────────────────────────────┘
"""

from collections import deque
from typing import TYPE_CHECKING

import torch

from ..interface import Pipeline
from .schema import ModScopeConfig

if TYPE_CHECKING:
    from ..base_schema import BasePipelineConfig

_DIGIT_PATTERNS: dict[str, list[int]] = {
    "0": [0b111, 0b101, 0b101, 0b101, 0b111],
    "1": [0b010, 0b110, 0b010, 0b010, 0b111],
    "2": [0b111, 0b001, 0b111, 0b100, 0b111],
    "3": [0b111, 0b001, 0b111, 0b001, 0b111],
    "4": [0b101, 0b101, 0b111, 0b001, 0b001],
    "5": [0b111, 0b100, 0b111, 0b001, 0b111],
    "6": [0b111, 0b100, 0b111, 0b101, 0b111],
    "7": [0b111, 0b001, 0b001, 0b001, 0b001],
    "8": [0b111, 0b101, 0b111, 0b101, 0b111],
    "9": [0b111, 0b101, 0b111, 0b001, 0b111],
    ".": [0b000, 0b000, 0b000, 0b000, 0b010],
    "-": [0b000, 0b000, 0b111, 0b000, 0b000],
    " ": [0b000, 0b000, 0b000, 0b000, 0b000],
}

# Character patterns for A-Z (3x5 bitmap, uppercase only, subset needed)
_ALPHA_PATTERNS: dict[str, list[int]] = {
    "A": [0b010, 0b101, 0b111, 0b101, 0b101],
    "B": [0b110, 0b101, 0b110, 0b101, 0b110],
    "C": [0b011, 0b100, 0b100, 0b100, 0b011],
    "D": [0b110, 0b101, 0b101, 0b101, 0b110],
    "E": [0b111, 0b100, 0b110, 0b100, 0b111],
    "G": [0b011, 0b100, 0b101, 0b101, 0b011],
    "I": [0b111, 0b010, 0b010, 0b010, 0b111],
    "K": [0b101, 0b110, 0b100, 0b110, 0b101],
    "L": [0b100, 0b100, 0b100, 0b100, 0b111],
    "M": [0b101, 0b111, 0b111, 0b101, 0b101],
    "N": [0b101, 0b111, 0b111, 0b101, 0b101],
    "O": [0b010, 0b101, 0b101, 0b101, 0b010],
    "P": [0b110, 0b101, 0b110, 0b100, 0b100],
    "R": [0b110, 0b101, 0b110, 0b101, 0b101],
    "S": [0b011, 0b100, 0b010, 0b001, 0b110],
    "T": [0b111, 0b010, 0b010, 0b010, 0b010],
    "U": [0b101, 0b101, 0b101, 0b101, 0b010],
    "V": [0b101, 0b101, 0b101, 0b101, 0b010],
    "X": [0b101, 0b101, 0b010, 0b101, 0b101],
    "_": [0b000, 0b000, 0b000, 0b000, 0b111],
}

ALL_PATTERNS = {**_DIGIT_PATTERNS, **_ALPHA_PATTERNS}

TRACE_LENGTH = 192
BEATS_PER_BAR = 4


TRACKED_PARAMS = [
    ("noise_scale", 0.0, 1.0, (0.2, 0.85, 1.0)),
    ("vace_context_scale", 0.0, 2.0, (1.0, 0.5, 0.2)),
    ("kv_cache_attention_bias", -1.0, 1.0, (0.3, 1.0, 0.5)),
]


class ModScopePipeline(Pipeline):
    """Oscilloscope visualizer for beat-synced parameter modulation."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return ModScopeConfig

    def __init__(
        self,
        height: int = 512,
        width: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        self.height = height
        self.width = width
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype
        self._output = torch.zeros(
            (1, height, width, 3), dtype=torch.float32, device=self.device
        )
        self._traces: dict[str, deque] = {
            name: deque(maxlen=TRACE_LENGTH) for name, *_ in TRACKED_PARAMS
        }
        self._frame_count = 0

    def __call__(self, **kwargs) -> dict:
        beat_phase = float(kwargs.get("beat_phase", 0.0))
        bar_position = float(kwargs.get("bar_position", 0.0))
        bpm = float(kwargs.get("bpm", 0.0))
        is_playing = bool(kwargs.get("is_playing", False))

        out = self._output
        H, W = self.height, self.width

        # Record current parameter values into rolling traces
        for name, *_ in TRACKED_PARAMS:
            val = float(kwargs.get(name, 0.0))
            self._traces[name].append(val)

        self._frame_count += 1

        # --- Background ---
        bg_pulse = 0.04 + 0.02 * (1.0 - beat_phase) if is_playing else 0.04
        out[0, :, :, 0] = bg_pulse * 0.6
        out[0, :, :, 1] = bg_pulse * 0.8
        out[0, :, :, 2] = bg_pulse * 1.0

        # --- Beat phase bar (top) ---
        bar_height = max(4, H // 64)
        bar_width = int(beat_phase * W) if is_playing else 0
        # Bar background
        out[0, 0:bar_height, :, :] = 0.06
        # Active portion
        if bar_width > 0:
            intensity = 0.3 + 0.4 * (1.0 - beat_phase)
            out[0, 0:bar_height, 0:bar_width, 0] = intensity * 0.4
            out[0, 0:bar_height, 0:bar_width, 1] = intensity * 0.8
            out[0, 0:bar_height, 0:bar_width, 2] = intensity

        # --- Layout: divide remaining space ---
        top_margin = bar_height + max(4, H // 32)
        bottom_area = max(60, H // 6)
        trace_area_top = top_margin
        trace_area_bottom = H - bottom_area
        trace_area_height = trace_area_bottom - trace_area_top
        num_traces = len(TRACKED_PARAMS)
        trace_slot_height = trace_area_height // num_traces
        trace_margin = max(2, trace_slot_height // 10)
        trace_graph_height = trace_slot_height - trace_margin * 2 - max(8, H // 40)
        trace_x_start = max(8, W // 32)
        trace_x_end = W - trace_x_start

        scale = max(1, min(H // 256, W // 256, 3))

        # --- BPM display (top-left) ---
        if bpm > 0:
            bpm_str = f"{bpm:.1f}"
            self._draw_text(
                out,
                bpm_str,
                trace_x_start,
                top_margin - max(4, H // 40),
                scale=scale + 1,
                color=(0.6, 0.6, 0.6),
            )

        # --- Draw oscilloscope traces ---
        for idx, (name, val_min, val_max, color) in enumerate(TRACKED_PARAMS):
            slot_y = trace_area_top + idx * trace_slot_height
            label_y = slot_y + trace_margin
            graph_y = label_y + max(8, H // 40)
            graph_h = trace_graph_height
            current_val = float(kwargs.get(name, 0.0))

            # Label and value
            short_name = name.upper().replace("_", " ")[:20]
            val_str = f"{current_val:.3f}"
            self._draw_text(
                out, short_name, trace_x_start, label_y, scale=scale, color=color
            )
            self._draw_text(
                out,
                val_str,
                trace_x_end,
                label_y,
                scale=scale,
                align_right=True,
                color=(0.5, 0.5, 0.5),
            )

            # Graph border
            self._draw_rect_outline(
                out,
                trace_x_start,
                graph_y,
                trace_x_end,
                graph_y + graph_h,
                color=(0.08, 0.08, 0.12),
            )

            # Graph background
            out[
                0, graph_y + 1 : graph_y + graph_h, trace_x_start + 1 : trace_x_end, :
            ] = 0.02

            # Zero/center line
            center_y = graph_y + graph_h // 2
            out[0, center_y, trace_x_start + 1 : trace_x_end, :] = 0.06

            # Draw the trace
            trace = self._traces[name]
            if len(trace) > 1:
                graph_w = trace_x_end - trace_x_start - 2
                n_points = min(len(trace), graph_w)
                start_idx = max(0, len(trace) - n_points)

                for i in range(n_points):
                    val = trace[start_idx + i]
                    # Normalize to [0, 1] within the display range
                    val_range = val_max - val_min
                    if val_range > 0:
                        norm = (val - val_min) / val_range
                    else:
                        norm = 0.5
                    norm = max(0.0, min(1.0, norm))

                    # Map to pixel Y (inverted: 0=top=max, 1=bottom=min)
                    py = graph_y + 1 + int((1.0 - norm) * (graph_h - 2))
                    px = trace_x_start + 1 + i

                    if 0 <= py < H and 0 <= px < W:
                        # Main trace pixel + glow
                        out[0, py, px, 0] = color[0]
                        out[0, py, px, 1] = color[1]
                        out[0, py, px, 2] = color[2]
                        # Vertical glow (1px above and below)
                        for dy in [-1, 1]:
                            gy = py + dy
                            if graph_y < gy < graph_y + graph_h:
                                out[0, gy, px, 0] = max(
                                    out[0, gy, px, 0].item(), color[0] * 0.3
                                )
                                out[0, gy, px, 1] = max(
                                    out[0, gy, px, 1].item(), color[1] * 0.3
                                )
                                out[0, gy, px, 2] = max(
                                    out[0, gy, px, 2].item(), color[2] * 0.3
                                )

                # Bright dot at the current value (rightmost point)
                if n_points > 0:
                    last_val = trace[-1]
                    val_range = val_max - val_min
                    norm = (last_val - val_min) / val_range if val_range > 0 else 0.5
                    norm = max(0.0, min(1.0, norm))
                    dot_y = graph_y + 1 + int((1.0 - norm) * (graph_h - 2))
                    dot_x = trace_x_start + 1 + n_points - 1
                    dot_r = max(2, scale)
                    for dy in range(-dot_r, dot_r + 1):
                        for dx in range(-dot_r, dot_r + 1):
                            if dy * dy + dx * dx <= dot_r * dot_r:
                                py2 = dot_y + dy
                                px2 = dot_x + dx
                                if 0 <= py2 < H and 0 <= px2 < W:
                                    out[0, py2, px2, 0] = min(1.0, color[0] + 0.3)
                                    out[0, py2, px2, 1] = min(1.0, color[1] + 0.3)
                                    out[0, py2, px2, 2] = min(1.0, color[2] + 0.3)

        # --- Level meters (bottom) ---
        meter_bottom = H - max(10, H // 32)
        meter_top = H - bottom_area + max(8, H // 32)
        meter_h = meter_bottom - meter_top
        meter_w = max(16, W // 16)
        meter_gap = max(8, W // 20)
        total_meters_w = num_traces * meter_w + (num_traces - 1) * meter_gap
        meter_start_x = (W - total_meters_w) // 2

        for idx, (name, val_min, val_max, color) in enumerate(TRACKED_PARAMS):
            current_val = float(kwargs.get(name, 0.0))
            val_range = val_max - val_min
            norm = (current_val - val_min) / val_range if val_range > 0 else 0.5
            norm = max(0.0, min(1.0, norm))

            mx = meter_start_x + idx * (meter_w + meter_gap)

            # Meter background
            out[0, meter_top:meter_bottom, mx : mx + meter_w, :] = 0.04

            # Filled portion (from bottom up)
            fill_h = max(1, int(norm * meter_h))
            fill_top = meter_bottom - fill_h
            # Gradient: brighter toward top
            for row in range(fill_top, meter_bottom):
                row_norm = 1.0 - (row - fill_top) / max(1, fill_h)
                brightness = 0.3 + 0.7 * row_norm
                out[0, row, mx + 1 : mx + meter_w - 1, 0] = color[0] * brightness
                out[0, row, mx + 1 : mx + meter_w - 1, 1] = color[1] * brightness
                out[0, row, mx + 1 : mx + meter_w - 1, 2] = color[2] * brightness

            # Meter border
            self._draw_rect_outline(
                out,
                mx,
                meter_top,
                mx + meter_w,
                meter_bottom,
                color=(0.12, 0.12, 0.18),
            )

        # --- Beat dots (bottom-right) ---
        dot_r = max(3, H // 100)
        dot_spacing = dot_r * 4
        current_beat = int(bar_position) % BEATS_PER_BAR if is_playing else -1
        dot_area_w = BEATS_PER_BAR * dot_spacing
        dot_start_x = W - trace_x_start - dot_area_w
        dot_y = meter_top + (meter_bottom - meter_top) // 2

        for i in range(BEATS_PER_BAR):
            cx = dot_start_x + i * dot_spacing + dot_r
            is_active = i == current_beat

            for dy in range(-dot_r, dot_r + 1):
                for dx in range(-dot_r, dot_r + 1):
                    if dy * dy + dx * dx <= dot_r * dot_r:
                        py = dot_y + dy
                        px = cx + dx
                        if 0 <= py < H and 0 <= px < W:
                            if is_active:
                                brightness = 0.6 + 0.4 * (1.0 - beat_phase)
                                if i == 0:
                                    out[0, py, px, 0] = brightness
                                    out[0, py, px, 1] = brightness * 0.4
                                    out[0, py, px, 2] = brightness * 0.2
                                else:
                                    out[0, py, px, :] = brightness
                            else:
                                out[0, py, px, :] = 0.1

        return {"video": out.clamp(0, 1)}

    def _draw_text(
        self,
        canvas: torch.Tensor,
        text: str,
        x: int,
        y: int,
        scale: int = 2,
        align_right: bool = False,
        color: tuple[float, float, float] = (0.7, 0.7, 0.7),
    ) -> None:
        """Render text using the 3x5 bitmap font."""
        char_w = 3 * scale
        gap = scale
        total_width = len(text) * (char_w + gap) - gap
        if align_right:
            x = x - total_width

        for ch in text:
            pattern = ALL_PATTERNS.get(ch)
            if pattern is None:
                x += char_w + gap
                continue
            for row_idx, row_bits in enumerate(pattern):
                for col_idx in range(3):
                    if row_bits & (1 << (2 - col_idx)):
                        py = y + row_idx * scale
                        px = x + col_idx * scale
                        py2 = min(self.height, py + scale)
                        px2 = min(self.width, px + scale)
                        if py >= 0 and px >= 0 and py < self.height and px < self.width:
                            canvas[0, py:py2, px:px2, 0] = color[0]
                            canvas[0, py:py2, px:px2, 1] = color[1]
                            canvas[0, py:py2, px:px2, 2] = color[2]
            x += char_w + gap

    def _draw_rect_outline(
        self,
        canvas: torch.Tensor,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: tuple[float, float, float] = (0.1, 0.1, 0.15),
    ) -> None:
        """Draw a 1px rectangle outline."""
        H, W = self.height, self.width
        x1 = max(0, min(W - 1, x1))
        x2 = max(0, min(W, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(0, min(H, y2))

        for c in range(3):
            canvas[0, y1, x1:x2, c] = color[c]
            canvas[0, y2 - 1, x1:x2, c] = color[c]
            canvas[0, y1:y2, x1, c] = color[c]
            canvas[0, y1:y2, x2 - 1, c] = color[c]
