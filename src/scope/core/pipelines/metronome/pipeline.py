"""Metronome test pipeline for beat-sync latency compensation testing.

Renders a visual metronome that shows:
- A pulsing beat indicator (bright flash on each beat)
- Current beat number within the bar (1-based) and bar count
- Toggleable color overlays (layers A/B/C) that mix additively

The artificial latency simulates real pipeline processing delay by
buffering layer toggle states: a parameter set at time T only takes
visual effect at time T + latency_ms. This lets you verify that the
scheduler's lookahead correctly compensates for pipeline latency.
"""

import logging
import time
from typing import TYPE_CHECKING

import torch

from ..interface import Pipeline, Requirements
from .schema import MetronomeConfig

if TYPE_CHECKING:
    from ..base_schema import BasePipelineConfig

logger = logging.getLogger(__name__)

# Layer colors (RGB, 0-1 range) chosen for clear visual distinction
# and readable mixing when combined
_LAYER_COLORS = {
    "layer_a": torch.tensor([0.85, 0.15, 0.65]),  # magenta
    "layer_b": torch.tensor([0.10, 0.75, 0.80]),  # cyan
    "layer_c": torch.tensor([0.90, 0.75, 0.10]),  # gold
}

# Beat-number digit patterns (5x3 bitmap font)
_DIGITS = {
    0: [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
    1: [0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1],
    2: [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
    3: [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    4: [1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
    5: [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    6: [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    7: [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    8: [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    9: [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
}


def _draw_digit(
    canvas: torch.Tensor, digit: int, x: int, y: int, scale: int, color: torch.Tensor
):
    """Draw a single digit onto canvas at (x, y) with given pixel scale."""
    pattern = _DIGITS.get(digit, _DIGITS[0])
    for row in range(5):
        for col in range(3):
            if pattern[row * 3 + col]:
                y0 = y + row * scale
                y1 = y0 + scale
                x0 = x + col * scale
                x1 = x0 + scale
                canvas[y0:y1, x0:x1] = color


def _draw_number(
    canvas: torch.Tensor, number: int, x: int, y: int, scale: int, color: torch.Tensor
):
    """Draw a multi-digit number onto canvas. Returns width drawn."""
    digits = [int(d) for d in str(number)]
    char_w = 3 * scale + scale  # digit width + spacing
    for i, d in enumerate(digits):
        _draw_digit(canvas, d, x + i * char_w, y, scale, color)
    return len(digits) * char_w


class MetronomePipeline(Pipeline):
    """Visual metronome for testing beat-sync parameter scheduling."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return MetronomeConfig

    def __init__(self, device: torch.device | None = None, **kwargs):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._layer_colors = {k: v.to(self.device) for k, v in _LAYER_COLORS.items()}

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Input video cannot be None for MetronomePipeline")

        # Simulate pipeline latency
        latency_ms = kwargs.get("latency_ms", 0.0)
        if latency_ms > 0:
            time.sleep(latency_ms / 1000.0)

        # Get frame dimensions from input
        frame = video[0].squeeze(0)  # (H, W, C)
        H, W = frame.shape[0], frame.shape[1]

        # Beat state (injected by PipelineProcessor)
        bpm = kwargs.get("bpm", 0.0)
        beat_phase = kwargs.get("beat_phase", 0.0)
        beat_count = kwargs.get("beat_count", 0)
        beats_per_bar = 4  # standard 4/4

        # Beat position within bar (0-based)
        beat_in_bar = beat_count % beats_per_bar
        bar_number = beat_count // beats_per_bar

        # --- Build the frame ---
        canvas = torch.zeros(H, W, 3, device=self.device)

        # Background: dark gray base
        canvas[:] = torch.tensor([0.10, 0.10, 0.12], device=self.device)

        # Beat pulse: flash brightness on the downbeat region
        # Intensity fades from 1.0 at phase=0 to 0.0 at phase=1
        pulse = max(0.0, 1.0 - beat_phase * 3.0)  # quick decay

        # Downbeat (beat 1) gets a stronger, wider flash
        is_downbeat = beat_in_bar == 0
        pulse_color = (
            torch.tensor([1.0, 1.0, 1.0], device=self.device)
            if is_downbeat
            else torch.tensor([0.5, 0.5, 0.55], device=self.device)
        )

        # Flash the top third of the frame
        flash_h = H // 3
        canvas[:flash_h] += pulse * pulse_color * (0.7 if is_downbeat else 0.4)

        # --- Beat indicator pips at the bottom ---
        pip_area_top = H - H // 5
        pip_h = H // 8
        pip_w = W // (beats_per_bar * 2 + 1)
        pip_spacing = pip_w
        total_pip_width = beats_per_bar * pip_w + (beats_per_bar - 1) * pip_spacing
        pip_x_start = (W - total_pip_width) // 2
        pip_y = pip_area_top + (H - pip_area_top - pip_h) // 2

        for i in range(beats_per_bar):
            px = pip_x_start + i * (pip_w + pip_spacing)
            if i == beat_in_bar:
                # Active beat: bright white/yellow
                brightness = 0.5 + 0.5 * (1.0 - beat_phase)
                pip_color = torch.tensor(
                    [brightness, brightness, brightness * 0.85],
                    device=self.device,
                )
            elif i < beat_in_bar:
                # Past beats in this bar: dim
                pip_color = torch.tensor([0.25, 0.25, 0.28], device=self.device)
            else:
                # Future beats: darker
                pip_color = torch.tensor([0.15, 0.15, 0.17], device=self.device)
            canvas[pip_y : pip_y + pip_h, px : px + pip_w] = pip_color

        # --- Beat / Bar number display (center of frame) ---
        scale = max(3, min(H, W) // 40)
        text_color = torch.tensor([0.9, 0.9, 0.9], device=self.device)
        dim_color = torch.tensor([0.4, 0.4, 0.4], device=self.device)

        # Show: <beat_in_bar+1> . <bar_number>
        # Beat number (large, centered)
        beat_display = beat_in_bar + 1
        num_w = _draw_number(canvas, beat_display, 0, 0, scale, text_color)
        # Calculate position to center it
        center_x = (W - num_w) // 2
        center_y = H // 3 + (H // 3 - 5 * scale) // 2

        # Clear area first and redraw centered
        _draw_number(canvas, beat_display, center_x, center_y, scale, text_color)

        # Bar number (smaller, below beat number)
        bar_scale = max(2, scale * 2 // 3)
        bar_y = center_y + 6 * scale
        bar_num_w = _draw_number(canvas, bar_number, 0, 0, bar_scale, dim_color)
        bar_x = (W - bar_num_w) // 2
        _draw_number(canvas, bar_number, bar_x, bar_y, bar_scale, dim_color)

        # --- BPM display (top-right corner) ---
        if bpm > 0:
            bpm_int = int(round(bpm))
            bpm_scale = max(2, scale * 2 // 3)
            bpm_w = _draw_number(canvas, bpm_int, 0, 0, bpm_scale, dim_color)
            bpm_x = W - bpm_w - bpm_scale * 2
            bpm_y = bpm_scale * 2
            _draw_number(canvas, bpm_int, bpm_x, bpm_y, bpm_scale, dim_color)

        # --- Color layer overlays ---
        active_layers = []
        for layer_key, color in self._layer_colors.items():
            if kwargs.get(layer_key, False):
                active_layers.append(color)

        if active_layers:
            # Additive blend of active layers into the middle band
            overlay_top = H // 6
            overlay_bottom = H - H // 6
            overlay = torch.zeros(3, device=self.device)
            for color in active_layers:
                overlay += color * (0.5 / len(active_layers) + 0.15)
            canvas[overlay_top:overlay_bottom] += overlay
            # Also tint the beat pips
            canvas[pip_area_top:] += overlay * 0.3

        # Clamp and format output
        canvas = canvas.clamp(0, 1)
        result = canvas.unsqueeze(0)  # (1, H, W, C)

        return {"video": result}
