"""Controller Visualizer pipeline implementation.

Displays 4 directional keys (WASD + arrows combined) in lower-left,
and mouse x/y values in lower-right.
"""

from typing import TYPE_CHECKING

import torch

from scope.core.pipelines.controller import CtrlInput

from ..interface import Pipeline
from .schema import ControllerVisualizerConfig

if TYPE_CHECKING:
    from ..base_schema import BasePipelineConfig


class ControllerVisualizerPipeline(Pipeline):
    """Displays 4 directional keys + mouse values for debugging."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return ControllerVisualizerConfig

    def __init__(
        self,
        height: int = 512,
        width: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        **kwargs,  # Accept extra params from pipeline manager (loras, vae_type, etc.)
    ):
        self.height = height
        self.width = width
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype

        # Pre-allocate output buffer (T, H, W, C) - single frame
        self._output = torch.zeros(
            (1, height, width, 3), dtype=torch.float32, device=self.device
        )

        # 4 directional keys - each triggered by WASD or arrows
        # (col, row) positions, (keys that trigger it)
        self._directions = {
            "up": ((1, 0), {"KeyW", "ArrowUp"}),
            "left": ((0, 1), {"KeyA", "ArrowLeft"}),
            "down": ((1, 1), {"KeyS", "ArrowDown"}),
            "right": ((2, 1), {"KeyD", "ArrowRight"}),
        }

        # Mouse cursor position (accumulates over time, starts at center)
        self._cursor_x = width / 2.0
        self._cursor_y = height / 2.0

        # Mouse button indicators - positioned near cursor
        # (offset_x, offset_y) relative to cursor, color (R, G, B)
        self._mouse_buttons = {
            "MouseLeft": ((-15, -15), (0.2, 0.8, 0.2)),  # Green, top-left
            "MouseMiddle": ((0, -20), (0.8, 0.8, 0.2)),  # Yellow, top-center
            "MouseRight": ((15, -15), (0.2, 0.2, 0.8)),  # Blue, top-right
        }

    def __call__(self, **kwargs) -> torch.Tensor:
        """Render controller input visualization.

        Args:
            ctrl_input: CtrlInput with button set and mouse tuple

        Returns:
            Tensor of shape (1, H, W, C) in [0, 1] range
        """
        ctrl_input: CtrlInput = kwargs.get("ctrl_input") or CtrlInput()

        # Clear to dark background
        self._output.fill_(0.1)

        # Draw 4 directional keys in lower-left
        key_size = 30
        gap = 5
        margin = 20
        base_y = self.height - margin - key_size * 2 - gap

        for _direction, ((col, row), trigger_keys) in self._directions.items():
            x = margin + col * (key_size + gap)
            y = base_y + row * (key_size + gap)

            # Check if any trigger key is pressed
            is_pressed = bool(ctrl_input.button & trigger_keys)

            if is_pressed:
                self._output[0, y : y + key_size, x : x + key_size, :] = 0.9
            else:
                # Draw border only
                border = 2
                self._output[0, y : y + border, x : x + key_size, :] = 0.3
                self._output[
                    0, y + key_size - border : y + key_size, x : x + key_size, :
                ] = 0.3
                self._output[0, y : y + key_size, x : x + border, :] = 0.3
                self._output[
                    0, y : y + key_size, x + key_size - border : x + key_size, :
                ] = 0.3

        # Mouse indicator - red dot that tracks cursor position across full canvas
        mouse_dx, mouse_dy = ctrl_input.mouse

        # Accumulate mouse deltas into cursor position
        self._cursor_x += mouse_dx
        self._cursor_y += mouse_dy

        # Clamp to canvas bounds (with margin for dot size)
        dot_size = 5
        self._cursor_x = max(dot_size, min(self.width - dot_size, self._cursor_x))
        self._cursor_y = max(dot_size, min(self.height - dot_size, self._cursor_y))

        dot_x = int(self._cursor_x)
        dot_y = int(self._cursor_y)

        # Draw red dot
        dot_size = 5
        y1 = max(0, dot_y - dot_size)
        y2 = min(self.height, dot_y + dot_size)
        x1 = max(0, dot_x - dot_size)
        x2 = min(self.width, dot_x + dot_size)
        self._output[0, y1:y2, x1:x2, 0] = 0.9  # Red
        self._output[0, y1:y2, x1:x2, 1] = 0.1
        self._output[0, y1:y2, x1:x2, 2] = 0.1

        # Draw mouse button indicators near cursor
        indicator_size = 4
        for button_name, ((offset_x, offset_y), color) in self._mouse_buttons.items():
            is_pressed = button_name in ctrl_input.button
            if is_pressed:
                # Calculate indicator position relative to cursor
                ind_x = int(self._cursor_x + offset_x)
                ind_y = int(self._cursor_y + offset_y)

                # Clamp to bounds
                iy1 = max(0, ind_y - indicator_size)
                iy2 = min(self.height, ind_y + indicator_size)
                ix1 = max(0, ind_x - indicator_size)
                ix2 = min(self.width, ind_x + indicator_size)

                # Draw colored indicator
                self._output[0, iy1:iy2, ix1:ix2, 0] = color[0]
                self._output[0, iy1:iy2, ix1:ix2, 1] = color[1]
                self._output[0, iy1:iy2, ix1:ix2, 2] = color[2]

        return self._output.clamp(0, 1)
