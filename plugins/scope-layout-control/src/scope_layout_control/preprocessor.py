"""Layout Control preprocessor for VACE conditioning.

Generates white background + black circle contour frames based on keyboard/mouse
input for interactive point-based subject control.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from scope.core.pipelines.controller import CtrlInput
    from scope.core.plugins import PreprocessorContext

logger = logging.getLogger(__name__)


class LayoutControlPreprocessor:
    """Generates layout control frames from controller input.

    This preprocessor maintains internal state (circle position) and updates it
    based on WASD/arrow key input and mouse movement. It generates frames with
    a white background and black circle contour at the current position, suitable
    for VACE layout conditioning.

    The circle position persists between calls, allowing smooth movement as the
    user holds keys down.

    Example:
        ```python
        preprocessor = LayoutControlPreprocessor()

        # Create context with controller input
        ctx = PreprocessorContext(
            ctrl_input=ctrl_input,
            target_height=512,
            target_width=512,
            num_frames=12,
        )

        # Generate layout frames
        layout_frames = preprocessor(ctx)
        # Output: [1, C, F, H, W] in [-1, 1] range
        ```
    """

    def __init__(
        self,
        radius: int = 80,
        move_speed: float = 0.004,
        mouse_sensitivity: float = 0.002,
        initial_x: float = 0.5,
        initial_y: float = 0.35,
    ):
        """Initialize the layout control preprocessor.

        Args:
            radius: Circle radius in pixels (default 80)
            move_speed: Movement speed per frame for WASD keys (0.0-1.0 normalized)
            mouse_sensitivity: Mouse movement sensitivity multiplier
            initial_x: Initial X position (0.0-1.0 normalized, default center)
            initial_y: Initial Y position (0.0-1.0 normalized, default upper-center)
        """
        self._radius = radius
        self._move_speed = move_speed
        self._mouse_sensitivity = mouse_sensitivity

        # Circle position state (normalized 0.0-1.0)
        self._pos_x = initial_x
        self._pos_y = initial_y

        # Previous position for interpolation
        self._prev_x = initial_x
        self._prev_y = initial_y

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def reset(self, x: float = 0.5, y: float = 0.35):
        """Reset circle position to specified coordinates.

        Args:
            x: X position (0.0-1.0 normalized)
            y: Y position (0.0-1.0 normalized)
        """
        self._pos_x = x
        self._pos_y = y
        self._prev_x = x
        self._prev_y = y

    def _update_position(self, ctrl_input: "CtrlInput"):
        """Update circle position based on controller input.

        Args:
            ctrl_input: Controller input with pressed keys and mouse delta
        """
        # Store previous position for interpolation
        self._prev_x = self._pos_x
        self._prev_y = self._pos_y

        # Apply keyboard movement (WASD and arrow keys)
        buttons = ctrl_input.button
        if "KeyW" in buttons or "ArrowUp" in buttons:
            self._pos_y -= self._move_speed
        if "KeyS" in buttons or "ArrowDown" in buttons:
            self._pos_y += self._move_speed
        if "KeyA" in buttons or "ArrowLeft" in buttons:
            self._pos_x -= self._move_speed
        if "KeyD" in buttons or "ArrowRight" in buttons:
            self._pos_x += self._move_speed

        # Apply mouse movement
        mouse_dx, mouse_dy = ctrl_input.mouse
        self._pos_x += mouse_dx * self._mouse_sensitivity
        self._pos_y += mouse_dy * self._mouse_sensitivity

        # Clamp to valid range (leave margin for circle radius)
        margin = 0.1
        self._pos_x = max(margin, min(1.0 - margin, self._pos_x))
        self._pos_y = max(margin, min(1.0 - margin, self._pos_y))

    def _create_layout_frame(
        self, x_norm: float, y_norm: float, height: int, width: int
    ) -> np.ndarray:
        """Create a single layout control frame (white bg + black contour).

        Args:
            x_norm: Normalized X position (0.0-1.0)
            y_norm: Normalized Y position (0.0-1.0)
            height: Frame height in pixels
            width: Frame width in pixels

        Returns:
            Frame as numpy array [H, W, C] uint8 with white bg and black circle
        """
        # White background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Convert normalized position to pixel coordinates
        px = int(x_norm * width)
        py = int(y_norm * height)

        # Clamp to keep circle fully visible
        px = max(self._radius, min(width - self._radius, px))
        py = max(self._radius, min(height - self._radius, py))

        # Draw black circle contour using simple algorithm (no cv2 dependency)
        self._draw_circle_contour(frame, px, py, self._radius, thickness=3)

        return frame

    def _draw_circle_contour(
        self,
        frame: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
        thickness: int = 3,
    ):
        """Draw a circle contour on the frame (no fill, just outline).

        Uses Bresenham-like algorithm to avoid cv2 dependency.

        Args:
            frame: Frame to draw on [H, W, C]
            cx: Center X in pixels
            cy: Center Y in pixels
            radius: Circle radius in pixels
            thickness: Line thickness in pixels
        """
        height, width = frame.shape[:2]

        # Draw circle using parametric approach
        for t in range(360 * 4):  # Higher resolution for smoother circle
            angle = t * np.pi / (180 * 2)
            for r_offset in range(-thickness // 2, thickness // 2 + 1):
                r = radius + r_offset
                x = int(cx + r * np.cos(angle))
                y = int(cy + r * np.sin(angle))
                if 0 <= x < width and 0 <= y < height:
                    frame[y, x] = [0, 0, 0]  # Black

    def __call__(
        self,
        ctx: "PreprocessorContext",
    ) -> torch.Tensor:
        """Generate layout control frames from controller input.

        Args:
            ctx: PreprocessorContext with ctrl_input, target dimensions, and num_frames

        Returns:
            Layout frames in VACE format: [1, C, F, H, W] float in [-1, 1] range
        """
        # Update position based on controller input if available
        if ctx.ctrl_input is not None:
            self._update_position(ctx.ctrl_input)

        height = ctx.target_height
        width = ctx.target_width
        num_frames = ctx.num_frames

        # Generate frames with interpolation from previous to current position
        frames = []
        for i in range(num_frames):
            # Interpolate position across frames for smooth motion
            t = i / max(num_frames - 1, 1)  # 0 to 1
            interp_x = self._prev_x + (self._pos_x - self._prev_x) * t
            interp_y = self._prev_y + (self._pos_y - self._prev_y) * t

            frame = self._create_layout_frame(interp_x, interp_y, height, width)
            frames.append(frame)

        # Stack frames: [F, H, W, C]
        frames_np = np.stack(frames)

        # Convert to tensor and format for VACE: [1, C, F, H, W] in [-1, 1]
        frames_t = torch.from_numpy(frames_np).float() / 255.0  # [0, 1]
        frames_t = frames_t * 2.0 - 1.0  # [-1, 1]

        # [F, H, W, C] -> [1, C, F, H, W]
        frames_t = frames_t.permute(0, 3, 1, 2)  # [F, C, H, W]
        frames_t = frames_t.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, F, H, W]

        return frames_t.to(self._device)

    @property
    def position(self) -> tuple[float, float]:
        """Get current circle position (normalized 0.0-1.0)."""
        return (self._pos_x, self._pos_y)
