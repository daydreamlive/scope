"""Controller input data model for interactive pipelines.

This module provides the CtrlInput dataclass for capturing keyboard and mouse
input from the frontend. It uses W3C event.code strings as the universal standard
for key identification.

Pipelines that need different keycode formats (e.g., Windows Virtual Keycodes
for world_engine compatibility) should convert internally using the provided
W3C_TO_WIN mapping.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CtrlInput:
    """Controller input state for interactive pipelines.

    Uses W3C event.code strings for key identification, which is:
    - Universal (web standard, not OS-specific)
    - Self-documenting ("KeyW" is clearer than 87)
    - Layout-independent (physical key position, not character)

    Attributes:
        button: Set of currently pressed keys using W3C event.code strings.
                Example: {"KeyW", "KeyA", "Space", "ShiftLeft"}
        mouse: Mouse velocity/delta as (dx, dy) tuple.
               Values are typically normalized floats.
    """

    button: set[str] = field(default_factory=set)
    mouse: tuple[float, float] = (0.0, 0.0)


def parse_ctrl_input(data: dict[str, Any]) -> CtrlInput:
    """Parse controller input from frontend JSON format.

    Args:
        data: Dictionary with 'button' (list of strings) and 'mouse' (list of 2 floats)

    Returns:
        CtrlInput instance with parsed values
    """
    button = set(data.get("button", []))
    mouse_data = data.get("mouse", [0.0, 0.0])
    mouse = (float(mouse_data[0]), float(mouse_data[1])) if mouse_data else (0.0, 0.0)
    return CtrlInput(button=button, mouse=mouse)


# W3C event.code to Windows Virtual Keycode mapping
# For pipelines that need Windows keycodes (e.g., world_engine compatibility)
W3C_TO_WIN: dict[str, int] = {
    # Letters (WASD)
    "KeyW": 87,
    "KeyA": 65,
    "KeyS": 83,
    "KeyD": 68,
    # Other common letters
    "KeyQ": 81,
    "KeyE": 69,
    "KeyR": 82,
    "KeyF": 70,
    "KeyC": 67,
    "KeyX": 88,
    "KeyZ": 90,
    # Space and modifiers
    "Space": 32,
    "ShiftLeft": 160,
    "ShiftRight": 161,
    "ControlLeft": 162,
    "ControlRight": 163,
    "AltLeft": 164,
    "AltRight": 165,
    # Arrow keys
    "ArrowUp": 38,
    "ArrowDown": 40,
    "ArrowLeft": 37,
    "ArrowRight": 39,
    # Other common keys
    "Enter": 13,
    "Escape": 27,
    "Tab": 9,
    "Backspace": 8,
    # Number keys
    "Digit1": 49,
    "Digit2": 50,
    "Digit3": 51,
    "Digit4": 52,
    "Digit5": 53,
    "Digit6": 54,
    "Digit7": 55,
    "Digit8": 56,
    "Digit9": 57,
    "Digit0": 48,
    # Mouse buttons (using descriptive names from frontend)
    "MouseLeft": 1,  # VK_LBUTTON
    "MouseRight": 2,  # VK_RBUTTON
    "MouseMiddle": 4,  # VK_MBUTTON
    "MouseBack": 5,  # VK_XBUTTON1
    "MouseForward": 6,  # VK_XBUTTON2
}


def convert_to_win_keycodes(ctrl_input: CtrlInput) -> set[int]:
    """Convert W3C event.code strings to Windows Virtual Keycodes.

    Use this in pipelines that need Windows keycodes (e.g., world_engine).

    Args:
        ctrl_input: CtrlInput with W3C event.code strings

    Returns:
        Set of Windows Virtual Keycode integers
    """
    return {W3C_TO_WIN[code] for code in ctrl_input.button if code in W3C_TO_WIN}
