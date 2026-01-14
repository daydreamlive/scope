"""Layout Control preprocessor for VACE conditioning.

Generates layout control frames from keyboard/mouse input for interactive
point-based subject control in video generation.
"""

from scope.core.plugins import hookimpl

from .preprocessor import LayoutControlPreprocessor


@hookimpl
def register_preprocessors(register):
    """Register the layout control preprocessor."""
    register(
        "layout-control",
        "Layout Control (WASD)",
        LayoutControlPreprocessor,
    )


__all__ = ["LayoutControlPreprocessor"]
