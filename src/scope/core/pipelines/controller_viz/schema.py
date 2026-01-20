"""Schema for Controller Visualizer pipeline."""

from ..base_schema import BasePipelineConfig, CtrlInput, ModeDefaults


class ControllerVisualizerConfig(BasePipelineConfig):
    """Configuration for the Controller Visualizer pipeline.

    This pipeline visualizes WASD keyboard and mouse inputs in real-time,
    useful for testing and debugging the controller input system.
    """

    pipeline_id = "controller-viz"
    pipeline_name = "Controller Visualizer"
    pipeline_description = (
        "Visualizes WASD keyboard and mouse controller inputs in real-time. "
        "Useful for testing the controller input system."
    )

    # No prompts needed for visualization
    supports_prompts = False

    # Text mode (no video input required)
    modes = {"text": ModeDefaults(default=True)}

    # Controller input support - presence of this field enables controller input capture
    ctrl_input: CtrlInput | None = None
