"""Scribble pipeline configuration schema."""

from ..base_schema import BasePipelineConfig


class ScribbleConfig(BasePipelineConfig):
    """Configuration for Scribble contour extraction pipeline."""

    pipeline_id = "scribble"
    pipeline_name = "Scribble"
    pipeline_description = "Contour/scribble extraction for VACE conditioning."

    input_nc: int = 3
    output_nc: int = 1
    n_residual_blocks: int = 3
