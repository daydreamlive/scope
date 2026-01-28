from ..base_schema import BasePipelineConfig, ModeDefaults, UsageType


class GrayConfig(BasePipelineConfig):
    """Configuration for Gray preprocessor pipeline.

    This pipeline converts video frames to grayscale. It's a lightweight
    preprocessor optimized for realtime use - no model loading required.
    """

    pipeline_id = "gray"
    pipeline_name = "Grayscale"
    pipeline_description = (
        "Converts video frames to grayscale. Lightweight preprocessor "
        "with no model required, optimized for realtime use."
    )
    supports_prompts = False
    modified = True
    usage = [UsageType.PREPROCESSOR]

    modes = {"video": ModeDefaults(default=True)}
