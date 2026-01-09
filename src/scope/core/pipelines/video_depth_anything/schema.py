from ..base_schema import BasePipelineConfig, ModeDefaults


class VideoDepthAnythingConfig(BasePipelineConfig):
    """Configuration for Video Depth Anything pipeline.

    This pipeline performs consistent depth estimation for videos using the
    Video-Depth-Anything Small model from ByteDance. It provides temporally consistent
    depth maps for video sequences.
    """

    pipeline_id = "video-depth-anything"
    pipeline_name = "Video Depth Anything"
    pipeline_description = (
        "Video depth estimation pipeline providing temporally consistent depth maps "
        "for video sequences using Video-Depth-Anything Small model."
    )
    docs_url = "https://github.com/DepthAnything/Video-Depth-Anything"

    requires_models = True
    supports_prompts = False
    modified = True

    modes = {"video": ModeDefaults(default=True)}
