from ..artifacts import HuggingfaceRepoArtifact
from ..base_schema import BasePipelineConfig, ModeDefaults, UsageType


class ScribbleConfig(BasePipelineConfig):
    """Configuration for Scribble preprocessor pipeline.

    This pipeline extracts contour/scribble line art from video frames using
    a neural network model. Based on VACE's ScribbleAnnotator, optimized for
    realtime streaming.
    """

    pipeline_id = "scribble"
    pipeline_name = "Scribble"
    pipeline_description = (
        "Extracts contour/scribble line art from video frames using a neural network. "
        "Based on VACE's anime-style contour extraction model."
    )
    artifacts = [
        HuggingfaceRepoArtifact(
            repo_id="ali-vilab/VACE-Annotators",
            files=["scribble/anime_style/netG_A_latest.pth"],
        ),
    ]
    supports_prompts = False
    modified = True
    usage = [UsageType.PREPROCESSOR]

    modes = {"video": ModeDefaults(default=True)}
