from ..artifacts import HuggingfaceRepoArtifact
from ..base_schema import BasePipelineConfig, ModeDefaults, UsageType


class RIFEConfig(BasePipelineConfig):
    """Configuration for RIFE frame interpolation pipeline.

    This pipeline uses RIFE HDv3 (Real-Time Intermediate Flow Estimation)
    to double the frame rate of input video by generating intermediate frames.

    Model weights are from Practical-RIFE v4.25:
    https://github.com/hzwer/Practical-RIFE
    """

    pipeline_id = "rife"
    pipeline_name = "RIFE"
    pipeline_description = (
        "Frame interpolation pipeline using RIFE HDv3 to double the frame rate "
        "of input video by generating intermediate frames."
    )
    docs_url = "https://github.com/hzwer/Practical-RIFE"
    artifacts = [
        HuggingfaceRepoArtifact(
            repo_id="daydreamlive/RIFE",
            files=["config.json", "flownet.pkl"],
        ),
    ]
    supports_prompts = False
    modified = True

    usage = [UsageType.POSTPROCESSOR]

    modes = {"video": ModeDefaults(default=True)}
