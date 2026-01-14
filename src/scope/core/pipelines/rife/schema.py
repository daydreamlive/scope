from ..artifacts import GoogleDriveArtifact
from ..base_schema import BasePipelineConfig, ModeDefaults, UsageType


class RIFEConfig(BasePipelineConfig):
    """Configuration for RIFE frame interpolation pipeline.

    This pipeline uses RIFE HDv3 (Real-Time Intermediate Flow Estimation)
    to double the frame rate of input video by generating intermediate frames.

    Model weights are from Practical-RIFE v4.25:
    https://github.com/hzwer/Practical-RIFE

    This pipeline uses the v4.25 architecture with 5 blocks (block0-block4)
    and scale_list [16, 8, 4, 2, 1]. The model weights are downloaded from
    Google Drive and extracted from the ZIP archive.
    """

    pipeline_id = "rife"
    pipeline_name = "RIFE"
    pipeline_description = (
        "Frame interpolation pipeline using RIFE HDv3 to double the frame rate "
        "of input video by generating intermediate frames."
    )
    docs_url = "https://github.com/hzwer/Practical-RIFE"
    artifacts = [
        GoogleDriveArtifact(
            file_id="1Smy6gY7BkS_RzCjPCbMEy-TsX8Ma5B0R",  # Practical-RIFE v4.25
            files=["flownet.pkl"],
            name="RIFE",
        ),
    ]
    supports_prompts = False
    modified = True

    usage = [UsageType.POSTPROCESSOR]

    modes = {"video": ModeDefaults(default=True)}
