from ..base_schema import BasePipelineConfig, ModeDefaults


class PassthroughConfig(BasePipelineConfig):
    pipeline_id = "passthrough"
    pipeline_name = "Passthrough"
    pipeline_description = "A pipeline that returns the input video without any processing that is useful for testing and debugging."

    supports_prompts = False

    modes = {"video": ModeDefaults(default=True)}
