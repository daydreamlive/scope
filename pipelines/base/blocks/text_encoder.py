from diffusers.modular_pipelines import ModularPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
)


class Wan2_1RTTextEncoderBlock(ModularPipelineBlocks):
    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder"),
            ComponentSpec(
                "tokenizer",
            ),
        ]
