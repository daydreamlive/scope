from diffusers.modular_pipelines import AutoPipelineBlocks, ModularPipelineBlocks


class Wan2_1RTSetTimestepsBlock(ModularPipelineBlocks):
    pass


class Wan2_1InitCachesBlock(ModularPipelineBlocks):
    pass


# text-to-video
class Wan2_1RTBeforeDenoiseBlock(ModularPipelineBlocks):
    pass


# video-to-video
class Wan2_1RTVideoToVideoBeforeDenoiseBlock(ModularPipelineBlocks):
    pass


class Wan2_1RTAutoBeforeDenoiseBlock(AutoPipelineBlocks):
    block_classes = [
        Wan2_1RTBeforeDenoiseBlock,
        Wan2_1RTVideoToVideoBeforeDenoiseBlock,
    ]
    block_names = ["text-to-video", "video-to-video"]
    # If `chunk` is provided run the video-to-video block
    # Otherwise run the text-to-video block
    block_trigger_inputs = [None, "chunk"]
