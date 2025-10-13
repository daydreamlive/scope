import time

import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from lib.models_config import get_model_file_path, get_models_dir

from ..memory import is_cuda_low_memory
from .pipeline import LongLivePipeline

config = OmegaConf.load("pipelines/longlive/model.yaml")

models_dir = get_models_dir()
height = 480
width = 832

config["model_dir"] = str(models_dir)
config["generator_path"] = get_model_file_path("LongLive-1.3B/models/longlive_base.pt")
config["lora_path"] = get_model_file_path("LongLive-1.3B/models/lora.pt")
config["text_encoder_path"] = str(
    get_model_file_path("WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors")
)
config["height"] = height
config["width"] = width

device = torch.device("cuda")
pipeline = LongLivePipeline(
    config, low_memory=is_cuda_low_memory(device), device=device, dtype=torch.bfloat16
)

prompts = [
    "A realistic video of a Texas Hold'em poker event at a casino. A male player in his late 30s with a medium build, short dark hair, light stubble, and a sharp jawline wears a fitted navy blazer over a charcoal crew-neck tee, dark jeans, and a stainless-steel watch. He sits at a well-lit poker table and tightly grips his hole cards, wearing a tense, serious expression. The table is filled with chips of various colors, the dealer is seen dealing cards, and several rows of slot machines glow in the background. The camera focuses on the player's strained concentration. Wide shot to medium close-up.",
    "A realistic video of a Texas Hold'em poker event at a casino. The same male player—late 30s, medium build, short dark hair, light stubble, sharp jawline—dressed in a fitted navy blazer over a charcoal tee, dark jeans, and a stainless-steel watch—flicks his cards onto the felt, then leans back in the chair with arms spread wide in celebration. The dealer continues dealing to the table as stacks of multicolored chips crowd the surface; slot machines and nearby patrons fill the background. The camera locks onto the player’s exuberant reaction. Wide shot to medium close-up.",
    "A realistic video of a Texas Hold'em poker event at a casino. The same late-30s male player, medium build with short dark hair and light stubble, wearing a navy blazer, charcoal tee, dark jeans, and a stainless-steel watch, reveals the winning hand and leans back in celebration while the dealer keeps the game moving. A nearby patron claps and cheers for the winner, amplifying the festive atmosphere. The table brims with colorful chips, with slot machines and other tables behind. The camera centers on the winner’s reaction as the applause rises. Wide shot to medium close-up.",
    "A realistic video of a Texas Hold'em poker event at a casino. The same male player—late 30s, medium build, short dark hair, light stubble—still in his navy blazer, charcoal tee, dark jeans, and stainless-steel watch—sits upright and begins neatly arranging the stacks of chips in front of him, methodically straightening and organizing the piles. The dealer continues dealing, and rows of slot machines pulse in the background. The camera captures the composed, purposeful movements at the well-lit table. Wide shot to medium close-up.",
    "A realistic video of a Texas Hold'em poker event at a casino. The same late-30s male player with short dark hair, light stubble, and a sharp jawline, wearing a fitted navy blazer over a charcoal tee, dark jeans, and a stainless-steel watch, glances over his chips and breaks into a proud, self-assured smile, basking in the victorious moment. Multicolored chips crowd the felt, the dealer works the table, and slot machines glow behind. The camera emphasizes the winner’s pride and satisfaction. Wide shot to medium close-up.",
    "A realistic video of a Texas Hold'em poker event at a casino. The same male player—late 30s, medium build, short dark hair, light stubble—dressed in a navy blazer, charcoal tee, dark jeans, and a stainless-steel watch—shares a celebratory high-five with a nearby patron after the win, laughter and cheers rippling around the table. Stacks of chips are spread across the felt, the dealer continues dealing, and the background features rows of slot machines and other patrons. The camera focuses on the jubilant interaction. Wide shot to medium close-up.",
]

outputs = []
for i, prompt in enumerate(prompts):
    should_prepare = i == 0

    pipeline.prepare(prompts=[prompt], should_prepare=should_prepare)

    num_frames = 0
    max_output_frames = 81
    while num_frames < max_output_frames:
        start = time.time()

        output = pipeline()

        num_output_frames, _, _, _ = output.shape
        latency = time.time() - start
        fps = num_output_frames / latency

        print(
            f"Pipeline generated {num_output_frames} frames latency={latency:2f}s fps={fps}"
        )

        num_frames += num_output_frames
        outputs.append(output.detach().cpu())

# Concatenate all of the THWC tensors
output_video = torch.concat(outputs)
print(output_video.shape)
output_video_np = output_video.contiguous().numpy()
export_to_video(output_video_np, "pipelines/longlive/output.mp4", fps=16)
