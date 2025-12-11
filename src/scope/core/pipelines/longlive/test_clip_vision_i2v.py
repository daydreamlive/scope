import time
from pathlib import Path

import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from scope.core.config import get_model_file_path, get_models_dir
from scope.core.pipelines.clip_vision import CLIPVisionEncoder

from .pipeline import LongLivePipeline

print("=" * 80)
print("test_clip_vision_i2v: CLIP Vision Image-to-Video Integration Test")
print("=" * 80)

config = OmegaConf.create(
    {
        "model_dir": str(get_models_dir()),
        "generator_path": str(
            get_model_file_path("LongLive-1.3B/models/longlive_base.pt")
        ),
        "lora_path": str(get_model_file_path("LongLive-1.3B/models/lora.pt")),
        "text_encoder_path": str(
            get_model_file_path("WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors")
        ),
        "tokenizer_path": str(get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")),
        "model_config": OmegaConf.load(Path(__file__).parent / "model.yaml"),
        "height": 480,
        "width": 832,
    }
)

device = torch.device("cuda")

print(
    "\ntest_clip_vision_i2v: Loading LongLive Pipeline (t2v model with CLIP injection)..."
)
start_pipeline = time.time()
pipeline = LongLivePipeline(config, device=device, dtype=torch.bfloat16)
print(f"test_clip_vision_i2v: Pipeline loaded in {time.time() - start_pipeline:.3f}s")

print("\ntest_clip_vision_i2v: Loading CLIP Vision encoder...")
clip_vision_path = get_model_file_path(
    "CLIP_Vision/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
)
start_clip = time.time()
clip_vision = CLIPVisionEncoder(
    checkpoint_path=clip_vision_path,
    device=device,
    dtype=torch.bfloat16,
)
print(f"test_clip_vision_i2v: CLIP Vision loaded in {time.time() - start_clip:.3f}s")

example_image_path = (
    Path(__file__).parent.parent.parent.parent.parent.parent / "example.png"
)
print(f"\ntest_clip_vision_i2v: Encoding image: {example_image_path}")
start_encode = time.time()
clip_features = clip_vision.encode_image(example_image_path)
print(f"test_clip_vision_i2v: Image encoded in {time.time() - start_encode:.3f}s")
print(f"test_clip_vision_i2v: CLIP features shape: {clip_features.shape}")

# prompt_text = "A cheerful cartoon girl with blonde pigtails and a pink dress walking through a sunny meadow with green grass and blue sky, animation style, vibrant colors"
prompt_text = ""
print("\ntest_clip_vision_i2v: Generating video with CLIP Vision conditioning...")
print(f"test_clip_vision_i2v: Prompt: {prompt_text}")

# Monkey-patch the DenoiseBlock to inject clip_fea
# This is a temporary hack for testing before full pipeline integration
original_denoise_call = None
clip_projection = None


def patch_denoise_block():
    from ..wan2_1.blocks.denoise import DenoiseBlock

    global original_denoise_call, clip_projection
    original_denoise_call = DenoiseBlock.__call__

    def patched_call(self, components, state):
        global clip_projection

        # Inject clip_fea into the pipeline state
        if not hasattr(state, "_clip_fea_injected"):
            state.set("_clip_features_for_injection", clip_features)
            state._clip_fea_injected = True
            print("test_clip_vision_i2v: Injected CLIP features into pipeline state")

        block_state = self.get_block_state(state)

        # Get text conditioning embeddings
        text_embeds = block_state.conditioning_embeds

        # Concatenate CLIP features with text embeddings for style conditioning
        # For t2v models: keep total tokens at 512 (model's text_len) by truncating text
        clip_fea = state.get("_clip_features_for_injection")
        if clip_fea is not None:
            # Create projection layer on first use
            if clip_projection is None:
                import torch.nn as nn

                clip_dim = clip_fea.shape[-1]  # 1280
                text_dim = text_embeds.shape[-1]  # 4096
                clip_projection = nn.Linear(clip_dim, text_dim, bias=False).to(
                    device=text_embeds.device, dtype=text_embeds.dtype
                )
                # Initialize with small random weights for stable conditioning
                with torch.no_grad():
                    nn.init.xavier_uniform_(clip_projection.weight, gain=0.1)
                print(
                    f"test_clip_vision_i2v: Created CLIP projection layer {clip_dim} -> {text_dim}"
                )

            # Expand CLIP features to match batch size
            batch_size = text_embeds.shape[0]
            clip_fea_expanded = clip_fea.expand(batch_size, -1, -1).to(
                text_embeds.device, text_embeds.dtype
            )

            # Project CLIP features to match text embedding dimension
            clip_fea_projected = clip_projection(clip_fea_expanded)

            # Truncate text to make room for CLIP tokens (keep total at 512)
            clip_tokens = clip_fea_projected.shape[1]  # 257
            max_text_tokens = 512 - clip_tokens  # 255
            text_embeds_truncated = text_embeds[:, :max_text_tokens, :]

            # Concatenate: [CLIP tokens (257), truncated text tokens (255)] = 512 total
            combined_embeds = torch.cat(
                [clip_fea_projected, text_embeds_truncated], dim=1
            )
            print(
                f"test_clip_vision_i2v: Combined embeddings - CLIP {clip_fea_projected.shape} + Text {text_embeds_truncated.shape} = {combined_embeds.shape} (kept at 512 tokens)"
            )
        else:
            combined_embeds = text_embeds

        conditional_dict = {"prompt_embeds": combined_embeds}

        # Continue with original denoising logic but with modified conditional_dict
        scale_size = (
            components.config.vae_spatial_downsample_factor
            * components.config.patch_embedding_spatial_downsample_factor
        )
        frame_seq_length = (block_state.height // scale_size) * (
            block_state.width // scale_size
        )

        noise = block_state.latents
        batch_size = noise.shape[0]
        num_frames = noise.shape[1]
        denoising_step_list = block_state.current_denoising_step_list.clone()

        start_frame = block_state.current_start_frame
        if block_state.start_frame is not None:
            start_frame = block_state.start_frame

        end_frame = start_frame + num_frames

        if block_state.noise_scale is not None:
            denoising_step_list[0] = int(1000 * block_state.noise_scale) - 100

        # Denoising loop with clip_fea
        for index, current_timestep in enumerate(denoising_step_list):
            timestep = (
                torch.ones(
                    [batch_size, num_frames],
                    device=noise.device,
                    dtype=torch.int64,
                )
                * current_timestep
            )

            if index < len(denoising_step_list) - 1:
                _, denoised_pred = components.generator(
                    noisy_image_or_video=noise,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=block_state.kv_cache,
                    crossattn_cache=block_state.crossattn_cache,
                    current_start=start_frame * frame_seq_length,
                    current_end=end_frame * frame_seq_length,
                    kv_cache_attention_bias=block_state.kv_cache_attention_bias,
                )
                next_timestep = denoising_step_list[index + 1]
                flattened_pred = denoised_pred.flatten(0, 1)
                random_noise = torch.randn(
                    flattened_pred.shape,
                    device=flattened_pred.device,
                    dtype=flattened_pred.dtype,
                    generator=block_state.generator,
                )
                noise = components.scheduler.add_noise(
                    flattened_pred,
                    random_noise,
                    next_timestep
                    * torch.ones(
                        [batch_size * num_frames],
                        device=noise.device,
                        dtype=torch.long,
                    ),
                ).unflatten(0, denoised_pred.shape[:2])
            else:
                _, denoised_pred = components.generator(
                    noisy_image_or_video=noise,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=block_state.kv_cache,
                    crossattn_cache=block_state.crossattn_cache,
                    current_start=start_frame * frame_seq_length,
                    current_end=end_frame * frame_seq_length,
                    kv_cache_attention_bias=block_state.kv_cache_attention_bias,
                )

        block_state.latents = denoised_pred
        self.set_block_state(state, block_state)
        return components, state

    DenoiseBlock.__call__ = patched_call
    print("test_clip_vision_i2v: Patched DenoiseBlock to use CLIP features")


def unpatch_denoise_block():
    from ..wan2_1.blocks.denoise import DenoiseBlock

    if original_denoise_call is not None:
        DenoiseBlock.__call__ = original_denoise_call
        print("test_clip_vision_i2v: Unpatched DenoiseBlock")


try:
    patch_denoise_block()

    outputs = []
    latency_measures = []
    fps_measures = []

    num_frames = 0
    max_output_frames = 81

    while num_frames < max_output_frames:
        start = time.time()

        prompts = [{"text": prompt_text, "weight": 100}]
        output = pipeline(prompts=prompts)

        num_output_frames, _, _, _ = output.shape
        latency = time.time() - start
        fps = num_output_frames / latency

        print(
            f"test_clip_vision_i2v: Generated {num_output_frames} frames, latency={latency:.2f}s, fps={fps:.2f}"
        )

        latency_measures.append(latency)
        fps_measures.append(fps)
        num_frames += num_output_frames
        outputs.append(output.detach().cpu())

    output_video = torch.concat(outputs)
    print(f"\ntest_clip_vision_i2v: Final output shape: {output_video.shape}")
    output_video_np = output_video.contiguous().numpy()
    output_path = Path(__file__).parent / "output_clip_vision_i2v.mp4"
    export_to_video(output_video_np, output_path, fps=16)

    print("\n" + "=" * 80)
    print("test_clip_vision_i2v: Performance Statistics")
    print("=" * 80)
    print(
        f"test_clip_vision_i2v: Latency - Avg: {sum(latency_measures) / len(latency_measures):.2f}s, "
        f"Max: {max(latency_measures):.2f}s, Min: {min(latency_measures):.2f}s"
    )
    print(
        f"test_clip_vision_i2v: FPS - Avg: {sum(fps_measures) / len(fps_measures):.2f}, "
        f"Max: {max(fps_measures):.2f}, Min: {min(fps_measures):.2f}"
    )

    print("\n" + "=" * 80)
    print("test_clip_vision_i2v: Test Complete")
    print("=" * 80)
    print(f"test_clip_vision_i2v: Output video: {output_path}")
    print(f"test_clip_vision_i2v: Total frames: {output_video.shape[0]}")
    print("\ntest_clip_vision_i2v: Next steps:")
    print(
        "test_clip_vision_i2v: 1. Review the output video to see CLIP conditioning effect"
    )
    print(
        "test_clip_vision_i2v: 2. Compare with regular text-only generation (test.py)"
    )
    print(
        "test_clip_vision_i2v: 3. If results look good, proceed with full pipeline integration"
    )

finally:
    unpatch_denoise_block()
