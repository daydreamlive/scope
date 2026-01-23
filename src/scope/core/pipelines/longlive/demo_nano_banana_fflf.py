"""
Demo: Nano Banana + LongLive VACE First-Frame-Last-Frame (FFLF)

This demo combines:
1. Nano Banana (Google Gemini image generation) to create start/end frames
2. LongLive pipeline with VACE FFLF mode to generate video between frames

Nano Banana can also edit frames mid-stream by providing an image + text prompt.

Usage:
    # Set your API key
    export GOOGLE_API_KEY=your_api_key

    # Or pass directly in the config below
    python -m scope.core.pipelines.longlive.demo_nano_banana_fflf
"""

import os
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf
from PIL import Image

from scope.core.config import get_model_file_path, get_models_dir

from .pipeline import LongLivePipeline

# ============================= CONFIGURATION =============================

CONFIG = {
    # ===== NANO BANANA (GEMINI) CONFIG =====
    "api_key": os.environ.get("GOOGLE_API_KEY", ""),
    "model": "gemini-2.5-flash-image",  # or "gemini-3-pro-image-preview" for higher quality
    # ===== FRAME GENERATION PROMPTS =====
    # These prompts will be sent to Nano Banana to generate the start/end frames
    "first_frame_prompt": "A beautiful woman with flowing red hair standing in a sunlit meadow, photorealistic, golden hour lighting",
    "last_frame_prompt": "The same woman now sitting peacefully in the meadow, wind blowing through her hair, photorealistic, golden hour lighting",
    # ===== OPTIONAL: USE EXISTING IMAGES INSTEAD =====
    # Set to image paths to skip Nano Banana generation
    "use_existing_first_frame": None,  # e.g., "path/to/first.png"
    "use_existing_last_frame": None,  # e.g., "path/to/last.png"
    # ===== VIDEO GENERATION PROMPT =====
    # This describes the transition/motion between frames
    "video_prompt": "A woman gracefully transitions from standing to sitting in a sunlit meadow, wind blowing through her red hair",
    # ===== LONGLIVE/VACE PARAMETERS =====
    "num_chunks": 2,  # Number of generation chunks
    "frames_per_chunk": 12,  # Frames per chunk (12 = 3 latent * 4 temporal upsample)
    "height": 512,
    "width": 512,
    "vace_context_scale": 1.5,
    "vae_type": "tae",
    # ===== OUTPUT =====
    "output_dir": "vace_tests/nano_banana_fflf",
    "save_generated_frames": True,  # Save the Nano Banana generated frames
}

# ========================= END CONFIGURATION =========================


class NanoBananaClient:
    """Simple client for Nano Banana (Google Gemini) image generation."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-image"):
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "Please install the google-genai package: pip install google-genai"
            )

        self.genai = genai
        self.types = types
        self.client = genai.Client(api_key=api_key)
        self.model = model
        print(f"NanoBananaClient initialized with model: {model}")

    def generate_image(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        image_size: str = "1K",
    ) -> Image.Image:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of the image to generate
            aspect_ratio: Aspect ratio (e.g., "1:1", "16:9", "9:16")
            image_size: Size ("1K", "2K", or "4K" for Pro model)

        Returns:
            PIL Image
        """
        print(f"Generating image: '{prompt[:50]}...'")

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self.types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=self.types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                ),
            ),
        )

        for part in response.parts:
            if part.inline_data is not None:
                # Extract image bytes and convert to PIL Image
                image_bytes = part.inline_data.data
                image = Image.open(BytesIO(image_bytes))
                print(f"Generated image: {image.size}")
                return image

        raise ValueError("No image generated in response")

    def edit_image(
        self,
        image: Image.Image,
        prompt: str,
        aspect_ratio: str = "1:1",
    ) -> Image.Image:
        """
        Edit an existing image based on a text prompt.

        Args:
            image: Input PIL Image to edit
            prompt: Text description of the edit to apply
            aspect_ratio: Output aspect ratio

        Returns:
            Edited PIL Image
        """
        print(f"Editing image: '{prompt[:50]}...'")

        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, image],
            config=self.types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=self.types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                ),
            ),
        )

        for part in response.parts:
            if part.inline_data is not None:
                # Extract image bytes and convert to PIL Image
                image_bytes = part.inline_data.data
                edited_image = Image.open(BytesIO(image_bytes))
                print(f"Edited image: {edited_image.size}")
                return edited_image

        raise ValueError("No image generated in response")


def load_and_resize_image(path: str, target_size: tuple) -> Image.Image:
    """Load an image and resize to target dimensions."""
    img = Image.open(path).convert("RGB")
    return img.resize(target_size, Image.Resampling.LANCZOS)


def main():
    print("=" * 80)
    print("  Nano Banana + LongLive VACE FFLF Demo")
    print("=" * 80)

    config = CONFIG

    # Setup paths
    script_dir = Path(__file__).parent
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (script_dir / output_dir).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)

    target_size = (config["width"], config["height"])

    print("\nConfiguration:")
    print(f"  Model: {config['model']}")
    print(f"  Resolution: {config['width']}x{config['height']}")
    print(f"  Chunks: {config['num_chunks']} x {config['frames_per_chunk']} frames")
    print(f"  Output: {output_dir}")

    # ===== STEP 1: Generate or load start/end frames =====
    print("\n" + "=" * 40)
    print("  Step 1: Prepare Start/End Frames")
    print("=" * 40)

    first_frame_path = output_dir / "first_frame.png"
    last_frame_path = output_dir / "last_frame.png"

    if config["use_existing_first_frame"] and config["use_existing_last_frame"]:
        print("\nUsing existing images...")
        first_frame = load_and_resize_image(
            config["use_existing_first_frame"], target_size
        )
        last_frame = load_and_resize_image(
            config["use_existing_last_frame"], target_size
        )
    else:
        print("\nInitializing Nano Banana client...")
        nano_banana = NanoBananaClient(
            api_key=config["api_key"],
            model=config["model"],
        )

        # Generate first frame
        print("\n--- Generating First Frame ---")
        start_time = time.time()
        first_frame = nano_banana.generate_image(
            prompt=config["first_frame_prompt"],
            aspect_ratio="1:1",
        )
        first_frame = first_frame.resize(target_size, Image.Resampling.LANCZOS)
        print(f"First frame generated in {time.time() - start_time:.2f}s")

        # Generate last frame
        print("\n--- Generating Last Frame ---")
        start_time = time.time()
        last_frame = nano_banana.generate_image(
            prompt=config["last_frame_prompt"],
            aspect_ratio="1:1",
        )
        last_frame = last_frame.resize(target_size, Image.Resampling.LANCZOS)
        print(f"Last frame generated in {time.time() - start_time:.2f}s")

    # Save generated frames
    if config["save_generated_frames"]:
        first_frame.save(first_frame_path)
        last_frame.save(last_frame_path)
        print("\nSaved frames:")
        print(f"  First: {first_frame_path}")
        print(f"  Last: {last_frame_path}")

    # ===== STEP 2: Initialize LongLive Pipeline =====
    print("\n" + "=" * 40)
    print("  Step 2: Initialize LongLive Pipeline")
    print("=" * 40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    vace_path = str(
        get_model_file_path("Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors")
    )

    pipeline_config = OmegaConf.create(
        {
            "model_dir": str(get_models_dir()),
            "generator_path": str(
                get_model_file_path("LongLive-1.3B/models/longlive_base.pt")
            ),
            "lora_path": str(get_model_file_path("LongLive-1.3B/models/lora.pt")),
            "vace_path": vace_path,
            "text_encoder_path": str(
                get_model_file_path(
                    "WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
                )
            ),
            "tokenizer_path": str(
                get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")
            ),
            "model_config": OmegaConf.load(script_dir / "model.yaml"),
            "height": config["height"],
            "width": config["width"],
            "vae_type": config["vae_type"],
        }
    )

    # Set vace_in_dim for extension mode (masked encoding: 32 + 64 = 96 channels)
    pipeline_config.model_config.base_model_kwargs = (
        pipeline_config.model_config.base_model_kwargs or {}
    )
    pipeline_config.model_config.base_model_kwargs["vace_in_dim"] = 96

    print("Loading pipeline...")
    pipeline = LongLivePipeline(pipeline_config, device=device, dtype=torch.bfloat16)
    print("Pipeline ready!")

    # ===== STEP 3: Generate Video with FFLF =====
    print("\n" + "=" * 40)
    print("  Step 3: Generate Video (FFLF Mode)")
    print("=" * 40)

    print(f"\nVideo prompt: '{config['video_prompt']}'")
    print("Extension mode: firstlastframe")

    outputs = []
    latency_measures = []
    is_first_chunk = True

    for chunk_index in range(config["num_chunks"]):
        start_time = time.time()
        is_last_chunk = chunk_index == config["num_chunks"] - 1

        kwargs = {
            "prompts": [{"text": config["video_prompt"], "weight": 100}],
            "vace_context_scale": config["vace_context_scale"],
        }

        # Apply extension mode for first and last chunks
        if is_first_chunk or is_last_chunk:
            kwargs["extension_mode"] = "firstlastframe"
            kwargs["first_frame_image"] = str(first_frame_path)
            kwargs["last_frame_image"] = str(last_frame_path)

            chunk_type = "first" if is_first_chunk else "last"
            print(f"\nChunk {chunk_index} ({chunk_type}): Applying FFLF extension mode")
        else:
            print(f"\nChunk {chunk_index} (middle): Standard generation")

        # Generate chunk
        output = pipeline(**kwargs)

        # Metrics
        num_frames = output.shape[0]
        latency = time.time() - start_time
        fps = num_frames / latency

        print(
            f"Chunk {chunk_index}: {num_frames} frames, {latency:.2f}s, {fps:.2f} fps"
        )

        latency_measures.append(latency)
        outputs.append(output.detach().cpu())
        is_first_chunk = False

    # Concatenate outputs
    output_video = torch.concat(outputs)
    print(f"\nFinal output shape: {output_video.shape}")

    # ===== STEP 4: Save Output =====
    print("\n" + "=" * 40)
    print("  Step 4: Save Output")
    print("=" * 40)

    output_video_np = output_video.contiguous().numpy()
    output_video_np = np.clip(output_video_np, 0.0, 1.0)

    output_path = output_dir / "nano_banana_fflf_output.mp4"
    export_to_video(output_video_np, output_path, fps=16)
    print(f"\nSaved video: {output_path}")

    # Performance summary
    print("\n" + "=" * 40)
    print("  Performance Summary")
    print("=" * 40)
    total_latency = sum(latency_measures)
    total_frames = output_video.shape[0]
    print(f"Total frames: {total_frames}")
    print(f"Total time: {total_latency:.2f}s")
    print(f"Overall FPS: {total_frames / total_latency:.2f}")

    print("\n" + "=" * 80)
    print("  Demo Complete!")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}")
    print("  - first_frame.png: Nano Banana generated start frame")
    print("  - last_frame.png: Nano Banana generated end frame")
    print("  - nano_banana_fflf_output.mp4: Final interpolated video")


# ===== BONUS: Interactive Frame Editing Demo =====


def demo_frame_editing():
    """
    Demonstrates using Nano Banana to edit frames mid-stream.

    This could be used to:
    1. Take a frame from the generated video
    2. Edit it with Nano Banana (e.g., "add sunglasses", "change hair color")
    3. Use the edited frame as a new keyframe for continued generation
    """
    print("\n" + "=" * 80)
    print("  Nano Banana Frame Editing Demo")
    print("=" * 80)

    config = CONFIG

    nano_banana = NanoBananaClient(
        api_key=config["api_key"],
        model=config["model"],
    )

    # Example: Load an existing frame and edit it
    output_dir = Path(config["output_dir"])
    first_frame_path = output_dir / "first_frame.png"

    if not first_frame_path.exists():
        print("Run main() first to generate frames")
        return

    original = Image.open(first_frame_path)

    # Edit the frame
    edit_prompts = [
        "Add stylish sunglasses to the woman",
        "Change the background to a beach at sunset",
        "Add magical sparkles floating around her",
    ]

    for i, edit_prompt in enumerate(edit_prompts):
        print(f"\n--- Edit {i+1}: {edit_prompt} ---")
        edited = nano_banana.edit_image(original, edit_prompt)

        edit_path = output_dir / f"edited_frame_{i+1}.png"
        edited.save(edit_path)
        print(f"Saved: {edit_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--edit":
        demo_frame_editing()
    else:
        main()
