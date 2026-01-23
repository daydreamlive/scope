"""
VACE Blog Post Visual Assets Generator - Orchestration Script

Generates all visual assets for the VACE blog post by running existing test scripts
with appropriate configurations. All outputs go to a central location.

Output structure:
  blog_assets/
    preprocessing/     - depth_maps.mp4, scribble_maps.mp4, person_masks.mp4
    images/            - Generated reference images
    first_frame_conditioning/  - basic_i2v.mp4, scene_change.mp4
    first_last_frame/  - transition.mp4
    depth/             - depth_control.mp4
    scribble/          - scribble_control.mp4
    inpainting/        - character_transform.mp4, regional_lora.mp4
    layout_control/    - trajectory.mp4

Usage:
    uv run python -m scope.core.pipelines.longlive.vace_tests.generate_blog_assets
    uv run python -m scope.core.pipelines.longlive.vace_tests.generate_blog_assets --only layout
    uv run python -m scope.core.pipelines.longlive.vace_tests.generate_blog_assets --only depth
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from scope.core.config import get_model_file_path, get_models_dir
from scope.server.download_models import download_models

# ============================= CONFIGURATION =============================


def load_dotenv():
    """Load environment variables from .env file."""
    script_dir = Path(__file__).parent
    env_path = script_dir.parent.parent.parent.parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()


# Central output directory (at project root level)
OUTPUT_DIR = Path(__file__).parent.parent.parent.parent.parent.parent / "blog_assets"

CONFIG = {
    # === Global Settings ===
    "width": 368,
    "height": 640,
    "fps": 16,
    "frames_per_chunk": 12,
    "input_video": "controlnet_test/businesslady.mp4",
    "lora_input_video": "controlnet_test/pexels_woman_basketball.mp4",
    # === Chunk counts for each demo ===
    "i2v_num_chunks": 10,
    "depth_num_chunks": 10,
    "scribble_num_chunks": 10,
    "inpainting_num_chunks": 10,
    "lora_num_chunks": 9,
    "layout_num_chunks": 12,
    # === VACE Context Scale for each demo ===
    "i2v_vace_scale": 1.0,
    "depth_vace_scale": 0.80,
    "scribble_vace_scale": 0.90,
    "scribble2_vace_scale": 0.50,
    "inpainting_vace_scale": 0.99,
    "layout_vace_scale": 0.70,
    "lora_vace_scale": 0.70,
    # === LoRA Paths ===
    "highresfix_lora_path": "C:/Users/ryanf/.daydream-scope/models/lora/lora/wan1.3b/Wan2.1-1.3b-lora-highresfix-v1_new.safetensors",
    "toy_soldier_lora_path": "C:/Users/ryanf/.daydream-scope/models/lora/lora/wan1.3b/Studio_Ghibli_LORA_1_3B.safetensors",  # "C:/Users/ryanf/.daydream-scope/models/lora/lora/wan1.3b/ggg-3000-adapter-toy-soldier.safetensors",
    "scribble_lora_path": "C:/Users/ryanf/.daydream-scope/models/lora/lora/wan1.3b/Studio_Ghibli_LORA_1_3B.safetensors",
    # === Enable Highresfix LoRA per test ===
    "i2v_use_highresfix": True,
    "depth_use_highresfix": True,
    "scribble_use_highresfix": True,
    "inpainting_use_highresfix": True,
    "layout_use_highresfix": False,
    "lora_use_highresfix": True,
    # === Inpainting Options ===
    "inpainting_invert_mask": True,
    # === Prompts ===
    "i2v_image_prompt": "A  a waterfall in a lush forest",
    "i2v_video_prompt": "A  a waterfall in a lush forest, the water is flowing fast and bubbling",
    "depth_prompt": "a samurai with a hooded cloak reading a book, standing in front of a castle, cinematic lighting",
    "scribble_prompt": "a woman looking at her phone, studio ghibli style illustration",
    "inpainting_prompt": "a panda with floppy ears holding a cellphone, looking at it",
    "inpainting_background_prompt": "a beautiful woman standing in a lush garden, bathed in brilliant golden hour sunlight, bright warm lighting, sun rays streaming through leaves, radiant amber glow, vibrant colors, high key lighting",
    "inpainting_lora_prompt": "an anime style woman holding a basketball, sexy studio ghibli style",
}


# ============================= UTILITIES =============================


def save_video(frames: np.ndarray, path: Path, fps: int = 16) -> None:
    """Save numpy array as video."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if frames.max() > 1.0:
        frames = frames / 255.0
    frames = np.clip(frames, 0.0, 1.0)
    export_to_video(frames, str(path), fps=fps)
    print(f"  Saved: {path}")


def load_video(path: Path, target_height: int, target_width: int) -> np.ndarray:
    """Load video and resize to target dimensions."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_width, target_height))
        frames.append(frame)
    cap.release()
    return np.array(frames)


# ============================= VIDEO COMPOSITION =============================


def add_fps_overlay(
    frames: np.ndarray,
    fps: float,
    chunk_idx: int,
    total_chunks: int,
) -> np.ndarray:
    """
    Add FPS overlay to frames with semi-transparent background.

    Format: "FPS: XX.X | Chunk N/M"
    Position: top-left corner
    Style: semi-transparent dark background, white text

    Args:
        frames: Video frames [F, H, W, C] in [0, 255] uint8
        fps: Generation FPS for this chunk
        chunk_idx: Current chunk index (0-based)
        total_chunks: Total number of chunks

    Returns:
        Frames with FPS overlay burned in
    """
    frames = frames.copy()
    text = f"{fps:.1f} FPS | Chunk {chunk_idx + 1}/{total_chunks}"

    font = cv2.FONT_HERSHEY_DUPLEX  # Cleaner than SIMPLEX
    font_scale = 0.6
    thickness = 1
    color = (255, 255, 255)  # White text

    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Position: top-left with padding
    padding = 8
    x, y = padding, text_h + padding + 2

    for i in range(len(frames)):
        frame = frames[i]

        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (padding - 4, padding - 2),
            (x + text_w + 4, y + baseline + 4),
            (0, 0, 0),
            -1,
        )
        # Blend with original (0.7 = 70% overlay opacity)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Draw text
        cv2.putText(
            frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA
        )
        frames[i] = frame

    return frames


def add_label_bar(
    frame: np.ndarray,
    text: str,
    position: str = "bottom",
    bar_height: int = 28,
) -> np.ndarray:
    """
    Add a label bar to frame (solid background with centered text).

    Args:
        frame: Single frame [H, W, C]
        text: Label text
        position: "top" or "bottom"
        bar_height: Height of label bar in pixels

    Returns:
        Frame with label bar
    """
    h, w = frame.shape[:2]
    frame = frame.copy()

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)

    # Get text size for centering
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = (w - text_w) // 2
    text_y = bar_height - (bar_height - text_h) // 2 - 2

    if position == "top":
        # Draw bar at top
        cv2.rectangle(frame, (0, 0), (w, bar_height), (40, 40, 40), -1)
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    else:
        # Draw bar at bottom
        cv2.rectangle(frame, (0, h - bar_height), (w, h), (40, 40, 40), -1)
        cv2.putText(
            frame,
            text,
            (text_x, h - bar_height + text_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    return frame


def add_text_overlay(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int] = (10, 30),
    font_scale: float = 0.7,
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Add text overlay to a frame.

    Args:
        frame: Image [H, W, C]
        text: Text to overlay
        position: (x, y) position for text
        font_scale: Font scale
        color: Text color (RGB)

    Returns:
        Frame with text overlay
    """
    frame = frame.copy()
    font = cv2.FONT_HERSHEY_DUPLEX
    thickness = 1
    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return frame


def create_comparison_video(
    panels: list[dict],
    output_path: Path,
    fps: float,
    title: str = None,
) -> None:
    """
    Create a side-by-side comparison video from multiple input panels.

    Args:
        panels: List of dicts with keys:
            - 'frames': np.ndarray [F, H, W, C] or single image [H, W, C]
            - 'label': str label for this panel
        output_path: Where to save the output video
        fps: Playback FPS for the output video
        title: Optional title to display at top
    """
    # Determine number of frames (use max across all panels)
    num_frames = 1
    for panel in panels:
        frames = panel["frames"]
        if frames.ndim == 4:
            num_frames = max(num_frames, frames.shape[0])

    # Get panel dimensions (use first panel as reference)
    ref_frames = panels[0]["frames"]
    if ref_frames.ndim == 3:
        panel_h, panel_w = ref_frames.shape[:2]
    else:
        panel_h, panel_w = ref_frames.shape[1:3]

    # Calculate output dimensions
    num_panels = len(panels)
    output_w = panel_w * num_panels
    output_h = panel_h

    # Add space for title if provided
    title_h = 40 if title else 0
    total_h = output_h + title_h

    output_frames = []

    for frame_idx in range(num_frames):
        # Create canvas
        canvas = np.zeros((total_h, output_w, 3), dtype=np.uint8)

        # Add title bar if provided
        if title:
            canvas = add_text_overlay(
                canvas,
                title,
                position=(10, 28),
                font_scale=0.8,
                color=(255, 255, 255),
            )

        # Add each panel
        for panel_idx, panel in enumerate(panels):
            frames = panel["frames"]
            label = panel.get("label", "")

            # Get frame for this index (repeat last frame if needed, or use static image)
            if frames.ndim == 3:
                # Static image
                frame = frames
            else:
                # Video
                idx = min(frame_idx, frames.shape[0] - 1)
                frame = frames[idx]

            # Ensure frame is uint8
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

            # Add label overlay
            if label:
                frame = add_text_overlay(frame, label, position=(10, panel_h - 15))

            # Place in canvas
            x_start = panel_idx * panel_w
            canvas[title_h : title_h + panel_h, x_start : x_start + panel_w] = frame

        output_frames.append(canvas)

    output_frames = np.array(output_frames)
    save_video(output_frames, output_path, fps=int(fps))


def load_image_as_video(
    image_path: Path, num_frames: int, height: int, width: int
) -> np.ndarray:
    """Load an image and repeat it as video frames."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    img_np = np.array(img)
    return np.stack([img_np] * num_frames, axis=0)


# ============================= PREPROCESSING =============================


def generate_depth_maps(video: np.ndarray, device: torch.device) -> np.ndarray:
    """Generate depth maps using VideoDepthAnythingPipeline."""
    from scope.core.pipelines.video_depth_anything.pipeline import (
        VideoDepthAnythingPipeline,
    )

    download_models("video-depth-anything")

    config = OmegaConf.create({"fp32": False, "input_size": 518})
    pipeline = VideoDepthAnythingPipeline(config, device=device, dtype=torch.float16)

    video_list = [torch.from_numpy(frame).unsqueeze(0) for frame in video]
    depth = pipeline(video=video_list)

    return depth.numpy()


def generate_scribble_maps(video: np.ndarray, device: torch.device) -> np.ndarray:
    """Generate scribble maps using ScribblePipeline."""
    from scope.core.pipelines.scribble.temp_reference_pipeline import ScribblePipeline

    config = OmegaConf.create({"input_nc": 3, "output_nc": 1, "n_residual_blocks": 3})
    pipeline = ScribblePipeline(config, device=device, dtype=torch.float16)

    video_list = [torch.from_numpy(frame).unsqueeze(0) for frame in video]
    scribble = pipeline(video=video_list)

    return scribble.numpy()


def generate_person_masks(video: np.ndarray, device: torch.device) -> dict:
    """Generate person masks using YOLO."""
    from ultralytics import YOLO

    models_dir = get_models_dir() / "ultralytics"
    model_path = models_dir / "yolo26n-seg.pt"
    model = YOLO(str(model_path))

    masks_list = []
    h, w = video.shape[1], video.shape[2]

    for frame in video:
        results = model(frame, conf=0.5, classes=[0], verbose=False)
        result = results[0]

        if result.masks is not None and len(result.masks.data) > 0:
            all_masks = result.masks.data.float()
            combined_mask = all_masks.max(dim=0).values.cpu().numpy()
            if combined_mask.shape != (h, w):
                combined_mask = cv2.resize(combined_mask, (w, h))
        else:
            combined_mask = np.zeros((h, w), dtype=np.float32)

        masks_list.append((combined_mask > 0.5).astype(np.float32))

    return np.array(masks_list)


def preprocess_all(output_dir: Path, input_video_path: Path) -> dict:
    """Run all preprocessing steps."""
    print("\n" + "=" * 60)
    print("  PREPROCESSING PHASE")
    print("=" * 60)

    preprocessing_dir = output_dir / "preprocessing"
    preprocessing_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    # Load and resize input video
    print("\n1. Loading and resizing input video...")
    video = load_video(input_video_path, CONFIG["height"], CONFIG["width"])
    print(f"   Loaded and resized: {video.shape}")

    resized_path = preprocessing_dir / "input_video_resized.mp4"
    save_video(video, resized_path, fps=CONFIG["fps"])
    results["input_video"] = video
    results["input_video_path"] = resized_path

    # Generate depth maps
    print("\n2. Generating depth maps...")
    depth_maps = generate_depth_maps(video, device)
    depth_path = preprocessing_dir / "depth_maps.mp4"
    save_video(depth_maps, depth_path, fps=CONFIG["fps"])
    results["depth_maps"] = depth_maps
    results["depth_maps_path"] = depth_path

    # Generate scribble maps
    print("\n3. Generating scribble maps...")
    scribble_maps = generate_scribble_maps(video, device)
    scribble_path = preprocessing_dir / "scribble_maps.mp4"
    save_video(scribble_maps, scribble_path, fps=CONFIG["fps"])
    results["scribble_maps"] = scribble_maps
    results["scribble_maps_path"] = scribble_path

    # Generate person masks
    print("\n4. Generating person masks...")
    masks = generate_person_masks(video, device)
    mask_path = preprocessing_dir / "person_masks.mp4"
    mask_vis = np.stack([masks] * 3, axis=-1)
    save_video(mask_vis, mask_path, fps=CONFIG["fps"])
    results["person_masks"] = masks
    results["person_masks_path"] = mask_path

    print("\nPreprocessing complete!")
    return results


# ============================= IMAGE GENERATION =============================


class NanoBananaClient:
    """Simple client for Gemini image generation."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-image"):
        from google import genai
        from google.genai import types

        self.client = genai.Client(api_key=api_key)
        self.types = types
        self.model = model

    def generate_image(self, prompt: str, width: int = 368, height: int = 640):
        import io

        from PIL import Image

        # Use 9:16 aspect ratio for portrait (height > width)
        aspect_ratio = "9:16" if height > width else "16:9" if width > height else "1:1"

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
                image = Image.open(io.BytesIO(part.inline_data.data))
                # Use high-quality LANCZOS resampling to target size
                return image.resize((width, height), Image.Resampling.LANCZOS)

        raise RuntimeError("No image in response")


def generate_images(output_dir: Path) -> dict:
    """Generate all required images."""
    print("\n" + "=" * 60)
    print("  IMAGE GENERATION PHASE")
    print("=" * 60)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set in environment")

    client = NanoBananaClient(api_key)
    results = {}

    prompts = {
        "first_frame": CONFIG["i2v_image_prompt"],
        "layout_subject": "A three quarter shot of a beautiful young woman, chest-up, three-quarter view (turned slightly to the side), shoulders visible, head and upper torso in frame, natural proportions, shallow depth of field, soft directional lighting, sharp focus on eyes.",
    }

    for name, prompt in prompts.items():
        path = images_dir / f"{name}.png"
        if path.exists():
            print(f"  Using existing: {name}")
            results[name] = path
            continue

        print(f"\n  Generating {name}...")
        print(f"    Prompt: {prompt[:50]}...")
        image = client.generate_image(prompt, CONFIG["width"], CONFIG["height"])
        image.save(path)
        results[name] = path
        print(f"    Saved: {path}")

    return results


# ============================= PIPELINE INITIALIZATION =============================


def create_pipeline(device: torch.device, lora_paths: list[str] = None):
    """Create LongLive pipeline with optional LoRAs.

    Args:
        device: Device to run on
        lora_paths: List of LoRA file paths to load (will be merged permanently)
    """
    from ..pipeline import LongLivePipeline

    vace_path = str(
        get_model_file_path("Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors")
    )

    # Build LoRA list if paths are provided
    loras = []
    if lora_paths:
        for lora_path in lora_paths:
            if lora_path:
                loras.append({"path": lora_path, "scale": 1.0})

    pipeline_config = OmegaConf.create(
        {
            "model_dir": str(get_models_dir()),
            "generator_path": str(
                get_model_file_path("LongLive-1.3B/models/longlive_base.pt")
            ),
            "lora_path": str(get_model_file_path("LongLive-1.3B/models/lora.pt")),
            "loras": loras if loras else [],
            "vace_path": vace_path,
            "text_encoder_path": str(
                get_model_file_path(
                    "WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
                )
            ),
            "tokenizer_path": str(
                get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")
            ),
            "model_config": OmegaConf.load(Path(__file__).parent.parent / "model.yaml"),
            "height": CONFIG["height"],
            "width": CONFIG["width"],
            "vae_type": "wan",
        }
    )

    pipeline_config.model_config.base_model_kwargs = (
        pipeline_config.model_config.base_model_kwargs or {}
    )
    pipeline_config.model_config.base_model_kwargs["vace_in_dim"] = 96

    return LongLivePipeline(pipeline_config, device=device, dtype=torch.bfloat16)


# ============================= ASSET GENERATION =============================


def run_point_based_control_test(images: dict, output_dir: Path):
    """
    Run test_point_based_subject_control.py for layout/trajectory control.
    """
    print("\n" + "=" * 60)
    print("  ASSET: Layout Control (test_point_based_subject_control)")
    print("=" * 60)

    from .. import test_point_based_subject_control

    # Build LoRA list
    loras = []
    if CONFIG["layout_use_highresfix"]:
        loras.append(CONFIG["highresfix_lora_path"])

    test_point_based_subject_control.CONFIG.update(
        {
            "subject_image": str(images["layout_subject"]),
            "num_chunks": CONFIG["layout_num_chunks"],
            "height": CONFIG["height"],
            "width": CONFIG["width"],
            "output_dir": str(output_dir / "layout_control"),
            "trajectory_pattern": "nod",
            "vace_context_scale": CONFIG["layout_vace_scale"],
            "loras": loras,
        }
    )

    test_point_based_subject_control.main()


def run_depth_test(preprocessed: dict, output_dir: Path):
    """Run test_vace.py in depth mode."""
    print("\n" + "=" * 60)
    print("  ASSET: Depth Control (test_vace)")
    print("=" * 60)

    from .. import test_vace

    # Build LoRA list
    loras = []
    if CONFIG["depth_use_highresfix"]:
        loras.append(CONFIG["highresfix_lora_path"])

    test_vace.CONFIG.update(
        {
            "use_r2v": False,
            "use_depth": True,
            "use_inpainting": False,
            "use_extension": False,
            "depth_video": str(preprocessed["depth_maps_path"]),
            "prompt": CONFIG["depth_prompt"],
            "num_chunks": CONFIG["depth_num_chunks"],
            "height": CONFIG["height"],
            "width": CONFIG["width"],
            "output_dir": str(output_dir / "depth"),
            "vace_context_scale": CONFIG["depth_vace_scale"],
            "loras": loras,
        }
    )

    test_vace.main()


def run_scribble_test(
    preprocessed: dict, output_dir: Path, scale: float, suffix: str = ""
):
    """Run scribble control using test_vace.py depth mode with scribble maps.

    Args:
        preprocessed: Preprocessed data containing scribble maps
        output_dir: Base output directory
        scale: VACE context scale to use
        suffix: Optional suffix for output directory (e.g., "2" for scribble2)
    """
    print("\n" + "=" * 60)
    print(f"  ASSET: Scribble Control (scale={scale})")
    print("=" * 60)

    from .. import test_vace

    scribble_dir = output_dir / f"scribble{suffix}"

    # Build LoRA list (scribble LoRA + optional highresfix)
    loras = [CONFIG["scribble_lora_path"]]
    if CONFIG["scribble_use_highresfix"]:
        loras.append(CONFIG["highresfix_lora_path"])

    # Scribble uses same interface as depth
    test_vace.CONFIG.update(
        {
            "use_r2v": False,
            "use_depth": True,  # Scribble uses depth interface
            "use_inpainting": False,
            "use_extension": False,
            "depth_video": str(preprocessed["scribble_maps_path"]),
            "prompt": CONFIG["scribble_prompt"],
            "num_chunks": CONFIG["scribble_num_chunks"],
            "height": CONFIG["height"],
            "width": CONFIG["width"],
            "output_dir": str(scribble_dir),
            "vace_context_scale": scale,
            "loras": loras,
        }
    )

    test_vace.main()

    # Rename output files from depth to scribble naming
    import shutil

    if (scribble_dir / "output_depth.mp4").exists():
        shutil.move(
            scribble_dir / "output_depth.mp4", scribble_dir / "output_scribble.mp4"
        )
        print("  Renamed: output_depth.mp4 -> output_scribble.mp4")
    if (scribble_dir / "depth_maps.mp4").exists():
        shutil.move(scribble_dir / "depth_maps.mp4", scribble_dir / "scribble_maps.mp4")
        print("  Renamed: depth_maps.mp4 -> scribble_maps.mp4")


def run_inpainting_test(preprocessed: dict, output_dir: Path):
    """Run inpainting test using test_vace.py."""
    mode = "Background" if CONFIG["inpainting_invert_mask"] else "Character"
    print("\n" + "=" * 60)
    print(f"  ASSET: Inpainting - {mode} Transformation")
    print("=" * 60)

    from .. import test_vace

    # Build LoRA list
    loras = []
    if CONFIG["inpainting_use_highresfix"]:
        loras.append(CONFIG["highresfix_lora_path"])

    # Handle mask inversion if requested
    mask_video_path = preprocessed["person_masks_path"]
    prompt = CONFIG["inpainting_prompt"]
    if CONFIG["inpainting_invert_mask"]:
        print("  Inverting mask for background inpainting...")
        inverted_mask_path = OUTPUT_DIR / "preprocessing" / "person_masks_inverted.mp4"
        if not inverted_mask_path.exists():
            masks = preprocessed["person_masks"]
            inverted_masks = 1.0 - masks
            mask_vis = np.stack([inverted_masks] * 3, axis=-1)
            save_video(mask_vis, inverted_mask_path, fps=CONFIG["fps"])
        mask_video_path = inverted_mask_path
        prompt = CONFIG["inpainting_background_prompt"]

    test_vace.CONFIG.update(
        {
            "use_r2v": False,
            "use_depth": False,
            "use_inpainting": True,
            "use_extension": False,
            "input_video": str(preprocessed["input_video_path"]),
            "mask_video": str(mask_video_path),
            "prompt": prompt,
            "num_chunks": CONFIG["inpainting_num_chunks"],
            "height": CONFIG["height"],
            "width": CONFIG["width"],
            "output_dir": str(output_dir / "inpainting"),
            "vace_context_scale": CONFIG["inpainting_vace_scale"],
            "loras": loras,
        }
    )

    test_vace.main()


def run_i2v_test(images: dict, output_dir: Path):
    """Run basic I2V test using test_vace.py extension mode."""
    print("\n" + "=" * 60)
    print("  ASSET: First Frame I2V")
    print("=" * 60)

    from .. import test_vace

    # Build LoRA list
    loras = []
    if CONFIG["i2v_use_highresfix"]:
        loras.append(CONFIG["highresfix_lora_path"])

    test_vace.CONFIG.update(
        {
            "use_r2v": False,
            "use_depth": False,
            "use_inpainting": False,
            "use_extension": True,
            "extension_mode": "firstframe",
            "first_frame_image": str(images["first_frame"]),
            "prompt": CONFIG["i2v_video_prompt"],
            "num_chunks": CONFIG["i2v_num_chunks"],
            "height": CONFIG["height"],
            "width": CONFIG["width"],
            "output_dir": str(output_dir / "first_frame_conditioning"),
            "vace_context_scale": CONFIG["i2v_vace_scale"],
            "loras": loras,
        }
    )

    test_vace.main()


def run_inpainting_lora_test(
    preprocessed: dict, output_dir: Path, video_name: str = "businesslady"
):
    """Run inpainting with LoRA.

    Args:
        preprocessed: Dict with input_video and person_masks
        output_dir: Base output directory
        video_name: Name identifier for the input video (for output naming)
    """
    import time

    print("\n" + "=" * 60)
    print("  ASSET: Inpainting - Regional LoRA (Toy Soldier)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build LoRA list
    lora_paths = [CONFIG["toy_soldier_lora_path"]]
    if CONFIG["lora_use_highresfix"]:
        lora_paths.append(CONFIG["highresfix_lora_path"])

    # Create pipeline with LoRAs
    pipeline = create_pipeline(device, lora_paths=lora_paths)

    asset_dir = output_dir / "inpainting"
    asset_dir.mkdir(parents=True, exist_ok=True)

    # Load preprocessed data
    video = preprocessed["input_video"]
    masks = preprocessed["person_masks"]

    num_chunks = CONFIG["lora_num_chunks"]
    frames_per_chunk = CONFIG["frames_per_chunk"]

    # Prepare VACE tensors
    video_tensor = torch.from_numpy(video).float() / 255.0
    video_tensor = video_tensor * 2.0 - 1.0
    video_tensor = (
        video_tensor.permute(0, 3, 1, 2).unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
    )

    masks_tensor = torch.from_numpy(masks).float()
    masks_tensor = (
        masks_tensor.unsqueeze(1).unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
    )

    # Apply mask
    masks_expanded = masks_tensor.expand_as(video_tensor)
    vace_input = torch.where(
        masks_expanded > 0.5, torch.tensor(0.0, device=device), video_tensor
    )

    outputs = []

    for chunk_idx in range(num_chunks):
        start_time = time.time()

        start_frame = chunk_idx * frames_per_chunk
        end_frame = min(start_frame + frames_per_chunk, video.shape[0])

        input_chunk = vace_input[:, :, start_frame:end_frame, :, :]
        mask_chunk = masks_tensor[:, :, start_frame:end_frame, :, :]

        kwargs = {
            "prompts": [{"text": CONFIG["inpainting_lora_prompt"], "weight": 100}],
            "vace_input_frames": input_chunk,
            "vace_input_masks": mask_chunk,
            "vace_context_scale": CONFIG["lora_vace_scale"],
        }

        output = pipeline(**kwargs)
        latency = time.time() - start_time
        fps = output.shape[0] / latency

        print(
            f"    Chunk {chunk_idx}: {output.shape[0]} frames, {latency:.1f}s, {fps:.1f} fps"
        )

        # Convert to numpy uint8 and add FPS overlay
        output_np = output.detach().cpu().numpy()
        output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
        output_np = add_fps_overlay(output_np, fps, chunk_idx, num_chunks)
        outputs.append(output_np)

    output_video = np.concatenate(outputs, axis=0)
    save_video(output_video, asset_dir / "regional_lora.mp4", fps=CONFIG["fps"])


# ============================= MAIN =============================


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate VACE blog post visual assets"
    )
    parser.add_argument(
        "--only",
        type=str,
        choices=[
            "layout",
            "depth",
            "scribble",
            "scribble2",
            "inpainting",
            "i2v",
            "lora",
        ],
        help="Generate only a specific asset type",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("  VACE Blog Post Visual Assets Generator")
    print("=" * 80)

    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent.parent.parent.parent
    input_video_path = project_root / CONFIG["input_video"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nConfiguration:")
    print(f"  Resolution: {CONFIG['width']}x{CONFIG['height']}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Input video: {input_video_path}")
    if args.only:
        print(f"  Running only: {args.only}")

    if not input_video_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Preprocessing (always check for all files)
    preprocessing_complete = all(
        (OUTPUT_DIR / "preprocessing" / f).exists()
        for f in [
            "input_video_resized.mp4",
            "depth_maps.mp4",
            "scribble_maps.mp4",
            "person_masks.mp4",
        ]
    )
    if preprocessing_complete:
        print("\n[SKIP] Preprocessing already complete, loading existing data...")
        # Load the input video for inpainting tests
        input_video = load_video(
            OUTPUT_DIR / "preprocessing" / "input_video_resized.mp4",
            CONFIG["height"],
            CONFIG["width"],
        )
        # Load masks for inpainting tests
        mask_video = load_video(
            OUTPUT_DIR / "preprocessing" / "person_masks.mp4",
            CONFIG["height"],
            CONFIG["width"],
        )
        # Convert mask video to single channel
        person_masks = np.mean(mask_video, axis=-1) / 255.0
        person_masks = (person_masks > 0.5).astype(np.float32)

        preprocessed = {
            "input_video": input_video,
            "person_masks": person_masks,
            "input_video_path": OUTPUT_DIR
            / "preprocessing"
            / "input_video_resized.mp4",
            "depth_maps_path": OUTPUT_DIR / "preprocessing" / "depth_maps.mp4",
            "scribble_maps_path": OUTPUT_DIR / "preprocessing" / "scribble_maps.mp4",
            "person_masks_path": OUTPUT_DIR / "preprocessing" / "person_masks.mp4",
        }
    else:
        preprocessed = preprocess_all(OUTPUT_DIR, input_video_path)

    # 2. Image generation
    images = generate_images(OUTPUT_DIR)

    # 3. Run test scripts for each asset (output videos will be overwritten)

    # Layout Control
    if args.only is None or args.only == "layout":
        run_point_based_control_test(images, OUTPUT_DIR)

    # Depth Control
    if args.only is None or args.only == "depth":
        run_depth_test(preprocessed, OUTPUT_DIR)

    # Scribble Control (higher scale - more adherence)
    if args.only is None or args.only == "scribble":
        run_scribble_test(preprocessed, OUTPUT_DIR, CONFIG["scribble_vace_scale"], "")

    # Scribble Control 2 (lower scale - more freedom)
    if args.only is None or args.only == "scribble2":
        run_scribble_test(preprocessed, OUTPUT_DIR, CONFIG["scribble2_vace_scale"], "2")

    # Inpainting - Character Transformation
    if args.only is None or args.only == "inpainting":
        run_inpainting_test(preprocessed, OUTPUT_DIR)

    # First Frame I2V
    if args.only is None or args.only == "i2v":
        run_i2v_test(images, OUTPUT_DIR)

    # Inpainting with LoRA
    if args.only is None or args.only == "lora":
        # Check if we need to use a different input video for lora
        if CONFIG["lora_input_video"] != CONFIG["input_video"]:
            print("\n[INFO] Regional LoRA test using different input video...")
            lora_video_path = project_root / CONFIG["lora_input_video"]
            if not lora_video_path.exists():
                raise FileNotFoundError(
                    f"LoRA input video not found: {lora_video_path}"
                )

            # Check if preprocessing exists for lora video
            lora_preprocessing_dir = OUTPUT_DIR / "preprocessing_lora"
            lora_preprocessing_complete = all(
                (lora_preprocessing_dir / f).exists()
                for f in ["input_video_resized.mp4", "person_masks.mp4"]
            )

            if lora_preprocessing_complete:
                print(
                    "[SKIP] LoRA preprocessing already complete, loading existing data..."
                )
                lora_input_video = load_video(
                    lora_preprocessing_dir / "input_video_resized.mp4",
                    CONFIG["height"],
                    CONFIG["width"],
                )
                lora_mask_video = load_video(
                    lora_preprocessing_dir / "person_masks.mp4",
                    CONFIG["height"],
                    CONFIG["width"],
                )
                lora_person_masks = np.mean(lora_mask_video, axis=-1) / 255.0
                lora_person_masks = (lora_person_masks > 0.5).astype(np.float32)

                lora_preprocessed = {
                    "input_video": lora_input_video,
                    "person_masks": lora_person_masks,
                }
            else:
                print("Preprocessing LoRA input video...")
                lora_preprocessing_dir.mkdir(parents=True, exist_ok=True)

                # Load and resize
                lora_video = load_video(
                    lora_video_path, CONFIG["height"], CONFIG["width"]
                )
                resized_path = lora_preprocessing_dir / "input_video_resized.mp4"
                save_video(lora_video, resized_path, fps=CONFIG["fps"])

                # Generate person masks
                lora_masks = generate_person_masks(lora_video, device)
                mask_path = lora_preprocessing_dir / "person_masks.mp4"
                mask_vis = np.stack([lora_masks] * 3, axis=-1)
                save_video(mask_vis, mask_path, fps=CONFIG["fps"])

                lora_preprocessed = {
                    "input_video": lora_video,
                    "person_masks": lora_masks,
                }

            video_name = Path(CONFIG["lora_input_video"]).stem
            run_inpainting_lora_test(lora_preprocessed, OUTPUT_DIR, video_name)
        else:
            run_inpainting_lora_test(preprocessed, OUTPUT_DIR)

    # 4. Post-processing: Create comparison videos
    print("\n" + "=" * 80)
    print("  POST-PROCESSING: Creating Comparison Videos")
    print("=" * 80)

    h, w = CONFIG["height"], CONFIG["width"]
    fps = CONFIG["fps"]

    # Note: FPS overlay is now burned into videos during generation per-chunk
    # No need for estimated gen_fps in comparison video titles

    # Load original input video for comparisons
    input_video_resized = OUTPUT_DIR / "preprocessing" / "input_video_resized.mp4"
    original_input_vid = None
    if input_video_resized.exists():
        original_input_vid = load_video(input_video_resized, h, w)

    # First Frame I2V comparison
    i2v_output = (
        OUTPUT_DIR / "first_frame_conditioning" / "output_extension_firstframe.mp4"
    )
    if i2v_output.exists():
        print("\n  Creating: First Frame I2V comparison...")
        first_frame_vid = load_image_as_video(images["first_frame"], 60, h, w)
        output_vid = load_video(i2v_output, h, w)
        create_comparison_video(
            panels=[
                {"frames": first_frame_vid, "label": "Input"},
                {"frames": output_vid, "label": "Output"},
            ],
            output_path=OUTPUT_DIR / "first_frame_conditioning" / "i2v_comparison.mp4",
            fps=fps,
        )

    # Depth Control comparison
    depth_output = OUTPUT_DIR / "depth" / "output_depth.mp4"
    depth_input = OUTPUT_DIR / "preprocessing" / "depth_maps.mp4"
    if depth_output.exists() and depth_input.exists():
        print("\n  Creating: Depth Control comparison...")
        depth_vid = load_video(depth_input, h, w)
        output_vid = load_video(depth_output, h, w)
        # Match frame counts
        min_frames = min(depth_vid.shape[0], output_vid.shape[0])
        if original_input_vid is not None:
            min_frames = min(min_frames, original_input_vid.shape[0])
        panels = []
        if original_input_vid is not None:
            panels.append({"frames": original_input_vid[:min_frames], "label": "Input"})
        panels.extend(
            [
                {"frames": depth_vid[:min_frames], "label": "Depth"},
                {"frames": output_vid[:min_frames], "label": "Output"},
            ]
        )
        create_comparison_video(
            panels=panels,
            output_path=OUTPUT_DIR / "depth" / "depth_comparison.mp4",
            fps=fps,
        )

    # Scribble Control comparison (scale=0.9)
    scribble_output = OUTPUT_DIR / "scribble" / "output_scribble.mp4"
    scribble_input = OUTPUT_DIR / "preprocessing" / "scribble_maps.mp4"
    if scribble_output.exists() and scribble_input.exists():
        print("\n  Creating: Scribble Control comparison...")
        scribble_vid = load_video(scribble_input, h, w)
        output_vid = load_video(scribble_output, h, w)
        min_frames = min(scribble_vid.shape[0], output_vid.shape[0])
        if original_input_vid is not None:
            min_frames = min(min_frames, original_input_vid.shape[0])
        panels = []
        if original_input_vid is not None:
            panels.append({"frames": original_input_vid[:min_frames], "label": "Input"})
        panels.extend(
            [
                {"frames": scribble_vid[:min_frames], "label": "Scribble"},
                {"frames": output_vid[:min_frames], "label": "Output"},
            ]
        )
        create_comparison_video(
            panels=panels,
            output_path=OUTPUT_DIR / "scribble" / "scribble_comparison.mp4",
            fps=fps,
        )

    # Scribble Control 2 comparison (scale=0.5)
    scribble2_output = OUTPUT_DIR / "scribble2" / "output_scribble.mp4"
    if scribble2_output.exists() and scribble_input.exists():
        print("\n  Creating: Scribble Control 2 comparison...")
        scribble_vid = load_video(scribble_input, h, w)
        output_vid = load_video(scribble2_output, h, w)
        min_frames = min(scribble_vid.shape[0], output_vid.shape[0])
        if original_input_vid is not None:
            min_frames = min(min_frames, original_input_vid.shape[0])
        panels = []
        if original_input_vid is not None:
            panels.append({"frames": original_input_vid[:min_frames], "label": "Input"})
        panels.extend(
            [
                {"frames": scribble_vid[:min_frames], "label": "Scribble"},
                {"frames": output_vid[:min_frames], "label": "Output"},
            ]
        )
        create_comparison_video(
            panels=panels,
            output_path=OUTPUT_DIR / "scribble2" / "scribble_comparison.mp4",
            fps=fps,
        )

    # Inpainting comparison
    inpaint_output = OUTPUT_DIR / "inpainting" / "output_inpainting.mp4"
    inpaint_original = OUTPUT_DIR / "inpainting" / "input_original.mp4"
    inpaint_mask = OUTPUT_DIR / "inpainting" / "mask_visualization.mp4"
    if inpaint_output.exists() and inpaint_original.exists():
        print("\n  Creating: Inpainting comparison...")
        original_vid = load_video(inpaint_original, h, w)
        output_vid = load_video(inpaint_output, h, w)
        min_frames = min(original_vid.shape[0], output_vid.shape[0])
        panels = [
            {"frames": original_vid[:min_frames], "label": "Input"},
        ]
        if inpaint_mask.exists():
            mask_vid = load_video(inpaint_mask, h, w)
            panels.append({"frames": mask_vid[:min_frames], "label": "Mask"})
        panels.append({"frames": output_vid[:min_frames], "label": "Output"})
        create_comparison_video(
            panels=panels,
            output_path=OUTPUT_DIR / "inpainting" / "inpainting_comparison.mp4",
            fps=fps,
        )

    # Inpainting + LoRA comparison
    lora_output = OUTPUT_DIR / "inpainting" / "regional_lora.mp4"
    if lora_output.exists():
        print("\n  Creating: Inpainting + LoRA comparison...")
        output_vid = load_video(lora_output, h, w)

        # Use lora-specific input if different video was used
        if CONFIG["lora_input_video"] != CONFIG["input_video"]:
            lora_preprocessing_dir = OUTPUT_DIR / "preprocessing_lora"
            lora_input_resized = lora_preprocessing_dir / "input_video_resized.mp4"
            lora_mask_path = lora_preprocessing_dir / "person_masks.mp4"

            if lora_input_resized.exists():
                original_vid = load_video(lora_input_resized, h, w)
                min_frames = min(original_vid.shape[0], output_vid.shape[0])
                panels = [{"frames": original_vid[:min_frames], "label": "Input"}]

                if lora_mask_path.exists():
                    mask_vid = load_video(lora_mask_path, h, w)
                    panels.append({"frames": mask_vid[:min_frames], "label": "Mask"})

                panels.append({"frames": output_vid[:min_frames], "label": "Output"})
                create_comparison_video(
                    panels=panels,
                    output_path=OUTPUT_DIR / "inpainting" / "lora_comparison.mp4",
                    fps=fps,
                )
        elif inpaint_original.exists():
            # Use regular inpainting input
            original_vid = load_video(inpaint_original, h, w)
            min_frames = min(original_vid.shape[0], output_vid.shape[0])
            panels = [{"frames": original_vid[:min_frames], "label": "Input"}]

            if inpaint_mask.exists():
                mask_vid = load_video(inpaint_mask, h, w)
                panels.append({"frames": mask_vid[:min_frames], "label": "Mask"})

            panels.append({"frames": output_vid[:min_frames], "label": "Output"})
            create_comparison_video(
                panels=panels,
                output_path=OUTPUT_DIR / "inpainting" / "lora_comparison.mp4",
                fps=fps,
            )

    # Layout Control comparison
    layout_output = OUTPUT_DIR / "layout_control" / "output_nod.mp4"
    layout_control = OUTPUT_DIR / "layout_control" / "layout_control.mp4"
    if layout_output.exists():
        print("\n  Creating: Layout Control comparison...")
        output_vid = load_video(layout_output, h, w)
        num_frames = output_vid.shape[0]
        subject_vid = load_image_as_video(images["layout_subject"], num_frames, h, w)
        panels = [
            {"frames": subject_vid, "label": "Subject"},
        ]
        if layout_control.exists():
            control_vid = load_video(layout_control, h, w)
            panels.append({"frames": control_vid[:num_frames], "label": "Control"})
        panels.append({"frames": output_vid, "label": "Output"})
        create_comparison_video(
            panels=panels,
            output_path=OUTPUT_DIR / "layout_control" / "layout_comparison.mp4",
            fps=fps,
        )

    print("\n" + "=" * 80)
    print("  GENERATION COMPLETE")
    print("=" * 80)
    print(f"\n  All outputs saved to: {OUTPUT_DIR}")

    # List generated files
    print("\n  Generated assets:")
    for subdir in sorted(OUTPUT_DIR.iterdir()):
        if subdir.is_dir():
            for f in sorted(subdir.glob("*.mp4")):
                print(f"    - {f.relative_to(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
