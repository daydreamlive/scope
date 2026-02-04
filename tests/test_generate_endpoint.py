"""Test script for the /api/v1/generate endpoint.

Usage:
    python test_generate_endpoint.py <test_name>
    python test_generate_endpoint.py --list
"""

import base64
import json
import sys
import time

import numpy as np
import requests
from diffusers.utils import export_to_video

from scope.core.pipelines.video import load_video
from scope.server.schema import (
    GenerateRequest,
    LoRAConfig,
    LoRAMergeMode,
    PipelineLoadRequest,
    PipelineStatusResponse,
)

# =============================================================================
# Configuration
# =============================================================================

SERVER_URL = "http://localhost:8000"
DEFAULT_PIPELINE = "longlive"

# Asset paths (tests skip gracefully if missing)
LORA = "path/to/a/lora.safetensors"
TEST_VIDEO = "path/to/test_video.mp4"
VACE_CONDITIONING_VIDEO = "path/to/depth_video.mp4"
MASK_VIDEO = "path/to/mask_video.mp4"

# =============================================================================
# Test Definitions
# =============================================================================

TESTS = {
    "lora": {
        "description": "LoRA strength ramping over chunks",
        "pipeline": "longlive",
        "resolution": (576, 320),
        "num_frames": 96,
        "prompt": "a woman dissolving into particles, ethereal, magical transformation",
        "lora": LORA,
        "lora_ramp": [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0],
        "manage_cache": False,
    },
    "v2v": {
        "description": "Video-to-video transformation",
        "resolution": (512, 512),
        "num_frames": 48,
        "prompt": "A 3D animated scene. A **panda** sitting in the grass, looking around.",
        "input_video": TEST_VIDEO,
        "noise_scale": 0.6,
    },
    "v2v_lora": {
        "description": "Video-to-video with LoRA ramp (0 -> 1.5 -> 0)",
        "resolution": (512, 512),
        "num_frames": 120,
        "prompt": "a woman made of ral-dissolve, dissolving into particles",
        "input_video": TEST_VIDEO,
        "noise_scale": 0.7,
        "lora": LORA,
        "lora_ramp": [0.0, 0.3, 0.6, 1.0, 1.5, 1.5, 1.0, 0.6, 0.3, 0.0],
    },
    "vace_conditioning": {
        "description": "VACE structural conditioning (depth, pose, etc.)",
        "resolution": (576, 320),
        "num_frames": 48,
        "prompt": "a cat walking towards the camera",
        "vace_frames": VACE_CONDITIONING_VIDEO,
        "vace_context_scale": 1.5,
    },
    "inpainting": {
        "description": "VACE inpainting with mask",
        "resolution": (512, 512),
        "num_frames": 48,
        "prompt": "fireball doom flames",
        "vace_frames": TEST_VIDEO,
        "vace_masks": MASK_VIDEO,
    },
}

# =============================================================================
# Helpers
# =============================================================================


def encode_array(arr: np.ndarray) -> dict:
    """Encode numpy array as EncodedArray dict."""
    return {
        "base64": base64.b64encode(arr.tobytes()).decode("utf-8"),
        "shape": list(arr.shape),
    }


def load_video_for_v2v(path: str, height: int, width: int) -> dict:
    """Load video as [T, H, W, C] uint8 for video-to-video mode."""
    tensor = load_video(path, resize_hw=(height, width), normalize=False)
    arr = tensor.permute(1, 2, 3, 0).numpy().astype(np.uint8)
    return encode_array(arr)


def load_video_for_vace(path: str, height: int, width: int) -> dict:
    """Load video as [1, C, T, H, W] float32 for VACE conditioning."""
    tensor = load_video(path, resize_hw=(height, width))
    arr = tensor.unsqueeze(0).numpy().astype(np.float32)
    return encode_array(arr)


def load_mask_for_vace(path: str, height: int, width: int) -> dict:
    """Load video as [1, 1, T, H, W] binary mask for VACE inpainting."""
    tensor = load_video(path, resize_hw=(height, width))
    arr = (tensor[0:1].unsqueeze(0).numpy() > 0.0).astype(np.float32)
    return encode_array(arr)


def parse_sse_events(response):
    """Parse SSE events using iter_content (handles large payloads)."""
    buffer = ""
    event_type = None
    data_lines = []

    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        buffer += chunk
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")

            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
            elif line == "":
                if data_lines:
                    yield (event_type or "message", json.loads("\n".join(data_lines)))
                event_type = None
                data_lines = []


def wait_for_pipeline(timeout: int = 300):
    """Wait for pipeline to finish loading."""
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(f"{SERVER_URL}/api/v1/pipeline/status")
        status = PipelineStatusResponse.model_validate(resp.json())
        if status.status.value == "loaded":
            return time.time() - start
        if status.status.value == "error":
            raise RuntimeError(f"Pipeline failed: {status.error}")
        time.sleep(1)
    raise TimeoutError(f"Pipeline did not load within {timeout}s")


# =============================================================================
# Test Runner
# =============================================================================


def run_test(name: str):
    """Run a single test by name."""
    if name not in TESTS:
        print(f"Unknown test: {name}")
        print(f"Available: {', '.join(TESTS.keys())}")
        return

    cfg = TESTS[name]
    width, height = cfg.get("resolution", (576, 320))
    pipeline_id = cfg.get("pipeline", DEFAULT_PIPELINE)

    print(f"\n{'=' * 60}")
    print(f"Test: {name}")
    print(f"Description: {cfg['description']}")
    print(f"{'=' * 60}")

    # Build LoRA config if specified
    loras = None
    lora_scales = None
    if "lora" in cfg:
        loras = [
            LoRAConfig(
                path=cfg["lora"], scale=0.0, merge_mode=LoRAMergeMode.RUNTIME_PEFT
            )
        ]
        if "lora_ramp" in cfg:
            lora_scales = {cfg["lora"]: cfg["lora_ramp"]}
            print(f"LoRA ramp: {cfg['lora_ramp']}")

    # Load pipeline
    print(f"Loading pipeline '{pipeline_id}' at {width}x{height}...")
    load_params = {"height": height, "width": width}
    if loras:
        load_params["loras"] = [lora.model_dump() for lora in loras]
        load_params["lora_merge_mode"] = "runtime_peft"
    request = PipelineLoadRequest(pipeline_ids=[pipeline_id], load_params=load_params)
    requests.post(
        f"{SERVER_URL}/api/v1/pipeline/load", json=request.model_dump(mode="json")
    ).raise_for_status()
    load_time = wait_for_pipeline()
    print(f"Pipeline loaded in {load_time:.1f}s")

    # Load input video if specified
    input_video = None
    if "input_video" in cfg:
        input_video = load_video_for_v2v(cfg["input_video"], height, width)
        print(f"Input video: {input_video['shape']}")

    # Load VACE frames if specified
    vace_frames = None
    if "vace_frames" in cfg:
        vace_frames = load_video_for_vace(cfg["vace_frames"], height, width)
        print(f"VACE frames: {vace_frames['shape']}")

    # Load VACE masks if specified
    vace_masks = None
    if "vace_masks" in cfg:
        vace_masks = load_mask_for_vace(cfg["vace_masks"], height, width)
        print(f"VACE masks: {vace_masks['shape']}")

    # Build and send request
    gen_request = GenerateRequest(
        pipeline_id=pipeline_id,
        prompt=cfg["prompt"],
        num_frames=cfg["num_frames"],
        input_video=input_video,
        noise_scale=cfg.get("noise_scale", 0.7),
        vace_frames=vace_frames,
        vace_masks=vace_masks,
        vace_context_scale=cfg.get("vace_context_scale", 1.0),
        lora_scales=lora_scales,
        manage_cache=cfg.get("manage_cache", True),
    )

    print(f"Generating {cfg['num_frames']} frames...")
    start = time.time()

    with requests.post(
        f"{SERVER_URL}/api/v1/generate",
        json=gen_request.model_dump(exclude_none=True),
        stream=True,
        headers={"Accept": "text/event-stream"},
    ) as resp:
        resp.raise_for_status()
        result = None
        for event_type, data in parse_sse_events(resp):
            if event_type == "progress":
                print(
                    f"  Chunk {data['chunk']}/{data['total_chunks']}: {data['fps']:.1f} fps"
                )
            elif event_type == "complete":
                result = data
                break
            elif event_type == "error":
                raise RuntimeError(f"Generation failed: {data['error']}")

    if result is None:
        raise RuntimeError("No complete event received")

    # Decode and save
    video = np.frombuffer(
        base64.b64decode(result["video_base64"]), dtype=np.float32
    ).reshape(result["video_shape"])

    output_path = f"test_{name}.mp4"
    export_to_video(video, output_path, fps=16)

    print(f"\nComplete in {time.time() - start:.1f}s")
    print(f"Output: {output_path} ({result['video_shape']})")


def main():
    if len(sys.argv) < 2 or sys.argv[1] == "--list":
        print("Available tests:")
        for name, cfg in TESTS.items():
            print(f"  {name:20} - {cfg['description']}")
        print("\nUsage: python test_generate_endpoint.py <test_name>")
        print("       python test_generate_endpoint.py all")
        return

    if sys.argv[1] == "all":
        for name in TESTS:
            run_test(name)
    else:
        run_test(sys.argv[1])


if __name__ == "__main__":
    main()
