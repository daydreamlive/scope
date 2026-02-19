"""Test script for the /api/v1/generate endpoint.

Usage:
    python test_generate_endpoint.py <test_name>
    python test_generate_endpoint.py --list
"""

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
LORA = r"C:\Users\ryanf\.daydream-scope\models\lora\lora\output\model_245889_dissolve_imgvid\dissolve-000064.safetensors"
TEST_VIDEO = r"frontend\public\assets\test.mp4"
VACE_CONDITIONING_VIDEO = r"controlnet_test\control_frames_depth.mp4"
MASK_VIDEO = r"src\scope\core\pipelines\longlive\vace_tests\static_mask_half_white_half_black.mp4"

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


def upload_video_for_v2v(path: str, height: int, width: int) -> str:
    """Load and upload video for video-to-video mode. Returns input_path."""
    tensor = load_video(path, resize_hw=(height, width), normalize=False)
    arr = tensor.permute(1, 2, 3, 0).numpy().astype(np.uint8)
    num_frames, h, w, c = arr.shape

    response = requests.post(
        f"{SERVER_URL}/api/v1/generate/upload",
        data=arr.tobytes(),
        headers={
            "Content-Type": "application/octet-stream",
            "X-Video-Frames": str(num_frames),
            "X-Video-Height": str(h),
            "X-Video-Width": str(w),
            "X-Video-Channels": str(c),
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["input_path"]


def upload_vace_data(
    vace_frames_path: str | None,
    vace_masks_path: str | None,
    height: int,
    width: int,
    num_frames: int,
    chunk_size: int,
    vace_context_scale: float = 1.0,
) -> tuple[str, list[dict]]:
    """Load VACE frames/masks, pack into blob, upload, return (data_blob_path, chunk_specs)."""
    blob = bytearray()
    num_chunks = (num_frames + chunk_size - 1) // chunk_size
    chunk_specs = []

    # Load tensors
    vace_frames_tensor = None
    vace_masks_tensor = None
    if vace_frames_path:
        vace_frames_tensor = load_video(vace_frames_path, resize_hw=(height, width))
        vace_frames_tensor = vace_frames_tensor.unsqueeze(0).numpy().astype(np.float32)
    if vace_masks_path:
        masks_tensor = load_video(vace_masks_path, resize_hw=(height, width))
        vace_masks_tensor = (masks_tensor[0:1].unsqueeze(0).numpy() > 0.0).astype(
            np.float32
        )

    for chunk_idx in range(num_chunks):
        spec = {"chunk": chunk_idx, "vace_temporally_locked": True}
        start = chunk_idx * chunk_size
        end = start + chunk_size

        if vace_frames_tensor is not None:
            sliced = vace_frames_tensor[:, :, start:end, :, :]
            spec["vace_frames_offset"] = len(blob)
            spec["vace_frames_shape"] = list(sliced.shape)
            blob.extend(sliced.tobytes())

        if vace_masks_tensor is not None:
            sliced_masks = vace_masks_tensor[:, :, start:end, :, :]
            spec["vace_masks_offset"] = len(blob)
            spec["vace_masks_shape"] = list(sliced_masks.shape)
            blob.extend(sliced_masks.tobytes())

        if vace_context_scale != 1.0:
            spec["vace_context_scale"] = vace_context_scale

        chunk_specs.append(spec)

    # Upload blob
    response = requests.post(
        f"{SERVER_URL}/api/v1/generate/upload-data",
        data=bytes(blob),
        headers={"Content-Type": "application/octet-stream"},
        timeout=300,
    )
    response.raise_for_status()
    data_blob_path = response.json()["data_blob_path"]

    return data_blob_path, chunk_specs


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


def download_video(output_path: str) -> np.ndarray:
    """Download generated video from server."""
    response = requests.get(
        f"{SERVER_URL}/api/v1/generate/download",
        params={"path": output_path},
        timeout=300,
    )
    response.raise_for_status()

    num_frames = int(response.headers.get("X-Video-Frames", 0))
    height = int(response.headers.get("X-Video-Height", 0))
    width = int(response.headers.get("X-Video-Width", 0))
    channels = int(response.headers.get("X-Video-Channels", 3))

    # Skip header (ndim + shape)
    content = response.content
    header_size = 4 + 4 * 4
    video_bytes = content[header_size:]

    return np.frombuffer(video_bytes, dtype=np.uint8).reshape(
        (num_frames, height, width, channels)
    )


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
            lora_scales = cfg["lora_ramp"]
            print(f"LoRA ramp: {lora_scales}")

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

    # Build request kwargs
    request_kwargs = {
        "pipeline_id": pipeline_id,
        "prompt": cfg["prompt"],
        "num_frames": cfg["num_frames"],
        "noise_scale": cfg.get("noise_scale", 0.7),
        "vace_context_scale": cfg.get("vace_context_scale", 1.0),
        "manage_cache": cfg.get("manage_cache", True),
    }

    # Upload input video if specified
    if "input_video" in cfg:
        input_path = upload_video_for_v2v(cfg["input_video"], height, width)
        request_kwargs["input_path"] = input_path
        print(f"Input video uploaded: {input_path}")

    # Build chunk_specs for LoRA ramp
    chunk_specs = []
    if lora_scales and "lora" in cfg:
        for i, scale in enumerate(lora_scales):
            chunk_specs.append(
                {
                    "chunk": i,
                    "lora_scales": {cfg["lora"]: scale},
                }
            )

    # Handle VACE data
    if "vace_frames" in cfg or "vace_masks" in cfg:
        # Assume chunk_size=12 (default for longlive)
        chunk_size = 12
        data_blob_path, vace_specs = upload_vace_data(
            vace_frames_path=cfg.get("vace_frames"),
            vace_masks_path=cfg.get("vace_masks"),
            height=height,
            width=width,
            num_frames=cfg["num_frames"],
            chunk_size=chunk_size,
            vace_context_scale=cfg.get("vace_context_scale", 1.0),
        )
        request_kwargs["data_blob_path"] = data_blob_path
        # Merge VACE specs into chunk_specs
        existing_chunks = {s["chunk"] for s in chunk_specs}
        for vs in vace_specs:
            if vs["chunk"] in existing_chunks:
                # Merge into existing spec
                for cs in chunk_specs:
                    if cs["chunk"] == vs["chunk"]:
                        cs.update(vs)
                        break
            else:
                chunk_specs.append(vs)
        print(f"VACE data uploaded: {data_blob_path}")

    if chunk_specs:
        chunk_specs.sort(key=lambda s: s["chunk"])
        request_kwargs["chunk_specs"] = chunk_specs

    gen_request = GenerateRequest(**request_kwargs)

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

    # Download and save
    if "output_path" in result:
        video = download_video(result["output_path"])
        video_float = video.astype(np.float32) / 255.0
    else:
        raise RuntimeError("No output_path in result")

    output_path = f"test_{name}.mp4"
    export_to_video(video_float, output_path, fps=16)

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
