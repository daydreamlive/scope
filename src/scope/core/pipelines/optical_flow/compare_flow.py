"""Compare optical flow outputs: PyTorch vs torch.compile backends.

Loads a video, runs OpticalFlowPipeline with both backends, and saves
side-by-side comparison outputs so quality differences can be visually inspected.

Usage:
    uv run python -m scope.core.pipelines.optical_flow.compare_flow
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch._inductor.config as inductor_config

# Disable CPU codegen to avoid needing MSVC on Windows
inductor_config.disable_cpp_codegen = True

OUTPUT_DIR = Path("optical_flow_comparison_output")
INPUT_VIDEO = Path("frontend/public/assets/test.mp4")
HEIGHT = 480
WIDTH = 832


def load_video(path: Path, target_height: int, target_width: int) -> np.ndarray:
    """Load video and resize to target dimensions. Returns (T, H, W, 3) uint8 RGB."""
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


def save_video(frames: np.ndarray, path: Path, fps: int = 8):
    """Save (T, H, W, 3) uint8 RGB array as mp4."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1], frames.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  Saved: {path}")


def save_frames(frames: np.ndarray, output_dir: Path, prefix: str):
    """Save individual frames as PNGs for detailed comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        path = output_dir / f"{prefix}_frame_{i:04d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# OpticalFlowPipeline runner
# ---------------------------------------------------------------------------
def run_pipeline_flow(
    video: np.ndarray,
    device: torch.device,
    model_size: str = "large",
) -> np.ndarray:
    """Run OpticalFlowPipeline."""
    from .pipeline import OpticalFlowPipeline
    from .schema import OpticalFlowConfig

    print(f"\n--- OpticalFlowPipeline (model_size={model_size}) ---")

    config = OpticalFlowConfig(model_size=model_size)
    pipeline = OpticalFlowPipeline(config, device=device)

    # The plugin expects a list of tensors in THWC format
    # Each element: (1, H, W, C) or (H, W, C) — the plugin handles both
    video_tensors = [torch.from_numpy(frame).unsqueeze(0) for frame in video]

    result = pipeline(video=video_tensors)
    # result["video"]: (T, H, W, 3) float [0, 1]
    flow_tensor = result["video"]
    flow_np = (flow_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    print(f"  Done. Generated {flow_np.shape[0]} flow frames.")
    return flow_np


def run_compiled_flow(
    video: np.ndarray,
    device: torch.device,
    model_size: str = "large",
) -> np.ndarray:
    """Run optical flow with torch.compile backend.

    Matches OpticalFlowPipeline preprocessing exactly:
    - Resize to 512x512 for flow computation
    - Scale to [0, 255] range for RAFT
    - Pad to multiple of 8
    """
    import time

    import torch.nn.functional as F
    from torchvision.utils import flow_to_image

    from .engine import DEFAULT_HEIGHT, DEFAULT_WIDTH, load_raft_model

    print(f"\n--- torch.compile Backend (model_size={model_size}) ---")

    use_large = model_size == "large"
    model, _ = load_raft_model(use_large, device=str(device))

    # Compile the model
    print("  Compiling model...")
    start = time.time()
    compiled_model = torch.compile(model, backend="inductor", fullgraph=False)
    print(f"  Compile setup time: {time.time() - start:.2f}s")

    # Convert video to tensor (T, H, W, C) -> (T, C, H, W)
    video_tensor = torch.from_numpy(video).to(device=device, dtype=torch.float32)
    video_tensor = video_tensor.permute(0, 3, 1, 2)

    num_frames = video_tensor.shape[0]
    h, w = video_tensor.shape[2], video_tensor.shape[3]

    # Normalize to [0, 1] then resize to flow computation size (matching pipeline)
    if video_tensor.max() > 1.0:
        video_tensor = video_tensor / 255.0

    # Resize to flow computation size (512x512 by default)
    flow_h, flow_w = DEFAULT_HEIGHT, DEFAULT_WIDTH
    if h != flow_h or w != flow_w:
        frames_for_flow = F.interpolate(
            video_tensor, size=(flow_h, flow_w), mode="bilinear", align_corners=False
        )
    else:
        frames_for_flow = video_tensor

    # Scale to [0, 255] for RAFT (matching pipeline._preprocess_frame_batch)
    frames_scaled = frames_for_flow * 255.0

    # Pad to multiple of 8 if needed
    pad_h = (8 - flow_h % 8) % 8
    pad_w = (8 - flow_w % 8) % 8
    if pad_h > 0 or pad_w > 0:
        frames_scaled = F.pad(frames_scaled, [0, pad_w, 0, pad_h])

    # Preallocate output
    flow_rgb_list = []

    # Warmup with first pair
    print("  Warming up compiled model...")
    with torch.no_grad():
        _ = compiled_model(frames_scaled[0:1], frames_scaled[1:2])
    torch.cuda.synchronize()

    print("  Computing flow...")
    start = time.time()
    with torch.no_grad():
        for i in range(1, num_frames):
            frame1 = frames_scaled[i - 1 : i]
            frame2 = frames_scaled[i : i + 1]

            flow = compiled_model(frame1, frame2)[-1]

            # Remove padding from flow
            flow = flow[:, :, :flow_h, :flow_w]

            # Convert to RGB visualization
            flow_rgb = flow_to_image(flow[0]).float() / 255.0

            # Resize back to original size if needed
            if flow_rgb.shape[-2] != h or flow_rgb.shape[-1] != w:
                flow_rgb = F.interpolate(
                    flow_rgb.unsqueeze(0),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            flow_rgb_list.append(flow_rgb)

    torch.cuda.synchronize()
    elapsed = time.time() - start
    fps = (num_frames - 1) / elapsed
    print(f"  Inference time: {elapsed:.2f}s ({fps:.1f} FPS)")

    # Stack and duplicate first frame (matching pipeline behavior)
    flow_rgb_tensor = torch.stack(flow_rgb_list, dim=0)
    first_flow = flow_rgb_tensor[0:1]
    flow_rgb_tensor = torch.cat([first_flow, flow_rgb_tensor], dim=0)

    # Convert to numpy (T, C, H, W) -> (T, H, W, C)
    flow_rgb_tensor = flow_rgb_tensor.permute(0, 2, 3, 1)
    flow_np = (flow_rgb_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    print(f"  Done. Generated {flow_np.shape[0]} flow frames.")
    return flow_np


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------
def compute_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute absolute difference between two frame arrays, amplified for visibility."""
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    # Amplify by 3x for visibility
    diff = np.clip(diff * 3, 0, 255).astype(np.uint8)
    return diff


def create_side_by_side(
    baseline: np.ndarray,
    compare: np.ndarray,
    diff: np.ndarray,
    baseline_label: str = "PyTorch",
    compare_label: str = "Compiled",
) -> np.ndarray:
    """Create side-by-side comparison: [baseline | compare | diff]."""
    # Ensure same number of frames
    n = min(len(baseline), len(compare), len(diff))
    baseline = baseline[:n]
    compare = compare[:n]
    diff = diff[:n]

    # Add labels
    labeled_frames = []
    for i in range(n):
        baseline_frame = baseline[i].copy()
        compare_frame = compare[i].copy()
        diff_frame = diff[i].copy()

        cv2.putText(
            baseline_frame,
            baseline_label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            compare_frame,
            compare_label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            diff_frame,
            "Diff (3x)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        combined = np.concatenate([baseline_frame, compare_frame, diff_frame], axis=1)
        labeled_frames.append(combined)

    return np.array(labeled_frames)


def print_stats(
    baseline: np.ndarray,
    compare: np.ndarray,
    baseline_name: str = "PyTorch",
    compare_name: str = "Compiled",
):
    """Print numerical comparison statistics."""
    n = min(len(baseline), len(compare))
    baseline_arr = baseline[:n].astype(np.float32)
    compare_arr = compare[:n].astype(np.float32)
    diff = np.abs(baseline_arr - compare_arr)

    print(f"\n=== Comparison Statistics ({baseline_name} vs {compare_name}) ===")
    print(f"  Frames compared:    {n}")
    print(f"  Mean abs diff:      {diff.mean():.2f} / 255")
    print(f"  Max abs diff:       {diff.max():.0f} / 255")
    print(f"  Median abs diff:    {np.median(diff):.2f} / 255")
    print(f"  Std of diff:        {diff.std():.2f}")
    print(f"  % pixels diff > 5: {(diff > 5).mean() * 100:.1f}%")
    print(f"  % pixels diff > 10: {(diff > 10).mean() * 100:.1f}%")
    print(f"  % pixels diff > 25: {(diff > 25).mean() * 100:.1f}%")

    # Per-frame mean diff
    per_frame = diff.mean(axis=(1, 2, 3))
    print("\n  Per-frame mean diff (first 10):")
    for i, d in enumerate(per_frame[:10]):
        print(f"    Frame {i}: {d:.2f}")


def main():
    print("=" * 60)
    print("  Optical Flow Comparison: PyTorch vs torch.compile")
    print("=" * 60)

    if not INPUT_VIDEO.exists():
        print(f"ERROR: Input video not found: {INPUT_VIDEO}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type != "cuda":
        print("WARNING: CUDA not available.")
        return

    # Load video
    print(f"\nLoading video: {INPUT_VIDEO}")
    video = load_video(INPUT_VIDEO, HEIGHT, WIDTH)
    print(f"  Shape: {video.shape} (T, H, W, C)")

    # Run pipeline and standalone compiled model for comparison
    pipeline_flow = run_pipeline_flow(video, device, model_size="large")
    compiled_flow = run_compiled_flow(video, device, model_size="large")

    # Compute diff and stats
    diff_flow = compute_diff(pipeline_flow, compiled_flow)
    print_stats(pipeline_flow, compiled_flow, "Pipeline", "Standalone")

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving outputs to: {OUTPUT_DIR}/")

    save_video(pipeline_flow, OUTPUT_DIR / "pipeline_flow.mp4")
    save_video(compiled_flow, OUTPUT_DIR / "compiled_flow.mp4")
    save_video(diff_flow, OUTPUT_DIR / "diff_flow.mp4")

    # Side-by-side
    side_by_side = create_side_by_side(
        pipeline_flow, compiled_flow, diff_flow, "Pipeline", "Standalone"
    )
    save_video(side_by_side, OUTPUT_DIR / "comparison_side_by_side.mp4")

    # Save a few individual frames for close inspection
    save_frames(pipeline_flow[:5], OUTPUT_DIR / "frames", "pipeline")
    save_frames(compiled_flow[:5], OUTPUT_DIR / "frames", "standalone")
    save_frames(diff_flow[:5], OUTPUT_DIR / "frames", "diff")

    print("\n" + "=" * 60)
    print("  Done! Compare outputs in:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
