from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline


def _save_frames_to_video(frames: np.ndarray, video_path: str, fps: int = 16) -> None:
    """Save RGB frames [T, H, W, 3] to an MP4 video."""
    video_path = str(video_path)
    Path(video_path).parent.mkdir(parents=True, exist_ok=True)

    if frames.size == 0:
        return

    height, width = frames.shape[1], frames.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height))

    try:
        for frame in frames:
            frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()


def canny_preprocessor(
    video_path: str,
    num_frames: int,
    target_height: int,
    target_width: int,
    output_video_path: str,
) -> np.ndarray | None:
    """Extract canny edges and save them as a reference video."""
    from .extract_canny_edges import extract_canny_edges

    print(
        "canny_preprocessor: extracting canny edges from "
        f"{video_path} into {output_video_path}"
    )
    edges = extract_canny_edges(
        video_path,
        num_frames,
        target_height,
        target_width,
    )
    if edges is None:
        return None

    _save_frames_to_video(edges, output_video_path)
    return edges


def depth_preprocessor(
    video_path: str,
    num_frames: int,
    target_height: int,
    target_width: int,
    output_video_path: str,
    model_name: str = "Intel/dpt-large",
    detect_resolution: int = 512,
) -> np.ndarray | None:
    """Extract real depth maps using MiDaS depth estimation model."""
    print(
        "depth_preprocessor: extracting depth maps from "
        f"{video_path} into {output_video_path}"
    )

    print(f"depth_preprocessor: loading depth estimation model ({model_name})...")
    depth_estimator = pipeline(
        "depth-estimation",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("depth_preprocessor: ERROR: cannot open video")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)

    frames: list[np.ndarray] = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            print(f"depth_preprocessor: WARNING: failed to read frame {idx}")
            continue

        frame = cv2.resize(frame, (target_width, target_height))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        image_resized = pil_image.resize(
            (detect_resolution, detect_resolution), Image.LANCZOS
        )

        depth_result = depth_estimator(image_resized)
        depth_map = depth_result["depth"]

        if hasattr(depth_map, "cpu"):
            depth_np = depth_map.cpu().numpy()
        else:
            depth_np = np.array(depth_map)

        depth_min = depth_np.min()
        depth_max = depth_np.max()
        if depth_max > depth_min:
            depth_normalized = (
                ((depth_np - depth_min) / (depth_max - depth_min)) * 255
            ).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(depth_np, dtype=np.uint8)

        depth_resized = cv2.resize(
            depth_normalized,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )
        depth_rgb = np.stack([depth_resized] * 3, axis=-1)

        frames.append(depth_rgb)

    cap.release()

    if not frames:
        print("depth_preprocessor: ERROR: no frames extracted")
        return None

    depth_array = np.stack(frames, axis=0)
    _save_frames_to_video(depth_array, output_video_path)
    print(f"depth_preprocessor: saved {len(frames)} depth maps")
    return depth_array
