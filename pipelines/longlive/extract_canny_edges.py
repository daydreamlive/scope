"""
Extract canny edges from a video file.
"""

import cv2
import numpy as np


def extract_canny_edges(
    video_path,
    num_frames,
    target_height,
    target_width,
    low_threshold=30,
    high_threshold=100,
):
    """Extract canny edges from video."""
    print(f"extract_canny_edges: Loading {video_path}...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("extract_canny_edges: ERROR: Cannot open video")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"extract_canny_edges: Video has {total_frames} frames")
    print(f"extract_canny_edges: Extracting {num_frames} frames")
    print(f"extract_canny_edges: Target resolution: {target_height}x{target_width}")
    print(f"extract_canny_edges: Canny thresholds: {low_threshold}, {high_threshold}")

    edges_list = []
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for i, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            print(f"extract_canny_edges: ERROR: Could not read frame {idx}")
            continue

        # Resize
        frame = cv2.resize(frame, (target_width, target_height))

        # Convert BGR to RGB first
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to grayscale
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        # Canny edges
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        # Debug first frame
        if i == 0:
            print(
                f"extract_canny_edges: Frame 0 - gray min/max: {gray.min()}/{gray.max()}"
            )
            print(
                f"extract_canny_edges: Frame 0 - edges min/max: {edges.min()}/{edges.max()}"
            )
            print(
                f"extract_canny_edges: Frame 0 - edge pixels: {(edges > 0).sum()} / {edges.size}"
            )

        # Convert edges to RGB
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        edges_list.append(edges_rgb)

    cap.release()

    if not edges_list:
        print("extract_canny_edges: ERROR: No frames extracted")
        return None

    edges_array = np.stack(edges_list, axis=0)
    print(f"extract_canny_edges: Extracted {len(edges_list)} frames")
    print(f"extract_canny_edges: Shape: {edges_array.shape}")
    print(f"extract_canny_edges: Min: {edges_array.min()}, Max: {edges_array.max()}")
    print(
        f"extract_canny_edges: Total edge pixels across all frames: {(edges_array > 0).sum()}"
    )

    return edges_array


def main():
    video_path = "pipelines/streamdiffusionv2/assets/original.mp4"
    num_frames = 36
    height = 480
    width = 832

    edges = extract_canny_edges(video_path, num_frames, height, width)

    if edges is not None:
        # Save as video using opencv directly
        output_path = "pipelines/longlive/control_frames_canny.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, 16.0, (width, height))

        for frame in edges:
            # Convert RGB to BGR for opencv
            frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()

        print(f"\nextract_canny_edges: Saved video to {output_path}")
        print(f"extract_canny_edges: {len(edges)} frames at 16 fps")
    else:
        print("\nextract_canny_edges: FAILED")


if __name__ == "__main__":
    main()
