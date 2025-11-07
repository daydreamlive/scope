#!/usr/bin/env python3
"""
Video concatenation script for comparison purposes.
Concatenates an anchor video with each video in the same folder.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def get_video_files(directory):
    """Get all video files from a directory."""
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}
    video_files = []
    for file in Path(directory).iterdir():
        if file.is_file() and file.suffix.lower() in video_extensions:
            video_files.append(file)
    return sorted(video_files)


def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Error getting duration for {video_path}: {e}")
        return None


def get_video_resolution(video_path):
    """Get video resolution (width, height) using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        width, height = map(int, result.stdout.strip().split("x"))
        return width, height
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Error getting resolution for {video_path}: {e}")
        return None, None


def escape_text(text):
    """Escape special characters for ffmpeg drawtext filter."""
    # Escape backslashes, single quotes, and colons
    return text.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")


def find_windows_font():
    """Find a common Windows font file to use with drawtext."""
    import platform

    if platform.system() != "Windows":
        return None

    # Common Windows font paths
    font_paths = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\cour.ttf",
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            return font_path

    return None


def concatenate_videos(anchor_path, other_path, output_path, orientation="vertical"):
    """
    Concatenate two videos using ffmpeg.

    Args:
        anchor_path: Path to anchor video
        other_path: Path to other video
        output_path: Path for output video
        orientation: 'vertical' or 'horizontal'
    """
    anchor_path = Path(anchor_path)
    other_path = Path(other_path)
    output_path = Path(output_path)

    # Get video properties
    anchor_duration = get_video_duration(anchor_path)
    other_duration = get_video_duration(other_path)

    if anchor_duration is None or other_duration is None:
        print(f"Skipping {other_path.name} - could not get video properties")
        return False

    anchor_res = get_video_resolution(anchor_path)
    other_res = get_video_resolution(other_path)

    if anchor_res[0] is None or other_res[0] is None:
        print(f"Skipping {other_path.name} - could not get video resolution")
        return False

    anchor_w, anchor_h = anchor_res
    other_w, other_h = other_res

    # Determine output dimensions
    if orientation == "vertical":
        output_w = max(anchor_w, other_w)
        output_h = anchor_h + other_h
    else:  # horizontal
        output_w = anchor_w + other_w
        output_h = max(anchor_h, other_h)

    # Determine max duration for padding
    max_duration = max(anchor_duration, other_duration)
    anchor_pad_duration = max(0, max_duration - anchor_duration)
    other_pad_duration = max(0, max_duration - other_duration)

    # Escape filenames for text overlay
    anchor_name_escaped = escape_text(anchor_path.name)
    other_name_escaped = escape_text(other_path.name)

    # Find a Windows font if available
    font_file = find_windows_font()
    if font_file:
        # Escape the font file path for ffmpeg (convert backslashes to forward slashes)
        font_file_escaped = str(font_file).replace("\\", "/")
        font_param = f"fontfile='{font_file_escaped}':"
    else:
        font_param = ""

    # Build ffmpeg filter complex
    # Scale both videos to same width (for vertical) or height (for horizontal)
    if orientation == "vertical":
        # Scale both to same width, maintaining aspect ratio
        scale_w = max(anchor_w, other_w)
        anchor_scale = f"scale={scale_w}:-1"
        other_scale = f"scale={scale_w}:-1"

        # After scaling, get actual heights
        anchor_scale_h = int(anchor_h * scale_w / anchor_w)
        other_scale_h = int(other_h * scale_w / other_w)

        # Pad to ensure exact heights
        anchor_pad = f"pad={scale_w}:{anchor_scale_h}:0:0:black"
        other_pad = f"pad={scale_w}:{other_scale_h}:0:0:black"

        # Add text overlays with font specification
        anchor_text = f"drawtext={font_param}text='{anchor_name_escaped}':fontcolor=white:fontsize=24:x=w-tw-10:y=h-th-10:box=1:boxcolor=black@0.5"
        other_text = f"drawtext={font_param}text='{other_name_escaped}':fontcolor=white:fontsize=24:x=w-tw-10:y=h-th-10:box=1:boxcolor=black@0.5"

        # Build filter chains
        anchor_filters = [anchor_scale, anchor_pad, anchor_text]
        if anchor_pad_duration > 0:
            anchor_filters.append(
                f"tpad=stop_mode=add:stop_duration={anchor_pad_duration}"
            )

        other_filters = [other_scale, other_pad, other_text]
        if other_pad_duration > 0:
            other_filters.append(
                f"tpad=stop_mode=add:stop_duration={other_pad_duration}"
            )

        # Concatenate vertically
        filter_complex = (
            f"[0:v]{','.join(anchor_filters)}[v0];"
            f"[1:v]{','.join(other_filters)}[v1];"
            f"[v0][v1]vstack=inputs=2[v]"
        )
    else:  # horizontal
        # Scale both to same height, maintaining aspect ratio
        scale_h = max(anchor_h, other_h)
        anchor_scale = f"scale=-1:{scale_h}"
        other_scale = f"scale=-1:{scale_h}"

        # After scaling, get actual widths
        anchor_scale_w = int(anchor_w * scale_h / anchor_h)
        other_scale_w = int(other_w * scale_h / other_h)

        # Pad to ensure exact widths
        anchor_pad = f"pad={anchor_scale_w}:{scale_h}:0:0:black"
        other_pad = f"pad={other_scale_w}:{scale_h}:0:0:black"

        # Add text overlays with font specification
        anchor_text = f"drawtext={font_param}text='{anchor_name_escaped}':fontcolor=white:fontsize=24:x=w-tw-10:y=h-th-10:box=1:boxcolor=black@0.5"
        other_text = f"drawtext={font_param}text='{other_name_escaped}':fontcolor=white:fontsize=24:x=w-tw-10:y=h-th-10:box=1:boxcolor=black@0.5"

        # Build filter chains
        anchor_filters = [anchor_scale, anchor_pad, anchor_text]
        if anchor_pad_duration > 0:
            anchor_filters.append(
                f"tpad=stop_mode=add:stop_duration={anchor_pad_duration}"
            )

        other_filters = [other_scale, other_pad, other_text]
        if other_pad_duration > 0:
            other_filters.append(
                f"tpad=stop_mode=add:stop_duration={other_pad_duration}"
            )

        # Concatenate horizontally
        filter_complex = (
            f"[0:v]{','.join(anchor_filters)}[v0];"
            f"[1:v]{','.join(other_filters)}[v1];"
            f"[v0][v1]hstack=inputs=2[v]"
        )

    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-i",
        str(anchor_path),
        "-i",
        str(other_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-y",  # Overwrite output file
        str(output_path),
    ]

    try:
        print(
            f"Processing: {anchor_path.name} + {other_path.name} -> {output_path.name}"
        )
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Created: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing {other_path.name}: {e}")
        if e.stderr:
            print(f"FFmpeg error: {e.stderr.decode()}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate videos for comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python concatenate_videos.py video.mp4
  python concatenate_videos.py video.mp4 --horizontal
  python concatenate_videos.py /path/to/video.mp4 --output-dir ./output
        """,
    )
    parser.add_argument("anchor_video", help="Path to anchor video file")
    parser.add_argument(
        "--orientation",
        "-o",
        choices=["vertical", "horizontal"],
        default="vertical",
        help="Concatenation orientation (default: vertical)",
    )
    parser.add_argument(
        "--output-dir",
        "-d",
        help="Output directory (default: same as anchor video directory)",
    )

    args = parser.parse_args()

    anchor_path = Path(args.anchor_video)

    if not anchor_path.exists():
        print(f"Error: Anchor video not found: {anchor_path}")
        sys.exit(1)

    if not anchor_path.is_file():
        print(f"Error: Anchor path is not a file: {anchor_path}")
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = anchor_path.parent

    # Get all video files in the same directory
    video_dir = anchor_path.parent
    all_videos = get_video_files(video_dir)

    if not all_videos:
        print(f"No video files found in {video_dir}")
        sys.exit(1)

    # Filter out the anchor video itself
    other_videos = [v for v in all_videos if v != anchor_path]

    if not other_videos:
        print(f"No other videos found in {video_dir} to concatenate with anchor")
        sys.exit(1)

    print(
        f"Found {len(other_videos)} video(s) to concatenate with anchor: {anchor_path.name}"
    )
    print(f"Orientation: {args.orientation}")
    print(f"Output directory: {output_dir}\n")

    # Process each video
    success_count = 0
    for other_video in other_videos:
        # Create output filename
        anchor_stem = anchor_path.stem
        other_stem = other_video.stem
        orientation_suffix = "v" if args.orientation == "vertical" else "h"
        output_filename = f"{anchor_stem}_{other_stem}_{orientation_suffix}.mp4"
        output_path = output_dir / output_filename

        if concatenate_videos(anchor_path, other_video, output_path, args.orientation):
            success_count += 1

    print(
        f"\nCompleted: {success_count}/{len(other_videos)} videos processed successfully"
    )


if __name__ == "__main__":
    main()
