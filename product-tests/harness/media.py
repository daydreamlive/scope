"""Media-quality helpers — decode MP4s, reason about frame timing, compare images.

This is the machine half of "tests as well as a human". A human looking at a
recorded MP4 can tell when the framerate is wrong, when frames stutter, when
pixelation appears. This module gives tests the same signals by reading raw
presentation timestamps (``pts``) via ``ffprobe``, sampling frames, and
measuring structural/perceptual similarity between images.

None of this is multimodal or LLM-based. Multimodal lives in
``harness.visual_eval``. This module is pure signal processing — cheap, fast,
deterministic — and catches the bug class where the output is statistically
broken (wrong framerate, synthesized-looking timestamps, black frames, frozen
frames) regardless of whether a human would notice it at a glance.

Dependencies:
- ``ffmpeg`` and ``ffprobe`` must be on PATH. The CI runner installs them via
  ``apt-get install ffmpeg``. Locally on macOS: ``brew install ffmpeg``.
- ``opencv-python-headless`` (already in ``product-tests`` group) supplies the
  primitives for SSIM and frame decoding.
- ``numpy`` is pulled transitively by OpenCV.

Everything here is **sync** — we do not reach out to a network, and we are
happy to block a test for the couple of seconds needed to run ffprobe.
"""

from __future__ import annotations

import dataclasses
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

# -----------------------------------------------------------------------------
# ffprobe / ffmpeg shellouts
# -----------------------------------------------------------------------------


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"{name} not found on PATH. Install ffmpeg "
            "(`brew install ffmpeg` on macOS, `apt-get install ffmpeg` on Linux)."
        )
    return path


def ffprobe_pts(path: Path | str) -> list[float]:
    """Return the list of per-frame PTS values (seconds) for the first video stream.

    Uses ``ffprobe -select_streams v:0 -show_frames -print_format json``. The
    ``pts_time`` field is preferred; falls back to ``best_effort_timestamp_time``
    when ``pts_time`` is missing (common when the container doesn't carry real
    timestamps — which is exactly the pattern we want to catch).
    """
    ffprobe = _require_binary("ffprobe")
    cmd = [
        ffprobe,
        "-loglevel",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-print_format",
        "json",
        str(path),
    ]
    raw = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    data = json.loads(raw)
    pts: list[float] = []
    for f in data.get("frames", []):
        t = f.get("pts_time") or f.get("best_effort_timestamp_time")
        if t is None:
            continue
        try:
            pts.append(float(t))
        except (TypeError, ValueError):
            continue
    return pts


@dataclasses.dataclass(frozen=True)
class TimingReport:
    """Timing analysis for a recorded video's PTS sequence.

    - ``frame_count``: number of frames we got timestamps for.
    - ``duration_sec``: last PTS minus first PTS.
    - ``mean_fps``: ``(frame_count - 1) / duration_sec``.
    - ``mean_frame_duration_sec``: 1 / mean_fps.
    - ``jitter_stddev_sec``: stddev of inter-frame deltas. Synthesized timestamps
      typically produce a pathologically low stddev (near 0) across arbitrary
      content; real pipelines hover around the nominal frame duration with
      nonzero but small noise.
    - ``jitter_p95_sec``: 95th-percentile absolute deviation from the mean delta.
    - ``looks_synthesized``: heuristic — True when stddev/mean_delta < 0.01 for
      at least 30 frames. Synthesized PTS are suspiciously regular; real runner
      PTS from WebRTC / pipeline runners have measurable jitter even on a stable
      frame loop because of thread scheduling and encoder variance.
    """

    frame_count: int
    duration_sec: float
    mean_fps: float
    mean_frame_duration_sec: float
    jitter_stddev_sec: float
    jitter_p95_sec: float
    looks_synthesized: bool


def analyze_timing(pts: list[float]) -> TimingReport:
    n = len(pts)
    if n < 2:
        return TimingReport(
            frame_count=n,
            duration_sec=0.0,
            mean_fps=0.0,
            mean_frame_duration_sec=0.0,
            jitter_stddev_sec=0.0,
            jitter_p95_sec=0.0,
            looks_synthesized=False,
        )
    arr = np.asarray(pts, dtype=np.float64)
    deltas = np.diff(arr)
    duration = float(arr[-1] - arr[0])
    mean_delta = float(deltas.mean()) if deltas.size else 0.0
    mean_fps = (n - 1) / duration if duration > 0 else 0.0
    stddev = float(deltas.std(ddof=0)) if deltas.size else 0.0
    p95 = float(np.percentile(np.abs(deltas - mean_delta), 95)) if deltas.size else 0.0

    # Synthesized-timestamp heuristic: extremely regular deltas, across a
    # meaningful number of frames (<30 frames is too short to trust).
    looks_synth = bool(
        n >= 30 and mean_delta > 0 and (stddev / max(mean_delta, 1e-9)) < 0.01
    )

    return TimingReport(
        frame_count=n,
        duration_sec=duration,
        mean_fps=mean_fps,
        mean_frame_duration_sec=mean_delta,
        jitter_stddev_sec=stddev,
        jitter_p95_sec=p95,
        looks_synthesized=looks_synth,
    )


def sample_frames(
    path: Path | str,
    n: int,
    out_dir: Path | str | None = None,
    prefix: str = "frame",
) -> list[Path]:
    """Extract ``n`` evenly-spaced JPEGs from the given video.

    Returns the list of JPEG paths. ``out_dir`` defaults to a sibling ``_frames``
    directory next to the video. Uses ffmpeg's ``-vf select`` with
    ``fps=n/duration`` plus ``-frames:v n`` for accuracy.
    """
    if n <= 0:
        raise ValueError("n must be >= 1")
    ffmpeg = _require_binary("ffmpeg")
    ffprobe = _require_binary("ffprobe")
    path = Path(path)
    if out_dir is None:
        out_dir = path.parent / f"{path.stem}_frames"
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Duration — needed to compute fps.
    raw = subprocess.check_output(
        [
            ffprobe,
            "-loglevel",
            "error",
            "-show_entries",
            "format=duration",
            "-print_format",
            "json",
            str(path),
        ],
    )
    duration = float(json.loads(raw)["format"]["duration"])
    if duration <= 0:
        # Fall back to decoding the first n frames.
        fps_filter = "1"
    else:
        # Sample rate that yields approximately n frames across the duration.
        fps_filter = f"{max(n / duration, 1e-3):.6f}"

    pattern = str(out_dir_path / f"{prefix}_%03d.jpg")
    subprocess.check_call(
        [
            ffmpeg,
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-vf",
            f"fps={fps_filter}",
            "-frames:v",
            str(n),
            "-q:v",
            "3",
            pattern,
        ]
    )
    return sorted(out_dir_path.glob(f"{prefix}_*.jpg"))


# -----------------------------------------------------------------------------
# Similarity: SSIM + perceptual hash
# -----------------------------------------------------------------------------


def _read_gray(path: Path | str) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def ssim(a: Path | str, b: Path | str) -> float:
    """Structural Similarity Index between two images in [0, 1].

    1.0 = identical, 0.0 = completely different. Uses the standard SSIM with
    the default Gaussian window and C1/C2 constants from the original paper,
    implemented in pure OpenCV/numpy so we don't add a ``scikit-image``
    dependency for one function.

    Both inputs are loaded as grayscale. Images of different sizes are resized
    to the smaller one's dimensions before comparison.
    """
    import cv2

    g1 = _read_gray(a).astype(np.float64)
    g2 = _read_gray(b).astype(np.float64)
    if g1.shape != g2.shape:
        h = min(g1.shape[0], g2.shape[0])
        w = min(g1.shape[1], g2.shape[1])
        g1 = cv2.resize(g1, (w, h))
        g2 = cv2.resize(g2, (w, h))

    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    # Gaussian-blur each image and each squared.
    mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(g1 * g1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2 * g2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2

    num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = num / (den + 1e-12)
    return float(np.clip(ssim_map.mean(), 0.0, 1.0))


def perceptual_hash(path: Path | str, hash_size: int = 8) -> str:
    """Compute a dHash (difference-hash) perceptual hash for the image.

    dHash is cheap, robust to scale/compression/minor color shifts, and
    reduces an image to ``hash_size * hash_size`` bits. Hamming distance
    between two dhashes is a proxy for perceptual similarity.

    Returns the hex-encoded bitstring (16 chars for the default 8×8 = 64 bits).
    """
    import cv2

    g = _read_gray(path)
    # Resize to (hash_size + 1) x hash_size so we can take horizontal diffs.
    small = cv2.resize(g, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = diff.flatten()
    # Pack into an integer, then hex.
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return f"{value:0{math.ceil(len(bits) / 4)}x}"


def hamming_distance(h1: str, h2: str) -> int:
    """Bit distance between two hex-encoded perceptual hashes.

    ``phash`` values with distance <= 5 (for the default 64-bit hash) are
    near-identical; distance > 20 is almost certainly unrelated content.
    """
    i1 = int(h1, 16) if h1 else 0
    i2 = int(h2, 16) if h2 else 0
    return (
        (i1 ^ i2).bit_count() if hasattr(int, "bit_count") else bin(i1 ^ i2).count("1")
    )  # noqa: E501


# -----------------------------------------------------------------------------
# Quick-pathology checks — answer "is this frame black / frozen / all-one-color"
# -----------------------------------------------------------------------------


def mean_brightness(path: Path | str) -> float:
    """0..255 grayscale mean — useful for detecting all-black / all-white frames."""
    return float(_read_gray(path).mean())


def color_variance(path: Path | str) -> float:
    """Variance of grayscale intensity — near 0 when a frame is single-color."""
    return float(_read_gray(path).var())


def looks_black(path: Path | str, mean_threshold: float = 5.0) -> bool:
    return mean_brightness(path) < mean_threshold


def looks_monochrome(path: Path | str, variance_threshold: float = 4.0) -> bool:
    """All-one-color / near-flat frame (e.g. sink rendered solid gray)."""
    return color_variance(path) < variance_threshold
