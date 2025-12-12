import logging

import numpy as np
import torch
from einops import rearrange
from torchvision.transforms import v2

logger = logging.getLogger(__name__)

# Try to import torchcodec, fallback to OpenCV if unavailable
_torchcodec_available = False
_cv2_available = False

try:
    from torchcodec.decoders import VideoDecoder

    _torchcodec_available = True
except (ImportError, RuntimeError) as e:
    logger.warning(
        f"torchcodec not available ({type(e).__name__}), falling back to OpenCV for video loading"
    )

# Try to import OpenCV as fallback
if not _torchcodec_available:
    try:
        import cv2

        _cv2_available = True
    except ImportError:
        _cv2_available = False
        logger.error("Neither torchcodec nor OpenCV available for video loading")


def load_video(
    path: str,
    num_frames: int = None,
    resize_hw: tuple[int, int] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Loads a video as a CTHW tensor.

    Tries torchcodec first, falls back to OpenCV if unavailable (e.g., on Windows).
    """
    if _torchcodec_available:
        return _load_video_torchcodec(path, num_frames, resize_hw, normalize)
    elif _cv2_available:
        return _load_video_cv2(path, num_frames, resize_hw, normalize)
    else:
        raise RuntimeError(
            "No video loading backend available. Install torchcodec or opencv-python."
        )


def _load_video_torchcodec(
    path: str,
    num_frames: int = None,
    resize_hw: tuple[int, int] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Load video using torchcodec (preferred method)."""
    try:
        decoder = VideoDecoder(path)

        total_frames = len(decoder)
        video = decoder.get_frames_in_range(
            0, num_frames if num_frames is not None else total_frames
        ).data

        height, width = video.shape[2:]
        if resize_hw is not None and height != resize_hw[0] or width != resize_hw[1]:
            video = v2.Resize(resize_hw, antialias=True)(video)

        video = video.float()

        if normalize:
            # Normalize to [-1, 1]
            video = video / 127.5 - 1.0

        video = rearrange(video, "T C H W -> C T H W")

        return video
    except (RuntimeError, OSError) as e:
        logger.warning(
            f"torchcodec failed to load video ({type(e).__name__}: {e}), falling back to OpenCV"
        )
        if _cv2_available:
            return _load_video_cv2(path, num_frames, resize_hw, normalize)
        else:
            raise RuntimeError(
                f"torchcodec failed and OpenCV fallback unavailable: {e}"
            ) from e


def _load_video_cv2(
    path: str,
    num_frames: int = None,
    resize_hw: tuple[int, int] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Load video using OpenCV (Windows-compatible fallback)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        frame_count += 1
        if num_frames is not None and frame_count >= num_frames:
            break

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"load_video_cv2: No frames loaded from {path}")

    # Stack frames: list of HWC -> THWC
    video = torch.from_numpy(np.stack(frames))
    # Rearrange to TCHW for resize
    video = rearrange(video, "T H W C -> T C H W")

    height, width = video.shape[2:]
    if resize_hw is not None and (height != resize_hw[0] or width != resize_hw[1]):
        video = v2.Resize(resize_hw, antialias=True)(video)

    video = video.float()

    if normalize:
        # Normalize to [-1, 1]
        video = video / 127.5 - 1.0

    # Rearrange to CTHW
    video = rearrange(video, "T C H W -> C T H W")

    return video
