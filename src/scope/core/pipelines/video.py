import torch
from einops import rearrange
from torchcodec.decoders import VideoDecoder
from torchvision.transforms import v2


def load_video(
    path: str,
    num_frames: int = None,
    resize_hw: tuple[int, int] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Loads a video as a CTHW tensor
    """
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
