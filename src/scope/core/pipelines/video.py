import imageio.v3 as iio
import torch
from einops import rearrange
from torchvision.transforms import v2


def load_video(
    path: str,
    num_frames: int = None,
    resize_hw: tuple[int, int] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Loads a video as a CTHW tensor.
    """
    # Read video frames as numpy array (THWC format)
    video = iio.imread(path, plugin="pyav")

    if num_frames is not None:
        video = video[:num_frames]

    # Convert to torch tensor and rearrange from THWC to TCHW
    video = torch.from_numpy(video).permute(0, 3, 1, 2)

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
