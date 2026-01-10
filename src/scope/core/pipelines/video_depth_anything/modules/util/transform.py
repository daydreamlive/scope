# Modified from https://github.com/DepthAnything/Video-Depth-Anything
# The original repo is: https://github.com/DepthAnything/Video-Depth-Anything

import numpy as np
import torch
import torch.nn.functional as F

# Map cv2 interpolation constants to torch modes
INTER_AREA = "area"
INTER_NEAREST = "nearest"
INTER_CUBIC = "bicubic"
INTER_LINEAR = "bilinear"


class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(f"resize_method {self.__resize_method} not implemented")

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, min_val=self.__width)
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, max_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, max_val=self.__width)
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(sample["image"].shape[1], sample["image"].shape[0])

        # Resize using torch's interpolate (works with numpy arrays via conversion)
        # Convert numpy to torch tensor, resize, convert back
        img_tensor = torch.from_numpy(sample["image"]).permute(2, 0, 1).unsqueeze(0).float()  # HWC -> 1CHW
        mode = self._torch_interpolation_mode(self.__image_interpolation_method)
        # align_corners=False matches cv2's default behavior
        img_resized = F.interpolate(
            img_tensor,
            size=(height, width),
            mode=mode,
            align_corners=False,
        )
        sample["image"] = img_resized.squeeze(0).permute(1, 2, 0).numpy().astype(sample["image"].dtype)  # 1CHW -> HWC

        if self.__resize_target:
            if "depth" in sample:
                depth_tensor = torch.from_numpy(sample["depth"]).unsqueeze(0).unsqueeze(0).float()  # HW -> 11HW
                depth_resized = F.interpolate(
                    depth_tensor,
                    size=(height, width),
                    mode="nearest",
                    align_corners=False,
                )
                sample["depth"] = depth_resized.squeeze(0).squeeze(0).numpy()  # 11HW -> HW

            if "mask" in sample:
                mask_tensor = torch.from_numpy(sample["mask"].astype(np.float32)).unsqueeze(0).unsqueeze(0)  # HW -> 11HW
                mask_resized = F.interpolate(
                    mask_tensor,
                    size=(height, width),
                    mode="nearest",
                    align_corners=False,
                )
                sample["mask"] = mask_resized.squeeze(0).squeeze(0).numpy()  # 11HW -> HW

        return sample

    def _torch_interpolation_mode(self, interpolation_method):
        """Convert interpolation method to torch mode string."""
        if interpolation_method == INTER_AREA:
            return "area"
        elif interpolation_method == INTER_NEAREST:
            return "nearest"
        elif interpolation_method == INTER_CUBIC:
            return "bicubic"
        elif interpolation_method == INTER_LINEAR:
            return "bilinear"
        else:
            return "bilinear"  # default


class NormalizeImage(object):
    """Normlize image by given mean and std.
    """

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample


class PrepareForNet(object):
    """Prepare sample for usage as network input.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])

        return sample
