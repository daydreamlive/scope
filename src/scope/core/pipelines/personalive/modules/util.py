import importlib
import os
import os.path as osp
import shutil
import sys
from pathlib import Path

import av
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
import cv2


def save_checkpoint(model, save_dir, prefix, ckpt_num, logger, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    state_dict = model.state_dict()
    torch.save(state_dict, save_path)


def create_code_snapshot(root, dst_path, extensions=(".py", ".h", ".cpp", ".cu", ".cc", ".cuh", ".json", ".sh", ".bat", ".yaml"), exclude=()):
    """Creates tarball with the source code"""
    import tarfile
    from pathlib import Path
    with tarfile.open(str(dst_path), "w:gz") as tar:
        for path in Path(root).rglob("*"):
            if '.git' in path.parts:
                continue
            exclude_flag = False
            if len(exclude) > 0:
                for k in exclude:
                    if k in path.parts:
                        exclude_flag = True
            if exclude_flag:
                continue
            if path.suffix.lower() in extensions:
                try:
                    tar.add(path.as_posix(), arcname=path.relative_to(
                        root).as_posix(), recursive=True)
                except:
                    print(path)
                    assert False, 'Error occur in create_code_snapshot'

def seed_everything(seed):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


def import_filename(filename):
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def delete_additional_ckpt(base_path, num_keep):
    dirs = []
    for d in os.listdir(base_path):
        if d.startswith("checkpoint-"):
            dirs.append(d)
    num_tot = len(dirs)
    if num_tot <= num_keep:
        return
    # ensure ckpt is sorted and delete the ealier!
    del_dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))[: num_tot - num_keep]
    for d in del_dirs:
        path_to_dir = osp.join(base_path, d)
        if osp.exists(path_to_dir):
            shutil.rmtree(path_to_dir)


def save_videos_from_pil(pil_images, path, fps=8, crf=None):
    import av

    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = pil_images[0].size

    if save_fmt == ".mp4":
        if True:
            codec = "libx264"
            container = av.open(path, "w")
            stream = container.add_stream(codec, rate=fps)

            stream.width = width
            stream.height = height
            if crf is not None:
                stream.options = {'crf': str(crf)}

            for pil_image in pil_images:
                # pil_image = Image.fromarray(image_arr).convert("RGB")
                av_frame = av.VideoFrame.from_image(pil_image)
                container.mux(stream.encode(av_frame))
            container.mux(stream.encode())
            container.close()
        else:

            video_writer = cv2.VideoWriter(
                path.replace('.mp4', '_cv.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)
            )
            for pil_image in pil_images:
                img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                video_writer.write(img_np)
            video_writer.release()

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")


def save_videos_grid(videos_, path: str, rescale=False, n_rows=6, fps=8, crf=None):
    if not isinstance(videos_, list): videos_ = [videos_]

    outputs = []
    vid_len = videos_[0].shape[2]
    for i in range(vid_len):
        output = []
        for videos in videos_:
            videos = rearrange(videos, "b c t h w -> t b c h w")
            height, width = videos.shape[-2:]

            x = torchvision.utils.make_grid(videos[i], nrow=n_rows)  # (c h w)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
            if rescale:
                x = (x + 1.0) / 2.0  # -1,1 -> 0,1
            x = (x * 255).numpy().astype(np.uint8)
            output.append(x)

        output = Image.fromarray(np.concatenate(output, axis=0))
        outputs.append(output)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_videos_from_pil(outputs, path, fps, crf)


def save_videos_grid_ori(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    height, width = videos.shape[-2:]
    outputs = []

    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)  # (c h w)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)

        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_videos_from_pil(outputs, path, fps)


def read_frames(video_path):
    container = av.open(video_path)

    video_stream = next(s for s in container.streams if s.type == "video")
    frames = []
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            image = Image.frombytes(
                "RGB",
                (frame.width, frame.height),
                frame.to_rgb().to_ndarray(),
            )
            frames.append(image)

    return frames


def get_fps(video_path):
    container = av.open(video_path)
    video_stream = next(s for s in container.streams if s.type == "video")
    fps = video_stream.average_rate
    container.close()
    return fps

def draw_keypoints(keypoints, height=512, width=512, device="cuda"):
    colors = torch.tensor([
        [255, 0, 0],
        [255, 255, 0],
        [0, 255, 0],
        [0, 255, 255],
        [0, 0, 255],
        [255, 0, 255],
        [255, 0, 85],
    ], device=device, dtype=torch.float32)

    selected = torch.tensor([1, 2, 3, 4, 12, 15, 20], device=device)
    B = keypoints.shape[0]

    # [B, len(selected), 2]
    pts = keypoints[:, selected] * 0.5 + 0.5
    pts[..., 0] *= width
    pts[..., 1] *= height
    pts = pts.long()

    canvas = torch.zeros((B, 3, height, width), device=device)
    radius = 4

    for i, color in enumerate(colors):
        x = pts[:, i, 0]
        y = pts[:, i, 1]
        mask = (
            (x[:, None, None] - torch.arange(width, device=device)) ** 2
            + (y[:, None, None] - torch.arange(height, device=device)[:, None]) ** 2
        ) <= radius**2
        canvas[:, 0] += color[0] / 255.0 * mask
        canvas[:, 1] += color[1] / 255.0 * mask
        canvas[:, 2] += color[2] / 255.0 * mask

    return canvas.clamp(0, 1)

def get_boxes(keypoints, height=512, width=512):
    selected = torch.tensor([1, 2, 3, 4, 12, 15, 20])

    # [B, len(selected), 2]
    pts = keypoints[:, selected] * 0.5 + 0.5
    pts[..., 0] *= width
    pts[..., 1] *= height
    pts = pts.long()

    cx = pts[..., 0].float().mean(dim=1)   # [B]
    cy = pts[..., 1].float().mean(dim=1)   # [B]

    min_y = pts[..., 1].float().min(dim=1)[0]  # [B]

    side = (cy - min_y) * 2.0
    side = side * 1.7

    x1 = (cx - side / 2 * 0.95).clamp(0, width - 1).long()
    y1 = (cy - side / 2 * 0.95).clamp(0, height - 1).long()
    x2 = (cx + side / 2 * 1.05).clamp(0, width - 1).long()
    y2 = (cy + side / 2 * 1.05).clamp(0, height - 1).long()
    
    boxes = torch.stack([x1, y1, x2, y2], dim=1)   # [B, 4]

    return boxes

def crop_face(image_pil, face_mesh):
    image = np.array(image_pil)
    h, w = image.shape[:2]
    results = face_mesh.process(image)
    face_landmarks = results.multi_face_landmarks[0]
    coords = [(int(l.x * w), int(l.y * h)) for l in face_landmarks.landmark]
    xs, ys = zip(*coords)
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    face_box = (x1, y1, x2, y2)

    left, top, right, bot = scale_bb(face_box, scale=1.1, size=image.shape[:2])

    face_patch = image[int(top) : int(bot), int(left) : int(right)]

    return face_patch

def scale_bb(bbox, scale, size):
    left, top, right, bot = bbox
    width = right - left
    height = bot - top
    length = max(width, height) * scale
    center_X = (left + right) * 0.5
    center_Y = (top + bot) * 0.5
    left, top, right, bot = [
        center_X - length / 2,
        center_Y - length / 2,
        center_X + length / 2,
        center_Y + length / 2,
    ]
    left = max(0, left)
    top = max(0, top)
    right = min(size[1] - 1, right)
    bot = min(size[0] - 1, bot)
    return np.array([left, top, right, bot])