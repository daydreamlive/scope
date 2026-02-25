"""Drawing utilities for overlaying Moondream results on video frames."""

from __future__ import annotations

import torch
from PIL import Image, ImageDraw, ImageFont


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a (1, H, W, C) or (H, W, C) tensor in [0, 255] to a PIL Image."""
    frame = tensor.squeeze(0) if tensor.ndim == 4 else tensor
    frame_np = frame.cpu().byte().numpy()
    return Image.fromarray(frame_np, mode="RGB")


def pil_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    """Convert a PIL Image to a (1, H, W, C) tensor in [0, 1] float range."""
    import numpy as np

    arr = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def _get_font(font_scale: float) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Get a font at the requested scale."""
    size = int(16 * font_scale)
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()


def draw_bounding_boxes(
    image: Image.Image,
    objects: list[dict],
    opacity: float = 0.8,
    font_scale: float = 1.0,
) -> Image.Image:
    """Draw bounding boxes on the image.

    Args:
        image: PIL Image to draw on.
        objects: List of dicts with x_min, y_min, x_max, y_max (normalized 0-1).
        opacity: Overlay opacity (0-1).
        font_scale: Text size multiplier.

    Returns:
        Annotated PIL Image.
    """
    if not objects:
        return image

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    font = _get_font(font_scale)
    w, h = image.size

    box_color = (0, 220, 0)  # Green
    alpha = int(255 * opacity)
    fill_color = (0, 220, 0, int(40 * opacity))

    for i, obj in enumerate(objects):
        x1 = int(obj["x_min"] * w)
        y1 = int(obj["y_min"] * h)
        x2 = int(obj["x_max"] * w)
        y2 = int(obj["y_max"] * h)

        # Draw filled rectangle with transparency
        rect_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        rect_draw = ImageDraw.Draw(rect_overlay)
        rect_draw.rectangle([x1, y1, x2, y2], fill=fill_color)
        overlay = Image.alpha_composite(overlay.convert("RGBA"), rect_overlay).convert(
            "RGB"
        )
        draw = ImageDraw.Draw(overlay)

        # Draw border
        line_width = max(2, int(2 * font_scale))
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=line_width)

        # Draw label
        label = f"#{i + 1}"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        label_bg = [x1, y1 - text_h - 4, x1 + text_w + 6, y1]
        if label_bg[1] < 0:
            label_bg = [x1, y2, x1 + text_w + 6, y2 + text_h + 4]
        draw.rectangle(label_bg, fill=(0, 0, 0, alpha))
        draw.text((label_bg[0] + 3, label_bg[1] + 2), label, fill=box_color, font=font)

    return overlay


def draw_points(
    image: Image.Image,
    points: list[dict],
    opacity: float = 0.8,
    font_scale: float = 1.0,
) -> Image.Image:
    """Draw point markers on the image.

    Args:
        image: PIL Image to draw on.
        points: List of dicts with x, y (normalized 0-1).
        opacity: Overlay opacity (0-1).
        font_scale: Text size multiplier.

    Returns:
        Annotated PIL Image.
    """
    if not points:
        return image

    draw = ImageDraw.Draw(image)
    font = _get_font(font_scale)
    w, h = image.size
    radius = max(4, int(6 * font_scale))
    point_color = (220, 50, 50)  # Red

    for i, pt in enumerate(points):
        cx = int(pt["x"] * w)
        cy = int(pt["y"] * h)

        # Filled circle
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=point_color,
            outline=(255, 255, 255),
            width=max(1, int(font_scale)),
        )

        # Label
        label = f"#{i + 1}"
        draw.text(
            (cx + radius + 3, cy - radius),
            label,
            fill=point_color,
            font=font,
        )

    return image


def draw_text_overlay(
    image: Image.Image,
    text: str,
    opacity: float = 0.8,
    font_scale: float = 1.0,
    position: str = "bottom",
) -> Image.Image:
    """Draw a text bar overlay on the image.

    Args:
        image: PIL Image to draw on.
        text: Text to display.
        opacity: Background opacity (0-1).
        font_scale: Text size multiplier.
        position: "top" or "bottom".

    Returns:
        Annotated PIL Image.
    """
    if not text:
        return image

    font = _get_font(font_scale)
    w, h = image.size

    # Wrap text to fit width
    draw_temp = ImageDraw.Draw(image)
    max_text_width = w - 20
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        bbox = draw_temp.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= max_text_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    if not lines:
        return image

    # Calculate bar height
    line_height = draw_temp.textbbox((0, 0), "Ag", font=font)[3] + 4
    bar_height = len(lines) * line_height + 16
    bar_height = min(bar_height, h // 3)  # Cap at 1/3 of image height

    # Draw semi-transparent background bar
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    alpha = int(180 * opacity)

    if position == "top":
        bar_y = 0
    else:
        bar_y = h - bar_height

    overlay_draw.rectangle([0, bar_y, w, bar_y + bar_height], fill=(0, 0, 0, alpha))
    result = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(result)

    # Draw text lines
    text_y = bar_y + 8
    for line in lines:
        if text_y + line_height > bar_y + bar_height:
            break
        draw.text((10, text_y), line, fill=(255, 255, 255), font=font)
        text_y += line_height

    return result
