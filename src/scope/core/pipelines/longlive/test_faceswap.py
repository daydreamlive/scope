"""
Face Swap Test Script for LongLive Pipeline with VACE.

This implements "Face Swapping" as shown in VACE community demos, combining:
- Face detection with colored structural lines (pose/expression guidance)
- Face inpainting masks (where to generate the new face)
- Reference face via first_frame_image (identity to swap to)

## How It Works (VACE Community Demo Style)

The technique uses VACE's inpainting capabilities with face-specific conditioning:

**Conditioning Frames (vace_input_frames)**:
- Original video with face region filled GRAY (not black)
- Colored lines for face structure ON TOP of gray:
  - Face contour/oval (light gray)
  - Eyes (cyan)
  - Eyebrows (light blue)
  - Lips (pink/red)
- Background and body are PRESERVED (not masked)

**Inpainting Masks (vace_input_masks)**:
- Binary masks covering the face region
- White = generate (face area), Black = preserve (background/body)

**Face Detection Methods**:
- "mediapipe" (recommended): MediaPipe Face Mesh with 478 landmarks for precise
  face feature detection. Draws smooth colored lines matching VACE demo style.
- "insightface": InsightFace 106-point landmarks (legacy option)
- "simple": Fixed ellipse mask (for testing only)

**Reference Face (vace_ref_images)**:
- Image of the target face identity
- Provided via vace_ref_images parameter (r2v mode)
- Sent on every chunk for consistent identity guidance

## Autoregressive Approach

For each chunk:
1. Preprocess frames: detect face with MediaPipe, fill gray, draw colored lines
2. Create face masks from face oval landmarks
3. Pass vace_ref_images with reference face identity
4. Generate with VACE to produce swapped face

Sources:
- VACE Project: https://ali-vilab.github.io/VACE-Page/
- Community Face Swap Demo: https://github.com/ali-vilab/VACE/discussions

Usage:
    python -m scope.core.pipelines.longlive.test_faceswap
"""

import time
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf
from PIL import Image

from scope.core.config import get_model_file_path, get_models_dir

from .pipeline import LongLivePipeline

# =============================================================================
# MediaPipe Face Mesh Types and Constants
# =============================================================================


class FaceMeshResult(NamedTuple):
    """Result from MediaPipe Face Mesh detection."""

    landmarks: np.ndarray  # [478, 2] normalized coordinates (x, y)
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels


# MediaPipe Face Mesh connection indices for drawing
# These define which landmarks to connect with lines for each face feature
FACEMESH_FACE_OVAL = [
    (10, 338),
    (338, 297),
    (297, 332),
    (332, 284),
    (284, 251),
    (251, 389),
    (389, 356),
    (356, 454),
    (454, 323),
    (323, 361),
    (361, 288),
    (288, 397),
    (397, 365),
    (365, 379),
    (379, 378),
    (378, 400),
    (400, 377),
    (377, 152),
    (152, 148),
    (148, 176),
    (176, 149),
    (149, 150),
    (150, 136),
    (136, 172),
    (172, 58),
    (58, 132),
    (132, 93),
    (93, 234),
    (234, 127),
    (127, 162),
    (162, 21),
    (21, 54),
    (54, 103),
    (103, 67),
    (67, 109),
    (109, 10),
]

FACEMESH_LEFT_EYE = [
    (263, 249),
    (249, 390),
    (390, 373),
    (373, 374),
    (374, 380),
    (380, 381),
    (381, 382),
    (382, 362),
    (362, 263),
    (263, 466),
    (466, 388),
    (388, 387),
    (387, 386),
    (386, 385),
    (385, 384),
    (384, 398),
    (398, 362),
]

FACEMESH_RIGHT_EYE = [
    (33, 7),
    (7, 163),
    (163, 144),
    (144, 145),
    (145, 153),
    (153, 154),
    (154, 155),
    (155, 133),
    (133, 33),
    (33, 246),
    (246, 161),
    (161, 160),
    (160, 159),
    (159, 158),
    (158, 157),
    (157, 173),
    (173, 133),
]

FACEMESH_LEFT_EYEBROW = [
    (276, 283),
    (283, 282),
    (282, 295),
    (295, 285),
    (285, 300),
    (300, 293),
    (293, 334),
    (334, 296),
    (296, 336),
]

FACEMESH_RIGHT_EYEBROW = [
    (46, 53),
    (53, 52),
    (52, 65),
    (65, 55),
    (55, 70),
    (70, 63),
    (63, 105),
    (105, 66),
    (66, 107),
]

FACEMESH_LIPS = [
    (61, 146),
    (146, 91),
    (91, 181),
    (181, 84),
    (84, 17),
    (17, 314),
    (314, 405),
    (405, 321),
    (321, 375),
    (375, 291),
    (291, 409),
    (409, 270),
    (270, 269),
    (269, 267),
    (267, 0),
    (0, 37),
    (37, 39),
    (39, 40),
    (40, 185),
    (185, 61),
    (61, 78),
    (78, 95),
    (95, 88),
    (88, 178),
    (178, 87),
    (87, 14),
    (14, 317),
    (317, 402),
    (402, 318),
    (318, 324),
    (324, 308),
    (308, 415),
    (415, 310),
    (310, 311),
    (311, 312),
    (312, 13),
    (13, 82),
    (82, 81),
    (81, 80),
    (80, 191),
    (191, 78),
]

FACEMESH_NOSE = [
    (168, 6),
    (6, 197),
    (197, 195),
    (195, 5),
    (5, 4),
    (4, 1),
    (1, 19),
    (19, 94),
    (94, 2),
]

# Colors matching VACE community demo style (RGB)
MEDIAPIPE_FACE_COLORS = {
    "face_oval": (200, 200, 200),  # Light gray for face contour
    "left_eye": (255, 255, 0),  # Cyan for left eye
    "right_eye": (255, 255, 0),  # Cyan for right eye
    "left_eyebrow": (255, 150, 100),  # Light blue for eyebrows
    "right_eyebrow": (255, 150, 100),  # Light blue for eyebrows
    "lips": (150, 100, 255),  # Pink/red for lips
    "nose": (200, 200, 200),  # Light gray for nose
}

# ============================= CONFIGURATION =============================

CONFIG = {
    # ===== INPUT VIDEO =====
    # Video with face(s) to swap - will extract frames and detect faces
    "input_video": "controlnet_test/pexels_woman_book.mp4",
    # OR use a static image repeated as frames
    "input_image": None,  # "frontend/public/assets/woman1.jpg",
    # ===== REFERENCE FACE =====
    # Image of the target face to swap to
    "reference_face": "frontend/public/assets/xilin.png",
    # ===== FACE DETECTION =====
    # Method: "mediapipe" (478-point face mesh, recommended), "insightface" (106-point), or "simple" (fixed ellipse)
    "face_detection_method": "mediapipe",  # MediaPipe Face Mesh for best quality
    # Face mask expansion factor (1.0 = exact, 1.2 = 20% larger)
    "mask_expansion": 1.3,
    # ===== GENERATION PARAMETERS =====
    "prompt": "a person with natural face, high quality, detailed",
    "num_chunks": 8,
    "frames_per_chunk": 12,
    "height": 640,  # Must be divisible by 16
    "width": 352,
    "vace_context_scale": 1.0,
    # ===== MASK BEHAVIOR =====
    # True = inpaint face region (mask=1 where face is)
    # False = preserve face, inpaint background
    "inpaint_face": True,
    # ===== OUTPUT =====
    "output_dir": "vace_tests/faceswap",
    "vae_type": "tae",
    "save_intermediates": True,
}

# ========================= END CONFIGURATION =========================


# =============================================================================
# MediaPipe Face Mesh Detection and Drawing (Tasks API)
# =============================================================================

# Global cache for MediaPipe FaceLandmarker to avoid re-initialization
_mediapipe_cache = {}

# Path for MediaPipe model file
FACE_LANDMARKER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
FACE_LANDMARKER_MODEL_NAME = "face_landmarker.task"


def _download_mediapipe_model(models_dir: Path) -> Path:
    """Download MediaPipe FaceLandmarker model if not present."""
    import urllib.request

    model_path = models_dir / "mediapipe" / FACE_LANDMARKER_MODEL_NAME
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print("  Downloading MediaPipe FaceLandmarker model...")
        urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, model_path)
        print(f"  Saved to: {model_path}")

    return model_path


def get_mediapipe_face_landmarker(
    max_faces: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    """Get cached MediaPipe FaceLandmarker instance (Tasks API)."""
    global _mediapipe_cache

    cache_key = (max_faces, min_detection_confidence, min_tracking_confidence)
    if cache_key not in _mediapipe_cache:
        try:
            import mediapipe as mp
        except ImportError:
            print("MediaPipe not installed")
            print("  Install with: pip install mediapipe")
            raise

        # Download model if needed
        from scope.core.config import get_models_dir

        model_path = _download_mediapipe_model(get_models_dir())

        # Create FaceLandmarker with Tasks API
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        _mediapipe_cache[cache_key] = landmarker

    return _mediapipe_cache[cache_key]


def detect_face_mediapipe(
    frame: np.ndarray,
    max_faces: int = 1,
    min_confidence: float = 0.5,
) -> list[FaceMeshResult]:
    """
    Detect faces using MediaPipe FaceLandmarker (Tasks API).

    Returns list of FaceMeshResult with 478 landmarks per face.
    Landmarks are in pixel coordinates.
    """
    import mediapipe as mp

    height, width = frame.shape[:2]
    landmarker = get_mediapipe_face_landmarker(max_faces, min_confidence)

    # Convert frame to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Detect faces
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return []

    face_results = []
    for face_landmarks in result.face_landmarks:
        # Convert normalized landmarks to pixel coordinates
        landmarks = np.array([[lm.x * width, lm.y * height] for lm in face_landmarks])

        # Calculate bounding box from landmarks
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        x1, x2 = int(x_coords.min()), int(x_coords.max())
        y1, y2 = int(y_coords.min()), int(y_coords.max())

        face_results.append(
            FaceMeshResult(
                landmarks=landmarks,
                bbox=(x1, y1, x2, y2),
            )
        )

    return face_results


def draw_mediapipe_face_lines(
    canvas: np.ndarray,
    landmarks: np.ndarray,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Draw face mesh lines matching VACE community demo style.

    This draws smooth connected lines for each face feature:
    - Face oval (contour): light gray
    - Eyes: cyan
    - Eyebrows: light blue
    - Lips: pink/red
    - Nose: light gray (optional, subtle)

    Args:
        canvas: [H, W, 3] image to draw on (should be gray-filled face region)
        landmarks: [478, 2] array of landmark pixel coordinates
        line_thickness: thickness for lines
    """
    feature_connections = {
        "face_oval": FACEMESH_FACE_OVAL,
        "left_eye": FACEMESH_LEFT_EYE,
        "right_eye": FACEMESH_RIGHT_EYE,
        "left_eyebrow": FACEMESH_LEFT_EYEBROW,
        "right_eyebrow": FACEMESH_RIGHT_EYEBROW,
        "lips": FACEMESH_LIPS,
        # "nose": FACEMESH_NOSE,  # Often omitted in VACE style for cleaner look
    }

    for feature_name, connections in feature_connections.items():
        color = MEDIAPIPE_FACE_COLORS.get(feature_name, (200, 200, 200))

        for start_idx, end_idx in connections:
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue

            pt1 = (int(landmarks[start_idx][0]), int(landmarks[start_idx][1]))
            pt2 = (int(landmarks[end_idx][0]), int(landmarks[end_idx][1]))

            # Check bounds
            h, w = canvas.shape[:2]
            if (
                0 <= pt1[0] < w
                and 0 <= pt1[1] < h
                and 0 <= pt2[0] < w
                and 0 <= pt2[1] < h
            ):
                cv2.line(canvas, pt1, pt2, color, line_thickness, cv2.LINE_AA)

    return canvas


def create_face_mask_from_mediapipe(
    height: int,
    width: int,
    landmarks: np.ndarray,
    expansion: float = 1.3,
) -> np.ndarray:
    """
    Create face mask from MediaPipe landmarks using face oval + convex hull.

    Args:
        height: image height
        width: image width
        landmarks: [478, 2] array of landmark coordinates
        expansion: how much to expand the mask (1.0 = exact, 1.3 = 30% larger)
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    # Extract face oval landmarks (indices from FACEMESH_FACE_OVAL)
    oval_indices = list(set([idx for pair in FACEMESH_FACE_OVAL for idx in pair]))
    oval_points = landmarks[oval_indices].astype(np.int32)

    # Calculate center for expansion
    center = oval_points.mean(axis=0)

    # Expand points outward from center
    if expansion != 1.0:
        expanded_points = center + (oval_points - center) * expansion
        expanded_points = np.clip(
            expanded_points, [0, 0], [width - 1, height - 1]
        ).astype(np.int32)
    else:
        expanded_points = oval_points

    # Use convex hull for smooth mask
    hull = cv2.convexHull(expanded_points)
    cv2.fillConvexPoly(mask, hull, 255)

    return mask


def preprocess_faceswap_frames_mediapipe(
    frames: np.ndarray,
    mask_expansion: float = 1.3,
    gray_value: int = 128,
    line_thickness: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess frames for face swap using MediaPipe Face Mesh.

    This creates preprocessing matching the VACE community demo style:
    - vace_input_frames: Original video with face region filled gray + colored face lines
    - vace_input_masks: Binary masks where 1=face region (where to generate)

    The MediaPipe Face Mesh provides 478 landmarks for precise face feature drawing:
    - Face oval for contour
    - Eyes with proper shape
    - Eyebrows
    - Lips with inner/outer contours

    Args:
        frames: [F, H, W, 3] input video frames (uint8, RGB)
        mask_expansion: how much to expand face mask (1.0=exact, 1.3=30% larger)
        gray_value: gray level for face fill (0-255, default 128)
        line_thickness: thickness for face feature lines

    Returns:
        preprocessed_frames: [F, H, W, 3] frames with gray face + colored lines
        masks: [F, H, W] binary masks (255 where face is)
    """
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        print("MediaPipe not installed")
        print("  Install with: pip install mediapipe")
        raise

    num_frames, height, width, _ = frames.shape
    preprocessed_list = []
    masks_list = []

    print(f"  Preprocessing {num_frames} frames with MediaPipe Face Mesh...")
    last_landmarks = None
    last_mask = None

    for i in range(num_frames):
        frame = frames[i].copy()
        mask = np.zeros((height, width), dtype=np.uint8)

        # Detect faces with MediaPipe
        face_results = detect_face_mediapipe(frame, max_faces=1, min_confidence=0.5)

        if len(face_results) > 0:
            face = face_results[0]
            landmarks = face.landmarks

            # Create face mask from landmarks
            mask = create_face_mask_from_mediapipe(
                height, width, landmarks, expansion=mask_expansion
            )

            # Fill face region with gray
            bool_mask = mask > 0
            frame[bool_mask] = gray_value

            # Draw colored face lines ON TOP of gray region
            frame = draw_mediapipe_face_lines(frame, landmarks, line_thickness)

            last_landmarks = landmarks.copy()
            last_mask = mask.copy()

        else:
            # No face detected - use last valid detection for temporal consistency
            if last_mask is not None:
                mask = last_mask.copy()
                bool_mask = mask > 0
                frame[bool_mask] = gray_value
                if last_landmarks is not None:
                    frame = draw_mediapipe_face_lines(
                        frame, last_landmarks, line_thickness
                    )

        preprocessed_list.append(frame)
        masks_list.append(mask)

        # Progress indicator
        if (i + 1) % 20 == 0 or i == num_frames - 1:
            print(f"    Frame {i + 1}/{num_frames}")

    preprocessed_frames = np.array(preprocessed_list)
    masks = np.array(masks_list)

    return preprocessed_frames, masks


def load_video_frames(
    video_path: str,
    num_frames: int,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """
    Load frames from video file and resize.

    Returns [F, H, W, 3] uint8 array.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            # Loop video if needed
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize
        frame = cv2.resize(frame, (target_width, target_height))
        frames.append(frame)

    cap.release()
    return np.array(frames)


def load_and_resize_image(
    image_path: str,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """Load image and resize to target dimensions."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((target_width, target_height), Image.LANCZOS)
    return np.array(img)


def detect_faces_simple(
    frames: np.ndarray,
    mask_expansion: float = 1.3,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Simple face detection - assumes face is in upper-center region.

    This is a placeholder for more sophisticated face detection.
    For production, use InsightFace or MediaPipe.

    Returns:
        landmarks_frames: List of [H, W, 3] landmark visualizations (black bg, white points)
        mask_frames: List of [H, W] binary masks (255 where face is)
    """
    num_frames, height, width, _ = frames.shape
    landmarks_list = []
    masks_list = []

    # Assume face is in upper-center region
    # These are rough estimates - real detection would be better
    face_center_x = width // 2
    face_center_y = int(height * 0.35)  # Upper portion of frame
    face_radius = int(min(height, width) * 0.2 * mask_expansion)

    for i in range(num_frames):
        # Create landmark visualization (simplified - just face contour)
        landmarks = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw simplified face landmarks (oval shape with key points)
        # In production, use actual detected landmarks
        cv2.ellipse(
            landmarks,
            (face_center_x, face_center_y),
            (int(face_radius * 0.7), face_radius),
            0,
            0,
            360,
            (255, 255, 255),  # White landmarks
            2,
        )

        # Add some key face points
        eye_offset = int(face_radius * 0.3)
        nose_offset = int(face_radius * 0.1)
        mouth_offset = int(face_radius * 0.5)

        # Eyes
        cv2.circle(
            landmarks,
            (face_center_x - eye_offset, face_center_y - nose_offset),
            3,
            (255, 255, 255),
            -1,
        )
        cv2.circle(
            landmarks,
            (face_center_x + eye_offset, face_center_y - nose_offset),
            3,
            (255, 255, 255),
            -1,
        )
        # Nose
        cv2.circle(
            landmarks,
            (face_center_x, face_center_y + nose_offset),
            3,
            (255, 255, 255),
            -1,
        )
        # Mouth
        cv2.ellipse(
            landmarks,
            (face_center_x, face_center_y + mouth_offset),
            (int(eye_offset * 0.8), int(nose_offset * 0.6)),
            0,
            0,
            180,
            (255, 255, 255),
            2,
        )

        landmarks_list.append(landmarks)

        # Create face mask (ellipse covering face region)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(
            mask,
            (face_center_x, face_center_y),
            (int(face_radius * 0.8), face_radius),
            0,
            0,
            360,
            255,
            -1,  # Filled
        )
        masks_list.append(mask)

    return landmarks_list, masks_list


# =============================================================================
# NEW: Simple face lines preprocessing (matching VACE community demo style)
# =============================================================================

# Colors for face features (BGR for cv2, will convert to RGB)
FACE_LINE_COLORS = {
    "contour": (200, 200, 200),  # Light gray for face outline
    "left_eyebrow": (255, 100, 100),  # Blue-ish for eyebrows
    "right_eyebrow": (255, 100, 100),
    "left_eye": (255, 255, 0),  # Cyan for eyes
    "right_eye": (255, 255, 0),
    "nose": (200, 200, 200),  # Light gray for nose
    "outer_lip": (100, 100, 255),  # Red-ish for lips
    "inner_lip": (100, 100, 255),
}


def draw_simple_face_lines(
    canvas: np.ndarray,
    landmarks: np.ndarray,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Draw simple face lines matching VACE community demo style.

    This draws minimal face structure guidance:
    - Face contour outline
    - Eyebrows as simple curves
    - Eyes as simple ovals
    - Mouth as simple curve

    Uses colored lines (cyan eyes, pink mouth, gray contour) matching the example.

    Args:
        canvas: [H, W, 3] image to draw on (should be gray-filled face region)
        landmarks: [106, 2] array of InsightFace 106-point landmarks
        line_thickness: thickness for lines
    """
    height, width = canvas.shape[:2]

    # InsightFace 106-point indices for simple line drawing
    # We only draw key contours, not all points
    simple_regions = {
        # Face contour (jawline): indices 0-32, but we'll sample fewer points
        "contour": [0, 4, 8, 12, 16, 20, 24, 28, 32],
        # Left eyebrow: 33-42
        "left_eyebrow": [33, 35, 37, 39, 41, 42],
        # Right eyebrow: 43-51
        "right_eyebrow": [43, 45, 47, 49, 51],
        # Left eye: 64-75 (sample key points for oval)
        "left_eye": [64, 66, 68, 70, 72, 74, 64],  # closed loop
        # Right eye: 76-87
        "right_eye": [76, 78, 80, 82, 84, 86, 76],  # closed loop
        # Nose: just bridge and tip
        "nose": [52, 55, 57],
        # Outer lip
        "outer_lip": [88, 89, 90, 91, 92, 93, 94, 95, 88],  # closed loop
    }

    for region_name, indices in simple_regions.items():
        color = FACE_LINE_COLORS.get(region_name, (255, 255, 255))
        points = []

        for idx in indices:
            if idx < len(landmarks):
                px, py = int(landmarks[idx][0]), int(landmarks[idx][1])
                if 0 <= px < width and 0 <= py < height:
                    points.append((px, py))

        # Draw lines connecting points
        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(canvas, points[i], points[i + 1], color, line_thickness)

    return canvas


def preprocess_faceswap_frames(
    frames: np.ndarray,
    mask_expansion: float = 1.3,
    gray_value: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess frames for face swap matching VACE community demo style.

    Creates:
    - vace_input_frames: Original video with face region filled gray + simple face lines
    - vace_input_masks: Binary masks where 1=face region (where to generate)

    This matches the example preprocessing where:
    - Background/body is preserved
    - Face region is filled with gray
    - Simple colored lines show face structure (contour, eyes, mouth)

    Args:
        frames: [F, H, W, 3] input video frames (uint8, RGB)
        mask_expansion: how much to expand face mask (1.0=exact, 1.3=30% larger)
        gray_value: gray level for face fill (0-255, default 128)

    Returns:
        preprocessed_frames: [F, H, W, 3] frames with gray face + lines
        masks: [F, H, W] binary masks (255 where face is)
    """
    try:
        from insightface.app import FaceAnalysis  # noqa: F401
    except ImportError:
        print("InsightFace not installed")
        print("  Install with: pip install insightface onnxruntime-gpu")
        raise

    num_frames, height, width, _ = frames.shape
    preprocessed_list = []
    masks_list = []

    # Get cached face analyzer
    app = get_face_analyzer("buffalo_l")

    print(f"  Preprocessing {num_frames} frames for face swap...")
    last_landmarks = None
    last_mask = None

    for i in range(num_frames):
        frame = frames[i].copy()
        mask = np.zeros((height, width), dtype=np.uint8)

        # Detect faces
        faces = app.get(frame)

        if len(faces) > 0:
            face = faces[0]

            if "landmark_2d_106" in face:
                lmks = face["landmark_2d_106"]

                # Create face mask from landmarks
                mask = create_face_mask_from_landmarks(
                    height, width, lmks, expansion=mask_expansion
                )

                # Fill face region with gray
                bool_mask = mask > 0
                frame[bool_mask] = gray_value

                # Draw simple face lines ON TOP of gray region
                frame = draw_simple_face_lines(frame, lmks, line_thickness=2)

                last_landmarks = lmks.copy()
                last_mask = mask.copy()

            elif "kps" in face:
                # Fallback to bbox-based mask
                bbox = face["bbox"].astype(int)
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                w = int((x2 - x1) * mask_expansion)
                h = int((y2 - y1) * mask_expansion)
                cv2.ellipse(mask, (cx, cy), (w // 2, h // 2), 0, 0, 360, 255, -1)

                bool_mask = mask > 0
                frame[bool_mask] = gray_value
                # Can't draw detailed lines without 106 landmarks
                last_mask = mask.copy()

        else:
            # No face detected - use last valid detection
            if last_mask is not None:
                mask = last_mask.copy()
                bool_mask = mask > 0
                frame[bool_mask] = gray_value
                if last_landmarks is not None:
                    frame = draw_simple_face_lines(
                        frame, last_landmarks, line_thickness=2
                    )

        preprocessed_list.append(frame)
        masks_list.append(mask)

        # Progress indicator
        if (i + 1) % 20 == 0 or i == num_frames - 1:
            print(f"    Frame {i + 1}/{num_frames}")

    preprocessed_frames = np.array(preprocessed_list)
    masks = np.array(masks_list)

    return preprocessed_frames, masks


# =============================================================================
# Legacy 106-point landmark code (kept for reference)
# =============================================================================

# InsightFace 106-point landmark indices for face regions
# These define how to connect landmarks for structural guidance
INSIGHTFACE_106_CONNECTIONS = {
    # Face contour (jawline): 0-32
    "contour": list(range(33)),
    # Left eyebrow: 33-42
    "left_eyebrow": list(range(33, 43)),
    # Right eyebrow: 43-51
    "right_eyebrow": list(range(43, 52)),
    # Nose bridge and tip: 52-63
    "nose": list(range(52, 64)),
    # Left eye: 64-75 (closed contour)
    "left_eye": list(range(64, 76)) + [64],
    # Right eye: 76-87 (closed contour)
    "right_eye": list(range(76, 88)) + [76],
    # Outer lip: 88-95 (closed contour)
    "outer_lip": list(range(88, 96)) + [88],
    # Inner lip: 96-103 (closed contour)
    "inner_lip": list(range(96, 104)) + [96],
}


def draw_face_landmarks_106(
    canvas: np.ndarray,
    landmarks: np.ndarray,
    color: tuple[int, int, int] = (255, 255, 255),
    point_radius: int = 2,
    line_thickness: int = 1,
) -> np.ndarray:
    """
    Draw 106-point InsightFace landmarks with connecting lines.

    This provides structural guidance for face pose/expression by drawing:
    - Face contour (jawline)
    - Eyebrows
    - Eyes
    - Nose
    - Lips

    Args:
        canvas: [H, W, 3] image to draw on
        landmarks: [106, 2] array of landmark coordinates
        color: RGB color tuple
        point_radius: radius for landmark points
        line_thickness: thickness for connecting lines
    """
    height, width = canvas.shape[:2]

    # Draw connecting lines for each face region
    for region_name, indices in INSIGHTFACE_106_CONNECTIONS.items():
        points = []
        for idx in indices:
            if idx < len(landmarks):
                px, py = int(landmarks[idx][0]), int(landmarks[idx][1])
                if 0 <= px < width and 0 <= py < height:
                    points.append((px, py))

        # Draw lines connecting consecutive points
        for i in range(len(points) - 1):
            cv2.line(canvas, points[i], points[i + 1], color, line_thickness)

    # Draw points on top of lines
    for point in landmarks:
        px, py = int(point[0]), int(point[1])
        if 0 <= px < width and 0 <= py < height:
            cv2.circle(canvas, (px, py), point_radius, color, -1)

    return canvas


def create_face_mask_from_landmarks(
    height: int,
    width: int,
    landmarks: np.ndarray,
    expansion: float = 1.3,
    use_convex_hull: bool = True,
) -> np.ndarray:
    """
    Create face mask from landmarks using convex hull or bounding ellipse.

    Args:
        height: image height
        width: image width
        landmarks: [N, 2] array of landmark coordinates
        expansion: how much to expand the mask (1.0 = exact, 1.3 = 30% larger)
        use_convex_hull: if True, use convex hull; if False, use bounding ellipse
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    if len(landmarks) == 0:
        return mask

    # Get valid points within image bounds
    valid_points = []
    for point in landmarks:
        px, py = int(point[0]), int(point[1])
        if 0 <= px < width and 0 <= py < height:
            valid_points.append([px, py])

    if len(valid_points) < 3:
        return mask

    points = np.array(valid_points, dtype=np.int32)

    # Calculate center and expand points outward
    center = points.mean(axis=0)
    if expansion != 1.0:
        expanded_points = center + (points - center) * expansion
        expanded_points = np.clip(
            expanded_points, [0, 0], [width - 1, height - 1]
        ).astype(np.int32)
    else:
        expanded_points = points

    if use_convex_hull:
        # Use convex hull for natural face shape
        hull = cv2.convexHull(expanded_points)
        cv2.fillConvexPoly(mask, hull, 255)
    else:
        # Use bounding ellipse
        if len(expanded_points) >= 5:
            ellipse = cv2.fitEllipse(expanded_points)
            cv2.ellipse(mask, ellipse, 255, -1)
        else:
            # Fallback to convex hull if not enough points for ellipse
            hull = cv2.convexHull(expanded_points)
            cv2.fillConvexPoly(mask, hull, 255)

    return mask


# Global cache for FaceAnalysis to avoid re-initialization
_face_analyzer_cache = {}


def get_face_analyzer(model_name: str = "buffalo_l") -> "FaceAnalysis":
    """Get cached FaceAnalysis instance."""
    global _face_analyzer_cache

    if model_name not in _face_analyzer_cache:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name=model_name, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        _face_analyzer_cache[model_name] = app

    return _face_analyzer_cache[model_name]


def detect_faces_insightface(
    frames: np.ndarray,
    mask_expansion: float = 1.3,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Detect faces using InsightFace library with 106-point landmarks.

    This provides proper structural guidance for face swapping by:
    1. Detecting faces with InsightFace's buffalo_l model
    2. Drawing connected 106-point landmarks (contour, eyes, nose, lips)
    3. Creating convex hull masks from landmark points

    Returns:
        landmarks_frames: List of [H, W, 3] landmark visualizations
        mask_frames: List of [H, W] binary masks
    """
    try:
        from insightface.app import FaceAnalysis  # noqa: F401
    except ImportError:
        print("InsightFace not installed, falling back to simple detection")
        print("  Install with: pip install insightface onnxruntime-gpu")
        return detect_faces_simple(frames, mask_expansion)

    num_frames, height, width, _ = frames.shape
    landmarks_list = []
    masks_list = []

    # Get cached face analyzer
    try:
        app = get_face_analyzer("buffalo_l")
    except Exception as e:
        print(f"Failed to initialize InsightFace: {e}")
        print("  Falling back to simple detection")
        return detect_faces_simple(frames, mask_expansion)

    print(f"  Processing {num_frames} frames with InsightFace...")
    last_valid_landmarks = None
    last_valid_mask = None

    for i in range(num_frames):
        frame = frames[i]
        landmarks = np.zeros((height, width, 3), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)

        # Detect faces
        faces = app.get(frame)

        if len(faces) > 0:
            face = faces[0]  # Use first detected face

            # Draw 106-point landmarks with connections
            if "landmark_2d_106" in face:
                lmks = face["landmark_2d_106"]
                landmarks = draw_face_landmarks_106(landmarks, lmks)
                mask = create_face_mask_from_landmarks(
                    height, width, lmks, expansion=mask_expansion
                )
                last_valid_landmarks = landmarks.copy()
                last_valid_mask = mask.copy()
            # Fallback to 5-point keypoints if 106 not available
            elif "kps" in face:
                for point in face["kps"]:
                    px, py = int(point[0]), int(point[1])
                    if 0 <= px < width and 0 <= py < height:
                        cv2.circle(landmarks, (px, py), 4, (255, 255, 255), -1)

                # Use bbox for mask with kps fallback
                bbox = face["bbox"].astype(int)
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                w = int((x2 - x1) * mask_expansion)
                h = int((y2 - y1) * mask_expansion)
                cv2.ellipse(mask, (cx, cy), (w // 2, h // 2), 0, 0, 360, 255, -1)
                last_valid_landmarks = landmarks.copy()
                last_valid_mask = mask.copy()
        else:
            # No face detected - use last valid detection for temporal consistency
            if last_valid_landmarks is not None:
                landmarks = last_valid_landmarks.copy()
                mask = last_valid_mask.copy()

        landmarks_list.append(landmarks)
        masks_list.append(mask)

        # Progress indicator
        if (i + 1) % 20 == 0 or i == num_frames - 1:
            print(f"    Frame {i + 1}/{num_frames}")

    return landmarks_list, masks_list


def preprocess_frames_for_vace(
    frames: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Preprocess frames for VACE input.

    Converts [F, H, W, C] uint8 [0, 255] to [1, C, F, H, W] float [-1, 1].
    """
    tensor = torch.from_numpy(frames).float() / 255.0
    tensor = tensor * 2.0 - 1.0  # [-1, 1]
    tensor = tensor.permute(0, 3, 1, 2)  # [F, C, H, W]
    tensor = tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, C, F, H, W]
    return tensor.to(device)


def preprocess_masks_for_vace(
    masks: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Preprocess masks for VACE input.

    Converts [F, H, W] uint8 [0, 255] to [1, 1, F, H, W] float [0, 1].
    """
    tensor = torch.from_numpy(masks).float() / 255.0  # [F, H, W]
    tensor = tensor.unsqueeze(1)  # [F, 1, H, W]
    tensor = tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 1, F, H, W]
    return tensor.to(device)


def resolve_path(path_str: str, relative_to: Path) -> Path:
    """Resolve path relative to base directory."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (relative_to / path).resolve()


def main():
    print("=" * 80)
    print("  Face Swap Test with VACE")
    print("  (Face Detection -> Landmarks + Masks -> VACE Inpainting)")
    print("=" * 80)

    config = CONFIG

    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent.parent.parent
    output_dir = resolve_path(config["output_dir"], project_root)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nOutput directory: {output_dir}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Calculate total frames
    total_frames = config["num_chunks"] * config["frames_per_chunk"]
    print(f"Total frames needed: {total_frames}")

    # Load input frames (video or image)
    print("\nLoading input frames...")
    if config["input_video"]:
        video_path = resolve_path(config["input_video"], project_root)
        if not video_path.exists():
            print(f"  Warning: Video not found at {video_path}")
            print("  Creating synthetic test frames instead...")
            # Create synthetic test frames with a simple face pattern
            input_frames = create_synthetic_face_video(
                total_frames, config["height"], config["width"]
            )
        else:
            input_frames = load_video_frames(
                str(video_path),
                total_frames,
                config["height"],
                config["width"],
            )
    elif config["input_image"]:
        img_path = resolve_path(config["input_image"], project_root)
        single_frame = load_and_resize_image(
            str(img_path),
            config["height"],
            config["width"],
        )
        # Repeat for all frames
        input_frames = np.tile(single_frame[np.newaxis, ...], (total_frames, 1, 1, 1))
    else:
        print("  No input specified, creating synthetic test frames...")
        input_frames = create_synthetic_face_video(
            total_frames, config["height"], config["width"]
        )

    print(f"  Input frames shape: {input_frames.shape}")

    # Load reference face image
    print("\nLoading reference face...")
    ref_path = resolve_path(config["reference_face"], project_root)
    if not ref_path.exists():
        print(f"  Warning: Reference face not found at {ref_path}")
        print("  Using first input frame as reference instead")
        ref_face = input_frames[0]
    else:
        ref_face = load_and_resize_image(
            str(ref_path),
            config["height"],
            config["width"],
        )
    print(f"  Reference face shape: {ref_face.shape}")

    # Preprocess frames for face swap (VACE community demo style)
    # This creates:
    # - preprocessed_frames: Original video with face region gray + colored face lines
    # - mask_frames: Binary masks where 1=face region (where to generate)
    face_method = config["face_detection_method"]
    print(f"\nPreprocessing frames for face swap using {face_method}...")

    if face_method == "mediapipe":
        # MediaPipe Face Mesh - 478 landmarks, best quality colored lines
        preprocessed_frames, mask_frames = preprocess_faceswap_frames_mediapipe(
            input_frames,
            mask_expansion=config["mask_expansion"],
            gray_value=128,
            line_thickness=2,
        )
    elif face_method == "insightface":
        # InsightFace 106-point landmarks - fallback option
        preprocessed_frames, mask_frames = preprocess_faceswap_frames(
            input_frames,
            mask_expansion=config["mask_expansion"],
            gray_value=128,
        )
    else:
        # Simple detection - fixed ellipse, for testing only
        landmarks_list, mask_frames_list = detect_faces_simple(
            input_frames,
            mask_expansion=config["mask_expansion"],
        )
        # For simple detection, just create gray-filled frames
        preprocessed_frames = input_frames.copy()
        for i in range(len(preprocessed_frames)):
            bool_mask = mask_frames_list[i] > 0
            preprocessed_frames[i][bool_mask] = 128
        mask_frames = np.array(mask_frames_list)

    print(f"  Preprocessed frames shape: {preprocessed_frames.shape}")
    print(f"  Masks shape: {mask_frames.shape}")

    # Save intermediate visualizations
    if config["save_intermediates"]:
        print("\nSaving intermediate videos...")

        # Input video (original)
        export_to_video(input_frames / 255.0, output_dir / "input_video.mp4", fps=16)
        print("  Saved: input_video.mp4")

        # Preprocessed frames (gray face + simple lines - this is vace_input_frames)
        export_to_video(
            preprocessed_frames / 255.0, output_dir / "preprocessed.mp4", fps=16
        )
        print("  Saved: preprocessed.mp4")

        # Mask overlaid on input (magenta where mask is)
        mask_overlay_frames = input_frames.copy().astype(float)
        for i in range(len(mask_overlay_frames)):
            bool_mask = mask_frames[i] > 0
            # Magenta tint on masked region
            mask_overlay_frames[i][bool_mask] = (
                mask_overlay_frames[i][bool_mask] * 0.5 + np.array([255, 0, 255]) * 0.5
            )
        export_to_video(
            mask_overlay_frames.astype(np.uint8) / 255.0,
            output_dir / "mask_overlay.mp4",
            fps=16,
        )
        print("  Saved: mask_overlay.mp4")

        # Reference face
        Image.fromarray(ref_face).save(output_dir / "reference_face.png")
        print("  Saved: reference_face.png")

    # Initialize pipeline
    print("\nInitializing LongLive pipeline with VACE...")

    vace_path = str(
        get_model_file_path("Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors")
    )

    pipeline_config = OmegaConf.create(
        {
            "model_dir": str(get_models_dir()),
            "generator_path": str(
                get_model_file_path("LongLive-1.3B/models/longlive_base.pt")
            ),
            "lora_path": str(get_model_file_path("LongLive-1.3B/models/lora.pt")),
            "vace_path": vace_path,
            "text_encoder_path": str(
                get_model_file_path(
                    "WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
                )
            ),
            "tokenizer_path": str(
                get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")
            ),
            "model_config": OmegaConf.load(script_dir / "model.yaml"),
            "height": config["height"],
            "width": config["width"],
            "vae_type": config["vae_type"],
        }
    )

    # VACE in_dim = 96 for masked/inpainting mode
    pipeline_config.model_config.base_model_kwargs = (
        pipeline_config.model_config.base_model_kwargs or {}
    )
    pipeline_config.model_config.base_model_kwargs["vace_in_dim"] = 96

    pipeline = LongLivePipeline(pipeline_config, device=device, dtype=torch.bfloat16)
    print("Pipeline ready\n")

    # Preprocess for VACE
    # For face swap:
    # - vace_input_frames = preprocessed video (gray face + colored lines)
    # - vace_input_masks = face masks (white where face is = generate there)
    # - vace_ref_images = reference face path (for identity via r2v mode)
    vace_input_tensor = preprocess_frames_for_vace(preprocessed_frames, device)
    vace_mask_tensor = preprocess_masks_for_vace(mask_frames, device)

    print(f"VACE input tensor (preprocessed): {vace_input_tensor.shape}")
    print(f"VACE mask tensor (face regions): {vace_mask_tensor.shape}")
    print(f"Reference face (vace_ref_images): {ref_path}")

    # Generate video
    print("\n" + "=" * 40)
    print("  GENERATING VIDEO")
    print("=" * 40)

    outputs = []
    latency_measures = []
    frames_per_chunk = config["frames_per_chunk"]

    for chunk_idx in range(config["num_chunks"]):
        start_time = time.time()

        start_frame = chunk_idx * frames_per_chunk
        end_frame = start_frame + frames_per_chunk

        # Extract chunk tensors
        input_chunk = vace_input_tensor[:, :, start_frame:end_frame, :, :]
        mask_chunk = vace_mask_tensor[:, :, start_frame:end_frame, :, :]

        # Build kwargs for this chunk
        kwargs = {
            "prompts": [{"text": config["prompt"], "weight": 100}],
            "vace_context_scale": config["vace_context_scale"],
            "vace_input_frames": input_chunk,
            "vace_input_masks": mask_chunk,
            "vace_ref_images": [str(ref_path)],  # Reference face path(s) via r2v mode
        }

        mode_str = "inpaint + vace_ref_images (r2v identity)"
        print(f"Chunk {chunk_idx}: frames {start_frame}-{end_frame} ({mode_str})")

        # Generate
        output = pipeline(**kwargs)

        latency = time.time() - start_time
        fps = output.shape[0] / latency
        print(f"  -> {output.shape[0]} frames, {latency:.1f}s, {fps:.1f} fps")

        latency_measures.append(latency)
        outputs.append(output.detach().cpu())

    # Concatenate outputs
    output_video = torch.concat(outputs)
    print(f"\nFinal output: {output_video.shape}")

    # Save output
    output_video_np = output_video.contiguous().numpy()
    output_video_np = np.clip(output_video_np, 0.0, 1.0)

    output_path = output_dir / "faceswap_output.mp4"
    export_to_video(output_video_np, output_path, fps=16)
    print(f"\nSaved: {output_path}")

    # Create side-by-side comparison
    if config["save_intermediates"]:
        print("\nCreating comparison video...")

        # Read output video frames
        cap = cv2.VideoCapture(str(output_path))
        output_frames_list = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            output_frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        # Create 2x2 grid:
        # [preprocessed (gray+lines) | mask overlay on input]
        # [input video               | output video          ]
        min_frames = min(
            len(input_frames), len(output_frames_list), len(preprocessed_frames)
        )
        comparison_frames = []

        for i in range(min_frames):
            # Top left: preprocessed frame (gray face + simple lines)
            top_left = preprocessed_frames[i].copy()

            # Top right: mask overlaid on input (magenta)
            top_right = input_frames[i].copy().astype(float)
            bool_mask = mask_frames[i] > 0
            top_right[bool_mask] = (
                top_right[bool_mask] * 0.5 + np.array([255, 0, 255]) * 0.5
            )
            top_right = top_right.astype(np.uint8)

            top_row = np.hstack([top_left, top_right])

            # Bottom row: input and output
            bottom_left = input_frames[i]
            bottom_right = (
                output_frames_list[i]
                if i < len(output_frames_list)
                else np.zeros_like(bottom_left)
            )
            bottom_row = np.hstack([bottom_left, bottom_right])

            comparison = np.vstack([top_row, bottom_row])
            comparison_frames.append(comparison)

        comparison_array = np.array(comparison_frames) / 255.0
        export_to_video(comparison_array, output_dir / "comparison.mp4", fps=16)
        print("  Saved: comparison.mp4")

    # Stats
    total_time = sum(latency_measures)
    avg_fps = output_video.shape[0] / total_time
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Average FPS: {avg_fps:.2f}")

    print("\n" + "=" * 80)
    print("  COMPLETE")
    print("=" * 80)
    print(f"\nOutput: {output_path}")
    if config["save_intermediates"]:
        print("Intermediates:")
        print("  - input_video.mp4: Original input video")
        print("  - preprocessed.mp4: Gray face + simple lines (vace_input_frames)")
        print("  - mask_overlay.mp4: Face mask overlaid on input (magenta)")
        print("  - reference_face.png: Target face identity (first_frame_image)")
        print("  - comparison.mp4: 2x2 grid [preprocessed|mask] / [input|output]")


def create_synthetic_face_video(
    num_frames: int,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Create synthetic test video with a simple animated face.
    Used when no input video is provided.
    """
    frames = []

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Background gradient
        for y in range(height):
            frame[y, :, 0] = int(100 + 50 * y / height)  # R
            frame[y, :, 1] = int(80 + 40 * y / height)  # G
            frame[y, :, 2] = int(60 + 30 * y / height)  # B

        # Face position (slight animation)
        t = i / max(num_frames - 1, 1)
        face_x = width // 2 + int(20 * np.sin(2 * np.pi * t))
        face_y = int(height * 0.4) + int(10 * np.cos(2 * np.pi * t))
        face_radius = int(min(height, width) * 0.15)

        # Draw face (skin tone)
        cv2.ellipse(
            frame,
            (face_x, face_y),
            (int(face_radius * 0.8), face_radius),
            0,
            0,
            360,
            (210, 180, 150),  # Skin tone
            -1,
        )

        # Eyes
        eye_y = face_y - int(face_radius * 0.2)
        eye_offset = int(face_radius * 0.35)
        cv2.circle(frame, (face_x - eye_offset, eye_y), 8, (255, 255, 255), -1)
        cv2.circle(frame, (face_x + eye_offset, eye_y), 8, (255, 255, 255), -1)
        cv2.circle(frame, (face_x - eye_offset, eye_y), 4, (50, 50, 50), -1)
        cv2.circle(frame, (face_x + eye_offset, eye_y), 4, (50, 50, 50), -1)

        # Nose
        nose_y = face_y + int(face_radius * 0.1)
        cv2.line(frame, (face_x, eye_y + 10), (face_x, nose_y), (180, 150, 120), 2)

        # Mouth (animated smile)
        mouth_y = face_y + int(face_radius * 0.5)
        smile_angle = 30 + 20 * np.sin(4 * np.pi * t)
        cv2.ellipse(
            frame,
            (face_x, mouth_y),
            (int(face_radius * 0.3), int(face_radius * 0.15)),
            0,
            0,
            180,
            (150, 100, 100),
            2,
        )

        # Hair
        hair_y = face_y - face_radius
        cv2.ellipse(
            frame,
            (face_x, hair_y),
            (int(face_radius * 0.9), int(face_radius * 0.4)),
            0,
            180,
            360,
            (60, 40, 30),
            -1,
        )

        frames.append(frame)

    return np.array(frames)


if __name__ == "__main__":
    main()
