"""Face detection component using MediaPipe for PersonaLive pipeline."""

import logging
from typing import Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available. Face detection will be disabled.")


class FaceDetector:
    """Face detector using MediaPipe Face Mesh.

    Extracts face regions from images for motion encoding.
    """

    def __init__(self, static_image_mode: bool = True, max_num_faces: int = 1):
        """Initialize the face detector.

        Args:
            static_image_mode: If True, treats input as unrelated images (no tracking).
            max_num_faces: Maximum number of faces to detect.
        """
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError(
                "MediaPipe is required for PersonaLive face detection. "
                "Install with: pip install mediapipe"
            )

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect_face_landmarks(self, image: np.ndarray) -> list | None:
        """Detect face landmarks in an image.

        Args:
            image: RGB image as numpy array (H, W, 3), uint8.

        Returns:
            List of face landmarks or None if no face detected.
        """
        results = self.face_mesh.process(image)
        if results.multi_face_landmarks:
            return results.multi_face_landmarks
        return None

    def get_face_bounding_box(
        self,
        image: np.ndarray,
        padding_ratio: float = 0.2
    ) -> Tuple[int, int, int, int] | None:
        """Get bounding box for the detected face.

        Args:
            image: RGB image as numpy array (H, W, 3).
            padding_ratio: Extra padding around the face box.

        Returns:
            Tuple of (left, top, right, bottom) or None if no face.
        """
        landmarks = self.detect_face_landmarks(image)
        if landmarks is None:
            return None

        h, w = image.shape[:2]
        face_landmarks = landmarks[0]

        # Get all landmark coordinates
        xs = [lm.x * w for lm in face_landmarks.landmark]
        ys = [lm.y * h for lm in face_landmarks.landmark]

        # Compute bounding box
        left = min(xs)
        right = max(xs)
        top = min(ys)
        bottom = max(ys)

        # Add padding
        box_w = right - left
        box_h = bottom - top
        pad_w = box_w * padding_ratio
        pad_h = box_h * padding_ratio

        left = max(0, int(left - pad_w))
        top = max(0, int(top - pad_h))
        right = min(w, int(right + pad_w))
        bottom = min(h, int(bottom + pad_h))

        return (left, top, right, bottom)

    def crop_face(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (512, 512),
        padding_ratio: float = 0.2,
    ) -> np.ndarray | None:
        """Crop and resize face from image.

        Args:
            image: RGB image as numpy array (H, W, 3).
            target_size: Size to resize the cropped face to (W, H).
            padding_ratio: Extra padding around the face box.

        Returns:
            Cropped and resized face image, or None if no face detected.
        """
        from PIL import Image

        bbox = self.get_face_bounding_box(image, padding_ratio)
        if bbox is None:
            return None

        left, top, right, bottom = bbox
        face_patch = image[top:bottom, left:right]

        # Resize to target size
        face_pil = Image.fromarray(face_patch)
        face_pil = face_pil.resize(target_size, Image.BILINEAR)

        return np.array(face_pil)

    def crop_face_from_pil(self, pil_image, target_size: Tuple[int, int] = (512, 512)):
        """Crop face from a PIL image.

        Args:
            pil_image: PIL Image in RGB mode.
            target_size: Size to resize the cropped face to (W, H).

        Returns:
            PIL Image of cropped face, or None if no face detected.
        """
        from PIL import Image

        image_np = np.array(pil_image)
        face_np = self.crop_face(image_np, target_size)

        if face_np is None:
            return None

        return Image.fromarray(face_np)

    def crop_face_tensor(
        self,
        image_tensor: torch.Tensor,
        boxes: Tuple[int, int, int, int],
        target_size: Tuple[int, int] = (224, 224),
    ) -> torch.Tensor:
        """Crop face region from a tensor using precomputed box.

        Args:
            image_tensor: Image tensor of shape (C, H, W) or (B, C, H, W).
            boxes: Bounding box as (left, top, right, bottom).
            target_size: Output size (H, W).

        Returns:
            Cropped and resized face tensor.
        """
        import torch.nn.functional as F

        left, top, right, bottom = map(int, boxes)

        if image_tensor.dim() == 3:
            # (C, H, W) -> crop
            face_patch = image_tensor[:, top:bottom, left:right]
            face_patch = face_patch.unsqueeze(0)  # Add batch dim for interpolate
        else:
            # (B, C, H, W)
            face_patch = image_tensor[:, :, top:bottom, left:right]

        # Resize
        face_patch = F.interpolate(
            face_patch,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        return face_patch

    def close(self):
        """Release resources."""
        if hasattr(self, 'face_mesh') and self.face_mesh:
            self.face_mesh.close()

    def __del__(self):
        self.close()
