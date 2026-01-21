"""SAM3 Pipeline for real-time object masking and tracking.

Uses Meta's Segment Anything Model 3 (SAM3) for open-vocabulary segmentation
and video tracking. Supports text prompts to specify what objects to segment.
"""

import logging
import time
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from ..interface import Pipeline, Requirements
from .schema import SAM3Config

if TYPE_CHECKING:
    from ..base_schema import BasePipelineConfig

logger = logging.getLogger(__name__)


def _is_sam3_available() -> bool:
    """Check if sam3 package is available."""
    try:
        import sam3  # noqa: F401

        return True
    except ImportError:
        return False


class SAM3Pipeline(Pipeline):
    """SAM3 pipeline for real-time object masking and tracking.

    This pipeline uses Meta's Segment Anything Model 3 (SAM3) for
    open-vocabulary segmentation. Unlike SAM1/SAM2 which required
    visual prompts (clicks/boxes), SAM3 can segment objects based
    on text descriptions.

    The model is lazily loaded on first use to minimize startup time.
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return SAM3Config

    def __init__(
        self,
        config: SAM3Config,
        device: torch.device | None = None,
    ):
        """Initialize the SAM3 pipeline.

        Args:
            config: Pipeline configuration
            device: Target device (defaults to CUDA if available)
        """
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Configuration from schema
        self._segment_prompt = getattr(config, "segment_prompt", "person")
        self._output_mode = getattr(config, "output_mode", "mask")
        self._mask_color = getattr(config, "mask_color", (0, 255, 0))
        self._mask_opacity = getattr(config, "mask_opacity", 0.5)
        self._confidence_threshold = getattr(config, "confidence_threshold", 0.5)
        self._enable_tracking = getattr(config, "enable_tracking", True)

        # Lazy-loaded model components
        self._image_model = None
        self._image_processor = None
        self._video_predictor = None
        self._video_session_id = None

        # State tracking
        self._is_sam3_available = _is_sam3_available()
        self._current_prompt = self._segment_prompt
        self._frame_index = 0

        start = time.time()
        if self._is_sam3_available:
            logger.info(
                f"SAM3 pipeline initialized (model loads on first use). "
                f"Segment prompt: '{self._segment_prompt}'"
            )
        else:
            logger.warning(
                "SAM3 package not available. Install with: pip install sam3 "
                "(requires access to facebook/sam3 on HuggingFace)"
            )
        logger.info(f"Initialization time: {time.time() - start:.3f}s")

    def _ensure_image_model(self):
        """Lazily initialize the SAM3 image model.

        Returns:
            Tuple of (model, processor)
        """
        if self._image_model is not None:
            return self._image_model, self._image_processor

        if not self._is_sam3_available:
            raise RuntimeError(
                "SAM3 package not available. Install with: pip install sam3"
            )

        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        logger.info("Loading SAM3 image model...")
        start = time.time()

        self._image_model = build_sam3_image_model()
        self._image_model = self._image_model.to(self.device)
        self._image_model.eval()
        self._image_processor = Sam3Processor(self._image_model)

        logger.info(f"SAM3 image model loaded in {time.time() - start:.3f}s")
        return self._image_model, self._image_processor

    def _ensure_video_predictor(self):
        """Lazily initialize the SAM3 video predictor.

        Returns:
            Video predictor instance
        """
        if self._video_predictor is not None:
            return self._video_predictor

        if not self._is_sam3_available:
            raise RuntimeError(
                "SAM3 package not available. Install with: pip install sam3"
            )

        from sam3.model_builder import build_sam3_video_predictor

        logger.info("Loading SAM3 video predictor...")
        start = time.time()

        self._video_predictor = build_sam3_video_predictor()

        logger.info(f"SAM3 video predictor loaded in {time.time() - start:.3f}s")
        return self._video_predictor

    def _segment_frame_image_mode(
        self, frame: torch.Tensor, prompt: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Segment a single frame using image mode.

        Args:
            frame: Input frame tensor (HWC format, [0,1] range)
            prompt: Text prompt for segmentation

        Returns:
            Tuple of (masks, boxes, scores) tensors
        """
        import numpy as np
        from PIL import Image

        _, processor = self._ensure_image_model()

        # Convert tensor to PIL Image
        frame_np = (frame.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(frame_np)

        # Run segmentation
        inference_state = processor.set_image(pil_image)
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)

        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]

        return masks, boxes, scores

    def _apply_output_mode(
        self,
        frame: torch.Tensor,
        masks: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the configured output mode to produce final output.

        Args:
            frame: Original input frame (HWC format, [0,1] range)
            masks: Segmentation masks from SAM3
            scores: Confidence scores for each mask

        Returns:
            Output frame tensor (HWC format, [0,1] range)
        """
        h, w = frame.shape[:2]

        # Filter masks by confidence threshold
        if scores is not None and len(scores) > 0:
            valid_indices = scores >= self._confidence_threshold
            if isinstance(valid_indices, torch.Tensor):
                valid_indices = valid_indices.cpu()
            masks = masks[valid_indices] if valid_indices.any() else masks[:0]

        # Combine all valid masks into a single mask
        if masks is not None and len(masks) > 0:
            # Convert masks to tensor if needed
            if not isinstance(masks, torch.Tensor):
                masks = torch.from_numpy(masks)
            masks = masks.to(self.device)

            # Combine masks (union of all detected instances)
            combined_mask = masks.any(dim=0).float()

            # Resize mask to frame size if needed
            if combined_mask.shape[-2:] != (h, w):
                combined_mask = F.interpolate(
                    combined_mask.unsqueeze(0).unsqueeze(0),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
        else:
            # No valid masks - create empty mask
            combined_mask = torch.zeros(h, w, device=self.device)

        # Ensure frame is on device
        frame = frame.to(self.device)

        if self._output_mode == "mask":
            # Return binary mask as grayscale image (3 channels for consistency)
            mask_3ch = combined_mask.unsqueeze(-1).expand(-1, -1, 3)
            return mask_3ch

        elif self._output_mode == "overlay":
            # Overlay colored mask on original frame
            mask_color = (
                torch.tensor(self._mask_color, dtype=torch.float32, device=self.device)
                / 255.0
            )
            mask_expanded = combined_mask.unsqueeze(-1)

            # Blend: output = frame * (1 - mask*opacity) + color * mask * opacity
            overlay = frame * (1 - mask_expanded * self._mask_opacity)
            overlay = overlay + mask_color * mask_expanded * self._mask_opacity
            return overlay.clamp(0, 1)

        elif self._output_mode == "cutout":
            # Show only the segmented object, black background
            mask_expanded = combined_mask.unsqueeze(-1)
            return frame * mask_expanded

        else:
            # Fallback to mask mode
            mask_3ch = combined_mask.unsqueeze(-1).expand(-1, -1, 3)
            return mask_3ch

    def reset(self):
        """Reset the pipeline state.

        Call this between different video sequences to clear tracking state.
        """
        self._frame_index = 0
        self._video_session_id = None

    def prepare(self, **kwargs) -> Requirements:
        """Return pipeline requirements.

        Returns:
            Requirements specifying minimum input_size
        """
        return Requirements(input_size=1)

    def update_prompt(self, prompt: str):
        """Update the segmentation prompt.

        Args:
            prompt: New text prompt for segmentation
        """
        if prompt != self._current_prompt:
            self._current_prompt = prompt
            logger.info(f"Updated segment prompt to: '{prompt}'")
            # Reset tracking when prompt changes
            self.reset()

    def _normalize_frame(
        self, frame, target_shape: tuple | None
    ) -> tuple[torch.Tensor, tuple]:
        """Normalize a single frame to consistent format.

        Args:
            frame: Input frame (tensor or numpy array)
            target_shape: Target shape for consistency, or None to use this frame's shape

        Returns:
            Tuple of (normalized frame tensor in HWC [0,1], target_shape)
        """
        if isinstance(frame, torch.Tensor):
            frame_tensor = frame
        else:
            frame_tensor = torch.from_numpy(frame)

        # Squeeze T dimension: (1, H, W, C) -> (H, W, C)
        frame_tensor = frame_tensor.squeeze(0)

        # Use first frame's shape as target for consistency
        if target_shape is None:
            target_shape = frame_tensor.shape
        elif frame_tensor.shape != target_shape:
            # Resize frame to match target shape
            frame_chw = frame_tensor.permute(2, 0, 1).unsqueeze(0).float()
            frame_chw = F.interpolate(
                frame_chw,
                size=(target_shape[0], target_shape[1]),
                mode="bilinear",
                align_corners=False,
            )
            frame_tensor = frame_chw.squeeze(0).permute(1, 2, 0)

        # Normalize to [0, 1] if needed
        if frame_tensor.max() > 1.0:
            frame_tensor = frame_tensor.float() / 255.0
        else:
            frame_tensor = frame_tensor.float()

        return frame_tensor, target_shape

    def __call__(self, **kwargs) -> torch.Tensor:
        """Process video frames and return segmentation masks/overlays.

        Args:
            video: Input video frames as list of tensors (THWC format, [0, 255] range)
            segment_prompt: Optional text prompt override for this call

        Returns:
            Output frames as tensor in THWC format with values in [0, 1] range.
            Format depends on output_mode configuration.
        """
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Input video cannot be None for SAM3Pipeline")

        # Allow runtime prompt override
        prompt = kwargs.get("segment_prompt", self._current_prompt)
        if prompt != self._current_prompt:
            self.update_prompt(prompt)

        # Normalize all frames
        frames = []
        target_shape = None
        for frame in video:
            frame_tensor, target_shape = self._normalize_frame(frame, target_shape)
            frames.append(frame_tensor)

        num_frames = len(frames)
        output_frames = []

        # Process each frame
        for i in range(num_frames):
            frame = frames[i]

            try:
                # Segment the frame using image mode
                # (Video mode with tracking can be added later for improved temporal consistency)
                masks, boxes, scores = self._segment_frame_image_mode(frame, prompt)

                # Apply output mode (mask, overlay, or cutout)
                output = self._apply_output_mode(frame, masks, scores)
                output_frames.append(output)

            except Exception as e:
                logger.error(f"SAM3 segmentation failed on frame {i}: {e}")
                # Fallback: return original frame or empty mask
                if self._output_mode == "mask":
                    h, w = frame.shape[:2]
                    output_frames.append(torch.zeros(h, w, 3, device=self.device))
                else:
                    output_frames.append(frame.to(self.device))

            self._frame_index += 1

        # Stack output frames: [T, H, W, C]
        output_tensor = torch.stack(output_frames, dim=0)

        return output_tensor
