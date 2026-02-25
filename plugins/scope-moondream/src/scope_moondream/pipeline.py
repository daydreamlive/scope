"""Moondream vision language model pipeline for Daydream Scope."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from scope.core.pipelines.interface import Pipeline, Requirements

from .overlay import (
    draw_bounding_boxes,
    draw_points,
    draw_text_overlay,
    pil_to_tensor,
    tensor_to_pil,
)
from .schema import MoondreamConfig, MoondreamFeature

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

logger = logging.getLogger(__name__)


class MoondreamPipeline(Pipeline):
    """Real-time vision analysis using Moondream2.

    Processes video frames through Moondream's vision features (caption, query,
    detect, point) and overlays results on the output frames.
    """

    @classmethod
    def get_config_class(cls) -> type[BasePipelineConfig]:
        return MoondreamConfig

    def __init__(self, **kwargs: Any) -> None:
        """Load Moondream2 model.

        The pipeline manager passes all schema defaults merged with load_params
        as keyword arguments. Device is not provided -- we determine it ourselves.
        """
        from transformers import AutoModelForCausalLM

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_map = "cuda" if self.device.type == "cuda" else "cpu"

        logger.info(f"Loading Moondream2 model on {device_map}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        self.model.eval()

        if kwargs.get("compile_model", False):
            logger.info("Compiling Moondream model (this may take a moment)...")
            self.model.compile()

        logger.info("Moondream2 model loaded successfully")

        # Frame counter and result cache for inference_interval skipping
        self._frame_count = 0
        self._cached_result: dict[str, Any] | None = None
        self._cached_feature: str | None = None

    def prepare(self, **kwargs: Any) -> Requirements:
        """Moondream processes one frame at a time."""
        return Requirements(input_size=1)

    def __call__(self, **kwargs: Any) -> dict:
        """Process a video frame through Moondream and overlay results.

        Args:
            **kwargs: Runtime parameters including:
                video: List of tensors, each (1, H, W, C) in [0, 255]
                feature: MoondreamFeature enum value
                question: str for query mode
                detect_object: str for detect/point modes
                caption_length: CaptionLength enum value
                temperature: float
                max_objects: int
                inference_interval: int
                overlay_opacity: float
                font_scale: float

        Returns:
            {"video": tensor} in THWC format, [0, 1] range
        """
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Input video cannot be None")

        # Extract runtime parameters
        feature = kwargs.get("feature", MoondreamFeature.DETECT)
        if isinstance(feature, str):
            feature = MoondreamFeature(feature)

        question = kwargs.get("question", "What is in this image?")
        detect_object = kwargs.get("detect_object", "person")
        caption_length = kwargs.get("caption_length", "normal")
        if hasattr(caption_length, "value"):
            caption_length = caption_length.value
        temperature = kwargs.get("temperature", 0.5)
        max_objects = kwargs.get("max_objects", 10)
        inference_interval = kwargs.get("inference_interval", 1)
        overlay_opacity = kwargs.get("overlay_opacity", 0.8)
        font_scale = kwargs.get("font_scale", 1.0)

        # Get the input frame
        frame_tensor = video[0]  # (1, H, W, C) in [0, 255]
        pil_image = tensor_to_pil(frame_tensor)

        # Decide whether to run inference this frame
        should_infer = (
            self._frame_count % max(1, inference_interval) == 0
            or self._cached_result is None
            or self._cached_feature != feature.value
        )

        if should_infer:
            self._cached_result = self._run_inference(
                pil_image,
                feature=feature,
                question=question,
                detect_object=detect_object,
                caption_length=caption_length,
                temperature=temperature,
                max_objects=max_objects,
            )
            self._cached_feature = feature.value

        self._frame_count += 1

        # Overlay results on the frame
        annotated = self._draw_overlay(
            pil_image,
            feature=feature,
            result=self._cached_result,
            opacity=overlay_opacity,
            font_scale=font_scale,
        )

        # Convert back to tensor
        output = pil_to_tensor(annotated, self.device)  # (1, H, W, C) in [0, 1]
        return {"video": output}

    @torch.no_grad()
    def _run_inference(
        self,
        pil_image,
        *,
        feature: MoondreamFeature,
        question: str,
        detect_object: str,
        caption_length: str,
        temperature: float,
        max_objects: int,
    ) -> dict[str, Any]:
        """Run Moondream inference on a single image."""
        settings = {"temperature": temperature}

        if feature == MoondreamFeature.CAPTION:
            result = self.model.caption(
                pil_image, length=caption_length, settings=settings
            )
            return {"type": "caption", "text": result.get("caption", "")}

        elif feature == MoondreamFeature.QUERY:
            result = self.model.query(
                pil_image, question=question, settings=settings
            )
            return {
                "type": "query",
                "question": question,
                "answer": result.get("answer", ""),
            }

        elif feature == MoondreamFeature.DETECT:
            result = self.model.detect(
                pil_image,
                object=detect_object,
                settings={"max_objects": max_objects},
            )
            return {
                "type": "detect",
                "label": detect_object,
                "objects": result.get("objects", []),
            }

        elif feature == MoondreamFeature.POINT:
            result = self.model.point(
                pil_image,
                object=detect_object,
                settings={"max_objects": max_objects},
            )
            return {
                "type": "point",
                "label": detect_object,
                "points": result.get("points", []),
            }

        return {"type": "unknown"}

    def _draw_overlay(
        self,
        image,
        *,
        feature: MoondreamFeature,
        result: dict[str, Any],
        opacity: float,
        font_scale: float,
    ):
        """Draw inference results on the image."""
        if result is None:
            return image

        result_type = result.get("type")

        if result_type == "caption":
            return draw_text_overlay(
                image, result["text"], opacity=opacity, font_scale=font_scale
            )

        elif result_type == "query":
            text = f"Q: {result['question']}\nA: {result['answer']}"
            return draw_text_overlay(
                image, text, opacity=opacity, font_scale=font_scale
            )

        elif result_type == "detect":
            return draw_bounding_boxes(
                image,
                result.get("objects", []),
                opacity=opacity,
                font_scale=font_scale,
            )

        elif result_type == "point":
            return draw_points(
                image,
                result.get("points", []),
                opacity=opacity,
                font_scale=font_scale,
            )

        return image
