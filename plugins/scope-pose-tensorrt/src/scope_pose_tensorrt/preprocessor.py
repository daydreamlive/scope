"""Main pose TensorRT preprocessor for VACE conditioning."""

import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .download import (
    DEFAULT_ONNX_FILENAME,
    DEFAULT_REPO_ID,
    download_onnx_model,
    get_engine_path,
    get_models_dir,
)
from .engine import TensorRTEngine, compile_onnx_to_trt, get_gpu_name
from .visualization import render_pose_image

logger = logging.getLogger(__name__)


class PoseTensorRTPreprocessor:
    """TensorRT-accelerated pose estimation preprocessor for VACE conditioning.

    This preprocessor detects human poses in input frames and renders them
    as skeleton visualizations on a black background, suitable for use as
    VACE/ControlNet conditioning signals.

    The TensorRT engine is lazily initialized on first use, automatically
    downloading the ONNX model and compiling it to a GPU-specific TRT engine.

    Example:
        ```python
        preprocessor = PoseTensorRTPreprocessor()

        # Process video frames for VACE
        # Input: [B, C, F, H, W] in [0, 1] range
        pose_maps = preprocessor(input_frames)
        # Output: [1, C, F, H, W] in [-1, 1] range

        # Use with pipeline
        pipeline(vace_input_frames=pose_maps, ...)
        ```
    """

    def __init__(
        self,
        detect_resolution: int = 640,
        device: str = "cuda",
        fp16: bool = True,
        joint_thickness: int = 10,
        keypoint_radius: int = 10,
        repo_id: str = DEFAULT_REPO_ID,
        onnx_filename: str = DEFAULT_ONNX_FILENAME,
    ):
        """Initialize the pose preprocessor.

        Args:
            detect_resolution: Resolution for pose detection (should match ONNX model input)
            device: Target device for inference ("cuda" or "cpu")
            fp16: Whether to use FP16 precision for TRT compilation
            joint_thickness: Line thickness for skeleton edges in visualization
            keypoint_radius: Radius for keypoint circles in visualization
            repo_id: HuggingFace repository ID for ONNX model
            onnx_filename: ONNX model filename in the repository
        """
        self._detect_resolution = detect_resolution
        self._device = device
        self._fp16 = fp16
        self._joint_thickness = joint_thickness
        self._keypoint_radius = keypoint_radius
        self._repo_id = repo_id
        self._onnx_filename = onnx_filename

        self._engine: TensorRTEngine | None = None
        self._is_cuda_available = torch.cuda.is_available()

    def _ensure_engine(self) -> TensorRTEngine:
        """Lazily initialize the TensorRT engine.

        Downloads ONNX model and compiles to TRT on first call.

        Returns:
            Initialized TensorRTEngine
        """
        if self._engine is not None:
            return self._engine

        models_dir = get_models_dir()
        gpu_name = get_gpu_name()

        # Download ONNX model if needed
        onnx_path = download_onnx_model(
            repo_id=self._repo_id,
            filename=self._onnx_filename,
            models_dir=models_dir,
        )

        # Compile to TensorRT if needed (GPU-specific)
        engine_path = get_engine_path(onnx_path, gpu_name)
        compile_onnx_to_trt(onnx_path, engine_path, fp16=self._fp16)

        # Load and activate engine
        self._engine = TensorRTEngine(engine_path)
        self._engine.load()
        self._engine.activate()
        self._engine.allocate_buffers(self._device)

        logger.info("Pose TensorRT engine initialized successfully")
        return self._engine

    def _process_single_frame(self, frame: torch.Tensor) -> np.ndarray:
        """Process a single frame through pose detection.

        Args:
            frame: Input frame tensor [C, H, W] in [0, 1] range

        Returns:
            Pose visualization as numpy array [H, W, C] BGR uint8
        """
        engine = self._ensure_engine()

        # Resize to detection resolution
        frame = frame.unsqueeze(0)  # [1, C, H, W]
        frame_resized = F.interpolate(
            frame,
            size=(self._detect_resolution, self._detect_resolution),
            mode="bilinear",
            align_corners=False,
        )

        # Convert to uint8 for TRT input
        frame_uint8 = (frame_resized * 255.0).to(torch.uint8)

        if self._is_cuda_available and not frame_uint8.is_cuda:
            frame_uint8 = frame_uint8.cuda()

        # Run inference
        cuda_stream = torch.cuda.current_stream().cuda_stream
        result = engine.infer({engine.input_name: frame_uint8}, cuda_stream)

        # Extract output predictions only
        predictions = [result[key].cpu().numpy() for key in engine.output_names]

        # Render pose visualization
        pose_image = render_pose_image(
            predictions,
            detect_resolution=self._detect_resolution,
            joint_thickness=self._joint_thickness,
            keypoint_radius=self._keypoint_radius,
        )

        return pose_image

    def __call__(
        self,
        frames: torch.Tensor,
        target_height: int | None = None,
        target_width: int | None = None,
    ) -> torch.Tensor:
        """Process input frames to produce pose maps for VACE conditioning.

        Args:
            frames: Input video frames. Supported formats:
                - [B, C, F, H, W]: Batch of videos (B batches, F frames each)
                - [C, F, H, W]: Single video (F frames)
                - [F, H, W, C]: Single video in FHWC format (will be converted)
                All inputs should be in [0, 1] float range.
            target_height: Output height. If None, uses detect_resolution.
            target_width: Output width. If None, uses detect_resolution.

        Returns:
            Pose maps in VACE format: [1, C, F, H, W] float in [-1, 1] range,
            with pose skeletons rendered on black background.
        """
        # Handle different input formats
        if frames.dim() == 4:
            if frames.shape[-1] == 3:
                # [F, H, W, C] -> [C, F, H, W]
                frames = frames.permute(3, 0, 1, 2)
            # [C, F, H, W] -> [1, C, F, H, W]
            frames = frames.unsqueeze(0)

        batch_size, channels, num_frames, height, width = frames.shape

        # Default target size to detect resolution
        if target_height is None:
            target_height = self._detect_resolution
        if target_width is None:
            target_width = self._detect_resolution

        # Process each frame
        pose_frames = []
        for b in range(batch_size):
            batch_poses = []
            for f in range(num_frames):
                frame = frames[b, :, f]  # [C, H, W]
                pose_image = self._process_single_frame(frame)

                # Convert BGR to RGB
                pose_image = cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB)

                # Resize to target size if needed
                if (
                    pose_image.shape[0] != target_height
                    or pose_image.shape[1] != target_width
                ):
                    pose_image = cv2.resize(
                        pose_image,
                        (target_width, target_height),
                        interpolation=cv2.INTER_LINEAR,
                    )

                batch_poses.append(pose_image)

            pose_frames.append(batch_poses)

        # Convert to tensor [B, C, F, H, W] in [-1, 1] range
        pose_tensor = torch.from_numpy(np.array(pose_frames)).float()
        # [B, F, H, W, C] -> [B, C, F, H, W]
        pose_tensor = pose_tensor.permute(0, 4, 1, 2, 3)
        # Normalize to [-1, 1]
        pose_tensor = pose_tensor / 255.0 * 2.0 - 1.0

        if self._is_cuda_available:
            pose_tensor = pose_tensor.cuda()

        return pose_tensor

    def process_pil_image(self, image) -> torch.Tensor:
        """Process a single PIL image to pose map.

        Convenience method for processing individual images.

        Args:
            image: PIL Image in RGB format

        Returns:
            Pose map tensor [1, C, 1, H, W] in [-1, 1] range
        """
        import numpy as np

        # Convert PIL to tensor
        img_array = np.array(image)
        img_tensor = torch.from_numpy(img_array).float() / 255.0
        # [H, W, C] -> [C, H, W] -> [1, C, 1, H, W]
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)

        return self(img_tensor)

    def warmup(self):
        """Warm up the TensorRT engine.

        Call this before processing to avoid latency on first frame.
        """
        logger.info("Warming up pose TensorRT engine...")
        self._ensure_engine()

        # Run a dummy inference
        dummy_input = torch.zeros(
            1, 3, 1, self._detect_resolution, self._detect_resolution
        )
        if self._is_cuda_available:
            dummy_input = dummy_input.cuda()

        _ = self(dummy_input)
        logger.info("Warmup complete")
