"""RIFE (Real-Time Intermediate Flow Estimation) HDv3 frame interpolation module.

This module provides frame interpolation functionality using RIFE HDv3 to double
the frame rate of video output from the pipeline.

Modified from https://github.com/hzwer/Practical-RIFE
The original repo is: https://github.com/hzwer/Practical-RIFE
"""

import logging
from pathlib import Path

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Try to import RIFE model
RIFE_AVAILABLE = False
RIFE_MODEL_CLASS = None

# Import RIFE HDv3 model from our codebase
try:
    from .RIFE_HDv3 import Model as RIFEModel

    RIFE_MODEL_CLASS = RIFEModel
    RIFE_AVAILABLE = True
    logger.info("RIFE HDv3 model found and imported")
except ImportError as e:
    RIFE_AVAILABLE = False
    logger.debug(f"RIFE HDv3 import failed: {e}")


class RIFEInterpolator:
    """RIFE HDv3-based frame interpolator.

    This class handles frame interpolation using RIFE HDv3 to generate intermediate
    frames between consecutive frames, effectively doubling the frame rate.

    Attributes:
        enabled: Whether interpolation is enabled
        device: Device to run interpolation on
        model: RIFE HDv3 model instance (if available)
        model_path: Path to RIFE HDv3 model weights directory
    """

    def __init__(
        self,
        enabled: bool = False,
        device: torch.device | None = None,
        model_path: str | None = None,
    ):
        """Initialize RIFE interpolator.

        Args:
            enabled: Whether interpolation is enabled
            device: Device to run interpolation on (defaults to CUDA if available, else CPU)
            model_path: Optional path to RIFE model weights file
        """
        self.enabled = enabled
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = None
        self.model_path = model_path

        if enabled:
            if not RIFE_AVAILABLE:
                raise ImportError(
                    "RIFE interpolation requested but RIFE HDv3 is not available. "
                    "Please install RIFE HDv3 from https://github.com/hzwer/arXiv2020-RIFE. "
                    "See docs/rife.md for installation instructions."
                )

            try:
                self._load_model()
                if self.model is None:
                    raise RuntimeError(
                        "RIFE HDv3 model weights not found. "
                        "Please download RIFE HDv3 model weights and place flownet.pkl in "
                        "arXiv2020-RIFE/train_log/ directory. "
                        "See docs/rife.md for installation instructions."
                    )
                logger.info("RIFE HDv3 model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load RIFE model: {e}")
                raise RuntimeError(
                    f"Failed to load RIFE model: {e}. "
                    "Please ensure RIFE is properly installed. "
                    "See docs/rife.md for installation instructions."
                ) from e

    def _load_model(self):
        """Load RIFE model."""
        if not RIFE_AVAILABLE or RIFE_MODEL_CLASS is None:
            return

        try:
            # Initialize RIFE model
            self.model = RIFE_MODEL_CLASS()

            # Find model weights directory (RIFE expects a directory containing flownet.pkl)
            model_dir_path = None
            if self.model_path:
                # If path is a file, get its parent directory
                model_path_obj = Path(self.model_path)
                if model_path_obj.is_file():
                    model_dir_path = str(model_path_obj.parent)
                elif model_path_obj.is_dir():
                    model_dir_path = str(model_path_obj)
            else:
                # Try default model directories
                from scope.server.models_config import get_models_dir

                # Get project root directory (where pyproject.toml is located)
                # Start from this file's directory and walk up to find project root
                current_file = Path(__file__).resolve()
                project_root = None
                for parent in current_file.parents:
                    if (parent / "pyproject.toml").exists():
                        project_root = parent
                        break

                default_model_dirs = []
                # First priority: project root weights folder
                if project_root:
                    weights_dir = project_root / "weights" / "RIFE"
                    default_model_dirs.append(weights_dir)

                # Second priority: configured models directory
                default_model_dirs.append(get_models_dir() / "RIFE")

                # Third priority: default home directory
                default_model_dirs.append(
                    Path.home() / ".daydream-scope" / "models" / "RIFE"
                )

                for model_dir in default_model_dirs:
                    if model_dir.exists() and (model_dir / "flownet.pkl").exists():
                        model_dir_path = str(model_dir)
                        break

            if model_dir_path:
                # Load model (RIFE uses -1 for latest version)
                # RIFE's load_model expects a directory path, not a file path
                self.model.load_model(model_dir_path, -1)
                self.model.eval()
                self.model.device()  # Move model to device
                logger.info(
                    f"Loaded RIFE HDv3 weights from {model_dir_path}/flownet.pkl"
                )

                # Compile model for faster inference (PyTorch 2.0+)
                # Note: torch.compile() is disabled due to CUDA graph issues with the mutable
                # backwarp_tenGrid cache in warplayer.py. The cache is mutated during inference,
                # which conflicts with CUDA graph capture. To enable compilation, the warplayer
                # module would need to be refactored to avoid mutable module-level state.
                # if hasattr(torch, "compile"):
                #     try:
                #         self.model.flownet = torch.compile(
                #             self.model.flownet,
                #             mode="default",  # Avoid CUDA graphs due to mutable cache in warplayer
                #             fullgraph=False,  # Allow graph breaks for compatibility
                #         )
                #         logger.info("RIFE flownet compiled with torch.compile()")
                #     except Exception as e:
                #         logger.warning(f"torch.compile() failed, using eager mode: {e}")
            else:
                raise FileNotFoundError(
                    "RIFE HDv3 model weights (flownet.pkl) not found. "
                    "Please download RIFE HDv3 model from https://github.com/hzwer/arXiv2020-RIFE "
                    "and place flownet.pkl in ~/.daydream-scope/models/RIFE/ directory. "
                    "See docs/rife.md for installation instructions."
                )
        except FileNotFoundError:
            # Re-raise FileNotFoundError for model weights
            raise
        except Exception as e:
            logger.error(f"Error loading RIFE HDv3 model: {e}", exc_info=True)
            raise RuntimeError(
                f"Failed to load RIFE HDv3 model: {e}. "
                "See docs/rife.md for installation instructions."
            ) from e

    def interpolate(self, frames: torch.Tensor) -> torch.Tensor:
        """Interpolate frames to double the frame rate.

        Args:
            frames: Input frames tensor of shape [T, H, W, C] with values in [0, 255] (uint8)

        Returns:
            Interpolated frames tensor of shape [T*2-1, H, W, C] with values in [0, 255] (uint8)

        Raises:
            RuntimeError: If RIFE is enabled but model is not available or loaded
        """
        if not self.enabled:
            return frames

        if frames.shape[0] < 2:
            # Can't interpolate with less than 2 frames
            return frames

        if not RIFE_AVAILABLE or self.model is None:
            raise RuntimeError(
                "RIFE interpolation is enabled but RIFE HDv3 model is not available. "
                "Please ensure RIFE HDv3 is properly installed and model weights are loaded. "
                "See docs/rife.md for installation instructions."
            )

        # Convert to float32 for processing
        frames_float = frames.float()

        # Use RIFE for interpolation
        return self._rife_interpolate(frames_float)

    def _rife_interpolate(self, frames: torch.Tensor) -> torch.Tensor:
        """Use RIFE model for interpolation.

        Optimized implementation with:
        - BF16 mixed precision for faster tensor core inference
        - Batched processing of all frame pairs in single forward pass
        - Pre-computed padding applied once to all frames
        - Minimal CPU-GPU transfers (single transfer at end)

        Args:
            frames: Input frames tensor of shape [T, H, W, C] with values in [0, 255]

        Returns:
            Interpolated frames tensor of shape [T*2-1, H, W, C] with values in [0, 255] (uint8)
        """
        num_frames = frames.shape[0]
        if num_frames < 2:
            return frames.clamp(0, 255).to(torch.uint8)

        T, H, W, C = frames.shape

        # Convert from [T, H, W, C] to [T, C, H, W] and normalize to [0, 1]
        # Move to GPU once at the start
        frames_chw = (frames.permute(0, 3, 1, 2) / 255.0).to(self.device).contiguous()

        # Calculate padding - RIFE v4.25 requires dimensions to be multiples of 32
        # For v4.25, we need to ensure minimum padding to handle all scales properly
        # Pad height to at least 512 if input is close (for v4.25 compatibility)
        tmp = 32
        ph = ((H - 1) // tmp + 1) * tmp
        pw = ((W - 1) // tmp + 1) * tmp
        padding = (0, pw - W, 0, ph - H)

        # Pre-compute padding for all frames at once (optimization)
        frames_padded = F.pad(frames_chw, padding)  # [T, C, pH, pW]

        with torch.no_grad():
            # Use BF16 mixed precision for faster inference on modern GPUs
            autocast_dtype = (
                torch.bfloat16 if self.device.type == "cuda" else torch.float32
            )
            with torch.amp.autocast(device_type=self.device.type, dtype=autocast_dtype):
                # Batch all frame pairs: frames[0:T-1] and frames[1:T]
                frames1_padded = frames_padded[:-1]  # [T-1, C, pH, pW]
                frames2_padded = frames_padded[1:]  # [T-1, C, pH, pW]

                # Batched inference - process all pairs at once
                mid_frames_padded = self.model.inference(
                    frames1_padded, frames2_padded, scale=1.0
                )  # [T-1, C, pH, pW]

            # Remove padding from interpolated frames (stay on GPU)
            mid_frames = mid_frames_padded[:, :, :H, :W].float()  # [T-1, C, H, W]

            # Get original frames without padding (stay on GPU)
            original_frames = frames_padded[:, :, :H, :W]  # [T, C, H, W]

            # Interleave original and interpolated frames on GPU
            # Result: [orig[0], mid[0], orig[1], mid[1], ..., orig[T-2], mid[T-2], orig[T-1]]
            result_frames = torch.zeros(
                (T * 2 - 1, C, H, W), dtype=torch.float32, device=self.device
            )
            result_frames[0::2] = original_frames  # Original frames at even indices
            result_frames[1::2] = mid_frames  # Interpolated frames at odd indices

            # Scale to [0, 255] and transfer to CPU once at the end
            result_chw = (result_frames * 255.0).cpu()

        # Convert back to [T*2-1, H, W, C]
        result = result_chw.permute(0, 2, 3, 1).contiguous()

        # Clamp to valid range and convert to uint8
        result = result.clamp(0.0, 255.0).to(torch.uint8)

        return result

    def set_enabled(self, enabled: bool):
        """Enable or disable interpolation.

        Args:
            enabled: Whether to enable interpolation

        Raises:
            RuntimeError: If enabling RIFE but model is not available
        """
        if enabled:
            if not RIFE_AVAILABLE:
                raise RuntimeError(
                    "RIFE interpolation cannot be enabled: RIFE HDv3 is not available. "
                    "Please install RIFE HDv3 from https://github.com/hzwer/arXiv2020-RIFE. "
                    "See docs/rife.md for installation instructions."
                )

            if self.model is None:
                # Try to load model if not already loaded
                try:
                    self._load_model()
                    if self.model is None:
                        raise RuntimeError(
                            "RIFE HDv3 model weights not found. "
                            "Please download RIFE HDv3 model weights. "
                            "See docs/rife.md for installation instructions."
                        )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load RIFE HDv3 model when enabling: {e}. "
                        "See docs/rife.md for installation instructions."
                    ) from e

        self.enabled = enabled


def is_rife_available() -> bool:
    """Check if RIFE is available for use.

    Returns:
        True if RIFE is available, False otherwise
    """
    return RIFE_AVAILABLE


def get_rife_model_path() -> Path | None:
    """Get the default RIFE HDv3 model directory path.

    Returns:
        Path to RIFE HDv3 model directory (containing flownet.pkl) if found, None otherwise
    """
    from scope.server.models_config import get_models_dir

    # Get project root directory
    current_file = Path(__file__).resolve()
    project_root = None
    for parent in current_file.parents:
        if (parent / "pyproject.toml").exists():
            project_root = parent
            break

    default_dirs = []
    # First priority: project root weights folder
    if project_root:
        weights_dir = project_root / "weights" / "RIFE"
        default_dirs.append(weights_dir)

    # Second priority: configured models directory
    default_dirs.append(get_models_dir() / "RIFE")

    # Third priority: default home directory
    default_dirs.append(Path.home() / ".daydream-scope" / "models" / "RIFE")

    for model_dir in default_dirs:
        if model_dir.exists() and (model_dir / "flownet.pkl").exists():
            return model_dir
    return None
