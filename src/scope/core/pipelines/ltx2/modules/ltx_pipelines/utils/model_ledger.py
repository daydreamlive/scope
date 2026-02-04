import logging
from dataclasses import replace
from enum import Enum

import torch
from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.model.audio_vae import (
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoder,
    AudioDecoderConfigurator,
    Vocoder,
    VocoderConfigurator,
)
from ltx_core.model.transformer import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXV_MODEL_COMFY_RENAMING_WITH_TRANSFORMER_LINEAR_DOWNCAST_MAP,
    UPCAST_DURING_INFERENCE,
    LTXModelConfigurator,
    X0Model,
)
from ltx_core.model.upsampler import LatentUpsampler, LatentUpsamplerConfigurator
from ltx_core.model.video_vae import (
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
    VideoDecoder,
    VideoDecoderConfigurator,
    VideoEncoder,
    VideoEncoderConfigurator,
)
from ltx_core.text_encoders.gemma import (
    AV_GEMMA_TEXT_ENCODER_KEY_OPS,
    AVGemmaTextEncoderModel,
    AVGemmaTextEncoderModelConfigurator,
    module_ops_from_gemma_root,
)

# Use scope logger namespace to ensure logs appear in the main application log
logger = logging.getLogger("scope.core.pipelines.ltx2.model_ledger")


class QuantizationType(str, Enum):
    """Quantization type for transformer weights."""

    NONE = "none"
    FP8 = "fp8"
    NVFP4 = "nvfp4"


# Default chunk size for FFN chunking
# 4096 is a good balance between memory savings and kernel launch overhead
DEFAULT_FFN_CHUNK_SIZE = 4096


class ModelLedger:
    """
    Central coordinator for loading and building models used in an LTX pipeline.
    The ledger wires together multiple model builders (transformer, video VAE encoder/decoder,
    audio VAE decoder, vocoder, text encoder, and optional latent upsampler) and exposes
    factory methods for constructing model instances.
    ### Model Building
    Each model method (e.g. :meth:`transformer`, :meth:`video_decoder`, :meth:`text_encoder`)
    constructs a new model instance on each call. The builder uses the
    :class:`~ltx_core.loader.registry.Registry` to load weights from the checkpoint,
    instantiates the model with the configured ``dtype``, and moves it to ``self.device``.
    .. note::
        Models are **not cached**. Each call to a model method creates a new instance.
        Callers are responsible for storing references to models they wish to reuse
        and for freeing GPU memory (e.g. by deleting references and calling
        ``torch.cuda.empty_cache()``).
    ### Constructor parameters
    dtype:
        Torch dtype used when constructing all models (e.g. ``torch.bfloat16``).
    device:
        Target device to which models are moved after construction (e.g. ``torch.device("cuda")``).
    checkpoint_path:
        Path to a checkpoint directory or file containing the core model weights
        (transformer, video VAE, audio VAE, text encoder, vocoder). If ``None``, the
        corresponding builders are not created and calling those methods will raise
        a :class:`ValueError`.
    gemma_root_path:
        Base path to Gemma-compatible CLIP/text encoder weights. Required to
        initialize the text encoder builder; if omitted, :meth:`text_encoder` cannot be used.
    spatial_upsampler_path:
        Optional path to a latent upsampler checkpoint. If provided, the
        :meth:`spatial_upsampler` method becomes available; otherwise calling it raises
        a :class:`ValueError`.
    loras:
        Optional collection of LoRA configurations (paths, strengths, and key operations)
        that are applied on top of the base transformer weights when building the model.
    registry:
        Optional :class:`Registry` instance for weight caching across builders.
        Defaults to :class:`DummyRegistry` which performs no cross-builder caching.
    fp8transformer:
        If ``True``, builds the transformer with FP8 quantization and upcasting during inference.
    ### Creating Variants
    Use :meth:`with_loras` to create a new ``ModelLedger`` instance that includes
    additional LoRA configurations while sharing the same registry for weight caching.
    """

    def __init__(
        self,
        dtype: torch.dtype,
        device: torch.device,
        checkpoint_path: str | None = None,
        gemma_root_path: str | None = None,
        spatial_upsampler_path: str | None = None,
        loras: LoraPathStrengthAndSDOps | None = None,
        registry: Registry | None = None,
        fp8transformer: bool = False,
        quantization: str | None = None,
        ffn_chunk_size: int | None = DEFAULT_FFN_CHUNK_SIZE,
    ):
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.gemma_root_path = gemma_root_path
        self.spatial_upsampler_path = spatial_upsampler_path
        self.loras = loras or ()
        self.registry = registry or DummyRegistry()
        self.ffn_chunk_size = ffn_chunk_size

        # Resolve quantization type from new parameter or legacy fp8transformer flag
        if quantization is not None:
            self.quantization = QuantizationType(quantization)
            logger.info(
                f"ModelLedger quantization set to: {self.quantization} (from '{quantization}')"
            )
        elif fp8transformer:
            self.quantization = QuantizationType.FP8
            logger.info(
                "ModelLedger quantization set to: FP8 (from fp8transformer=True)"
            )
        else:
            self.quantization = QuantizationType.NONE
            logger.info("ModelLedger quantization set to: NONE")

        # Keep fp8transformer for backwards compatibility
        self.fp8transformer = fp8transformer or (
            self.quantization == QuantizationType.FP8
        )

        self.build_model_builders()

    def build_model_builders(self) -> None:
        if self.checkpoint_path is not None:
            self.transformer_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=LTXModelConfigurator,
                model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
                loras=tuple(self.loras),
                registry=self.registry,
            )

            self.vae_decoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=VideoDecoderConfigurator,
                model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )

            self.vae_encoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=VideoEncoderConfigurator,
                model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )

            self.audio_decoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=AudioDecoderConfigurator,
                model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )

            self.vocoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=VocoderConfigurator,
                model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )

            if self.gemma_root_path is not None:
                self.text_encoder_builder = Builder(
                    model_path=self.checkpoint_path,
                    model_class_configurator=AVGemmaTextEncoderModelConfigurator,
                    model_sd_ops=AV_GEMMA_TEXT_ENCODER_KEY_OPS,
                    registry=self.registry,
                    module_ops=module_ops_from_gemma_root(self.gemma_root_path),
                )

        if self.spatial_upsampler_path is not None:
            self.upsampler_builder = Builder(
                model_path=self.spatial_upsampler_path,
                model_class_configurator=LatentUpsamplerConfigurator,
                registry=self.registry,
            )

    def _target_device(self) -> torch.device:
        if isinstance(self.registry, DummyRegistry) or self.registry is None:
            return self.device
        else:
            return torch.device("cpu")

    def with_loras(self, loras: LoraPathStrengthAndSDOps) -> "ModelLedger":
        return ModelLedger(
            dtype=self.dtype,
            device=self.device,
            checkpoint_path=self.checkpoint_path,
            gemma_root_path=self.gemma_root_path,
            spatial_upsampler_path=self.spatial_upsampler_path,
            loras=(*self.loras, *loras),
            registry=self.registry,
            quantization=self.quantization.value,
            ffn_chunk_size=self.ffn_chunk_size,
        )

    def _apply_ffn_chunking(self, model: X0Model) -> X0Model:
        """Apply FFN chunking to reduce activation memory during inference.

        FFN layers expand hidden dimensions by 4x, creating massive intermediate
        tensors. By processing the sequence in chunks, we reduce peak memory
        from O(seq_len * 4*dim) to O(chunk_size * 4*dim).

        For LTX-2 with 96 FFN layers (48 video + 48 audio), this can reduce
        activation memory from ~50GB to ~5GB at typical sequence lengths.

        Args:
            model: The X0Model containing the transformer

        Returns:
            The same model with FFN layers wrapped for chunked processing
        """
        if self.ffn_chunk_size is None:
            logger.info("FFN chunking disabled (ffn_chunk_size=None)")
            return model

        try:
            from ltx_core.model.transformer.chunked_ffn import (
                apply_chunked_ffn,
                estimate_ffn_memory_savings,
            )

            # Apply chunked FFN to the velocity model (the actual transformer)
            num_patched = apply_chunked_ffn(
                model.velocity_model,
                chunk_size=self.ffn_chunk_size,
                verbose=True,
            )

            if num_patched > 0:
                # Log estimated memory savings for typical sequence lengths
                # 720x1280 @ 33 frames ≈ 57000 tokens
                savings = estimate_ffn_memory_savings(
                    seq_len=57000,
                    dim=4096,
                    num_layers=num_patched,
                    chunk_size=self.ffn_chunk_size,
                )
                logger.info(
                    f"FFN chunking estimated savings: {savings['savings_gb']:.1f}GB "
                    f"(~{savings['reduction_factor']:.1f}x reduction)"
                )

        except Exception as e:
            logger.warning(f"Failed to apply FFN chunking: {e}")
            logger.warning("Continuing without FFN chunking")

        return model

    def transformer(self) -> X0Model:
        if not hasattr(self, "transformer_builder"):
            raise ValueError(
                "Transformer not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        logger.info(f"Building transformer with quantization: {self.quantization}")

        if self.quantization == QuantizationType.FP8:
            # FP8 quantization: downcast weights to FP8 and upcast during inference
            fp8_builder = replace(
                self.transformer_builder,
                module_ops=(UPCAST_DURING_INFERENCE,),
                model_sd_ops=LTXV_MODEL_COMFY_RENAMING_WITH_TRANSFORMER_LINEAR_DOWNCAST_MAP,
            )
            model = (
                X0Model(fp8_builder.build(device=self._target_device()))
                .to(self.device)
                .eval()
            )
            # Apply FFN chunking for memory-efficient inference
            return self._apply_ffn_chunking(model)

        elif self.quantization == QuantizationType.NVFP4:
            # NVFP4 quantization: build in BF16 first, then quantize to NVFP4
            import gc
            import traceback

            try:
                # Use shared NVFP4 module from scope.core.pipelines
                from scope.core.pipelines.nvfp4 import (
                    check_nvfp4_support,
                    quantize_model_nvfp4,
                    transformer_block_filter,
                )
            except Exception as e:
                logger.error(f"Failed to import nvfp4 module: {e}")
                logger.error(traceback.format_exc())
                raise

            supported, reason = check_nvfp4_support()
            if not supported:
                raise RuntimeError(f"NVFP4 quantization not supported: {reason}")

            logger.info("Building transformer with NVFP4 quantization...")

            # Check available VRAM to decide build strategy
            # For GPUs with < 40GB, use streaming quantization to avoid OOM
            import torch

            low_vram_mode = False
            if torch.cuda.is_available():
                free_mem = torch.cuda.mem_get_info()[0] / 1024**3
                total_mem = torch.cuda.mem_get_info()[1] / 1024**3
                # Transformer needs ~35GB in BF16 before quantization
                # If we have less than 40GB free, use low-VRAM mode
                if free_mem < 40:
                    low_vram_mode = True
                    logger.info(
                        f"Low VRAM mode: {free_mem:.1f}GB free of {total_mem:.1f}GB total. "
                        "Using streaming quantization to avoid OOM."
                    )

            if low_vram_mode:
                # Streaming quantization: build on CPU, quantize block-by-block on GPU
                # This keeps peak VRAM usage low by only having one block on GPU at a time
                logger.info("Building transformer on CPU for streaming quantization...")
                model = X0Model(
                    self.transformer_builder.build(
                        device=torch.device("cpu"), dtype=self.dtype
                    )
                )

                # Quantize using streaming mode - moves blocks to GPU one at a time
                logger.info(
                    "Quantizing transformer blocks to NVFP4 (streaming mode)..."
                )
                try:
                    quantize_model_nvfp4(
                        model,
                        layer_filter=transformer_block_filter,
                        streaming=True,  # Enable streaming quantization
                        target_device=self.device,
                    )
                except Exception as e:
                    logger.error(f"Failed to quantize model to NVFP4: {e}")
                    logger.error(traceback.format_exc())
                    raise

                # Move the final quantized model to GPU
                logger.info("Moving quantized model to GPU...")
                model = model.to(self.device)

                # Force cleanup after quantization
                gc.collect()
                torch.cuda.empty_cache()
            else:
                # Standard path: build directly on GPU
                model = X0Model(
                    self.transformer_builder.build(
                        device=self._target_device(), dtype=self.dtype
                    )
                ).to(self.device)

                # Quantize transformer blocks to NVFP4
                logger.info("Quantizing transformer blocks to NVFP4...")
                try:
                    quantize_model_nvfp4(model, layer_filter=transformer_block_filter)
                except Exception as e:
                    logger.error(f"Failed to quantize model to NVFP4: {e}")
                    logger.error(traceback.format_exc())
                    raise

            model = model.eval()
            # Apply FFN chunking for memory-efficient inference
            return self._apply_ffn_chunking(model)

        else:
            # No quantization (full precision BF16)
            model = (
                X0Model(
                    self.transformer_builder.build(
                        device=self._target_device(), dtype=self.dtype
                    )
                )
                .to(self.device)
                .eval()
            )
            # Apply FFN chunking for memory-efficient inference
            return self._apply_ffn_chunking(model)

    def video_decoder(self) -> VideoDecoder:
        if not hasattr(self, "vae_decoder_builder"):
            raise ValueError(
                "Video decoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        return (
            self.vae_decoder_builder.build(
                device=self._target_device(), dtype=self.dtype
            )
            .to(self.device)
            .eval()
        )

    def video_encoder(self) -> VideoEncoder:
        if not hasattr(self, "vae_encoder_builder"):
            raise ValueError(
                "Video encoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        return (
            self.vae_encoder_builder.build(
                device=self._target_device(), dtype=self.dtype
            )
            .to(self.device)
            .eval()
        )

    def text_encoder(self) -> AVGemmaTextEncoderModel:
        if not hasattr(self, "text_encoder_builder"):
            raise ValueError(
                "Text encoder not initialized. Please provide a checkpoint path and gemma root path to the "
                "ModelLedger constructor."
            )

        return (
            self.text_encoder_builder.build(
                device=self._target_device(), dtype=self.dtype
            )
            .to(self.device)
            .eval()
        )

    def audio_decoder(self) -> AudioDecoder:
        if not hasattr(self, "audio_decoder_builder"):
            raise ValueError(
                "Audio decoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        return (
            self.audio_decoder_builder.build(
                device=self._target_device(), dtype=self.dtype
            )
            .to(self.device)
            .eval()
        )

    def vocoder(self) -> Vocoder:
        if not hasattr(self, "vocoder_builder"):
            raise ValueError(
                "Vocoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        return (
            self.vocoder_builder.build(device=self._target_device(), dtype=self.dtype)
            .to(self.device)
            .eval()
        )

    def spatial_upsampler(self) -> LatentUpsampler:
        if not hasattr(self, "upsampler_builder"):
            raise ValueError(
                "Upsampler not initialized. Please provide upsampler path to the ModelLedger constructor."
            )

        return (
            self.upsampler_builder.build(device=self._target_device(), dtype=self.dtype)
            .to(self.device)
            .eval()
        )
