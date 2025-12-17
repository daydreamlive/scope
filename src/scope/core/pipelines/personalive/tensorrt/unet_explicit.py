"""UNet3DConditionModel with explicit reference inputs for TensorRT export.

This module provides a modified UNet3D that accepts reference hidden states
as explicit forward parameters instead of using the attention processor
writer/reader pattern. This is required for ONNX/TensorRT export.

Ported from PersonaLive official implementation:
PersonaLive/src/models/unet_3d_explicit_reference.py
"""

from collections import OrderedDict
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import AttentionProcessor
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME, BaseOutput, logging
from safetensors.torch import load_file

from ..modules.resnet import InflatedConv3d, InflatedGroupNorm
from ..modules.unet_3d_blocks import UNetMidBlock3DCrossAttn, get_down_block, get_up_block

logger = logging.get_logger(__name__)


@dataclass
class UNet3DConditionOutput(BaseOutput):
    sample: torch.FloatTensor


class UNet3DConditionModelExplicit(ModelMixin, ConfigMixin):
    """UNet3D with explicit reference hidden state inputs for TensorRT export.

    This model accepts reference hidden states (d00, d01, ..., u32) as explicit
    forward parameters instead of using attention processor hooks.
    """
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        mid_block_type: str = "UNetMidBlock3DCrossAttn",
        up_block_types: Tuple[str] = (
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        use_inflated_groupnorm=False,
        # Additional
        use_motion_module=False,
        use_temporal_module=False,
        motion_module_resolutions=(1, 2, 4, 8),
        motion_module_mid_block=False,
        motion_module_decoder_only=False,
        motion_module_type=None,
        temporal_module_type=None,
        motion_module_kwargs={},
        temporal_module_kwargs={},
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # input
        self.conv_in = InflatedConv3d(
            in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1)
        )

        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            res = 2**i
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,
                use_inflated_groupnorm=use_inflated_groupnorm,
                use_motion_module=use_motion_module
                and (res in motion_module_resolutions)
                and (not motion_module_decoder_only),
                use_temporal_module=use_temporal_module
                and (res in motion_module_resolutions)
                and (not motion_module_decoder_only),
                motion_module_type=motion_module_type,
                temporal_module_type=temporal_module_type,
                motion_module_kwargs=motion_module_kwargs,
                temporal_module_kwargs=temporal_module_kwargs
            )
            self.down_blocks.append(down_block)

        # mid
        if mid_block_type == "UNetMidBlock3DCrossAttn":
            self.mid_block = UNetMidBlock3DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,
                use_inflated_groupnorm=use_inflated_groupnorm,
                use_motion_module=use_motion_module and motion_module_mid_block,
                use_temporal_module=use_temporal_module and motion_module_mid_block,
                motion_module_type=motion_module_type,
                temporal_module_type=temporal_module_type,
                motion_module_kwargs=motion_module_kwargs,
                temporal_module_kwargs=temporal_module_kwargs,
            )
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")

        # count how many layers upsample the videos
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        only_cross_attention = list(reversed(only_cross_attention))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            res = 2 ** (3 - i)
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,
                use_inflated_groupnorm=use_inflated_groupnorm,
                use_motion_module=use_motion_module
                and (res in motion_module_resolutions),
                use_temporal_module=use_temporal_module
                and (res in motion_module_resolutions),
                motion_module_type=motion_module_type,
                temporal_module_type=temporal_module_type,
                motion_module_kwargs=motion_module_kwargs,
                temporal_module_kwargs=temporal_module_kwargs,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if use_inflated_groupnorm:
            self.conv_norm_out = InflatedGroupNorm(
                num_channels=block_out_channels[0],
                num_groups=norm_num_groups,
                eps=norm_eps,
            )
        else:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0],
                num_groups=norm_num_groups,
                eps=norm_eps,
            )
        self.conv_act = nn.SiLU()
        self.conv_out = InflatedConv3d(
            block_out_channels[0], out_channels, kernel_size=3, padding=1
        )

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        """Returns attention processors."""
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                if "temporal_transformer" not in sub_name:
                    fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            if "temporal_transformer" not in name:
                fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        """Sets the attention processor."""
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        pose_cond_fea: torch.Tensor,
        # Explicit reference hidden states for TensorRT
        d00: torch.Tensor,
        d01: torch.Tensor,
        d10: torch.Tensor,
        d11: torch.Tensor,
        d20: torch.Tensor,
        d21: torch.Tensor,
        m: torch.Tensor,
        u10: torch.Tensor,
        u11: torch.Tensor,
        u12: torch.Tensor,
        u20: torch.Tensor,
        u21: torch.Tensor,
        u22: torch.Tensor,
        u30: torch.Tensor,
        u31: torch.Tensor,
        u32: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        skip_mm: bool = False,
    ) -> Union[UNet3DConditionOutput, Tuple]:
        """Forward pass with explicit reference hidden states.

        Args:
            sample: Noisy latent input (batch, channel, frames, height, width).
            timestep: Timesteps tensor.
            encoder_hidden_states: [clip_embeds, motion_hidden_states] list.
            pose_cond_fea: Pose conditioning features.
            d00-d21: Down block reference hidden states.
            m: Mid block reference hidden states.
            u10-u32: Up block reference hidden states.

        Returns:
            UNet output sample tensor.
        """
        default_overall_up_factor = 2**self.num_upsamplers

        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # time embedding
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # pre-process
        sample = self.conv_in(sample)
        if pose_cond_fea is not None:
            sample = sample + pose_cond_fea

        # down blocks with explicit reference injection
        down_block_res_samples = (sample,)
        for i, downsample_block in enumerate(self.down_blocks):
            # Set reference hidden states for this block
            down_reference = None
            if i == 0:
                down_reference = [d00, d01]
            elif i == 1:
                down_reference = [d10, d11]
            elif i == 2:
                down_reference = [d20, d21]

            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    skip_mm=skip_mm,
                    down_reference=down_reference,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    skip_mm=skip_mm,
                )
            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)
            down_block_res_samples = new_down_block_res_samples

        # mid block with explicit reference
        sample = self.mid_block(
            sample,
            emb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            skip_mm=skip_mm,
            mid_reference=m,
        )

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # up blocks with explicit reference injection
        for i, upsample_block in enumerate(self.up_blocks):
            up_reference = None
            if i == 1:
                up_reference = [u10, u11, u12]
            elif i == 2:
                up_reference = [u20, u21, u22]
            elif i == 3:
                up_reference = [u30, u31, u32]

            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if (
                hasattr(upsample_block, "has_cross_attention")
                and upsample_block.has_cross_attention
            ):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    skip_mm=skip_mm,
                    up_reference=up_reference,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    encoder_hidden_states=encoder_hidden_states,
                    skip_mm=skip_mm,
                )

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample

    @classmethod
    def from_pretrained_2d(
        cls,
        pretrained_model_path: PathLike,
        motion_module_path: PathLike,
        subfolder=None,
        unet_additional_kwargs=None,
        mm_zero_proj_out=False,
    ):
        """Load from pretrained 2D UNet weights."""
        pretrained_model_path = Path(pretrained_model_path)
        motion_module_path = Path(motion_module_path)
        if subfolder is not None:
            pretrained_model_path = pretrained_model_path.joinpath(subfolder)
        logger.info(f"Loading temporal unet from {pretrained_model_path}...")

        config_file = pretrained_model_path / "config.json"
        if not (config_file.exists() and config_file.is_file()):
            raise RuntimeError(f"{config_file} does not exist or is not a file")

        unet_config = cls.load_config(config_file)
        unet_config["_class_name"] = cls.__name__
        unet_config["down_block_types"] = [
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ]
        unet_config["up_block_types"] = [
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
        ]
        unet_config["mid_block_type"] = "UNetMidBlock3DCrossAttn"

        model = cls.from_config(unet_config, **unet_additional_kwargs)

        # Load weights
        if pretrained_model_path.joinpath(SAFETENSORS_WEIGHTS_NAME).exists():
            state_dict = load_file(
                pretrained_model_path.joinpath(SAFETENSORS_WEIGHTS_NAME), device="cpu"
            )
        elif pretrained_model_path.joinpath(WEIGHTS_NAME).exists():
            state_dict = torch.load(
                pretrained_model_path.joinpath(WEIGHTS_NAME),
                map_location="cpu",
                weights_only=True,
            )
        else:
            raise FileNotFoundError(f"No weights file found in {pretrained_model_path}")

        # Load motion module
        if motion_module_path.exists() and motion_module_path.is_file():
            if motion_module_path.suffix.lower() in [".pth", ".pt", ".ckpt"]:
                motion_state_dict = torch.load(
                    motion_module_path, map_location="cpu", weights_only=True
                )
            elif motion_module_path.suffix.lower() == ".safetensors":
                motion_state_dict = load_file(motion_module_path, device="cpu")
            else:
                raise RuntimeError(f"Unknown format: {motion_module_path.suffix}")

            motion_state_dict = {
                k.replace('motion_modules.', 'temporal_modules.'): v
                for k, v in motion_state_dict.items() if "pos_encoder" not in k
            }

            if mm_zero_proj_out:
                motion_state_dict = {
                    k: v for k, v in motion_state_dict.items() if "proj_out" not in k
                }

            state_dict.update(motion_state_dict)

        m, u = model.load_state_dict(state_dict, strict=False)
        logger.debug(f"Missing keys: {len(m)}; Unexpected keys: {len(u)}")

        return model

