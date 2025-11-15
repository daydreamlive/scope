import sys
from pathlib import Path

import torch
from diffusers.modular_pipelines import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    ConfigSpec,
    InputParam,
    OutputParam,
)


class ControlNetConditioningBlock(ModularPipelineBlocks):
    """Optional ControlNet conditioning block for LongLive.

    This block prepares frame-aligned controlnet_states for the underlying
    causal Wan model, based on a precomputed buffer of control frames.

    It is intentionally minimal and experimental: it assumes that
    control_frames_buffer has already been prepared and stored in the
    pipeline state as a tensor with shape [B, 3, T_total, H, W].
    """

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("generator", torch.nn.Module),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec("num_frame_per_block", 3),
            ConfigSpec("local_attn_size", 12),
            ConfigSpec("vae_spatial_downsample_factor", 8),
            ConfigSpec("patch_embedding_spatial_downsample_factor", 2),
        ]

    @property
    def description(self) -> str:
        return "Prepare ControlNet conditioning states (experimental, LongLive-only)"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="Current noisy latents [B, F, C, H_latent, W_latent]",
            ),
            InputParam(
                "prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Text embeddings used to condition denoising",
            ),
            InputParam(
                "current_start_frame",
                required=True,
                type_hint=int,
                description="Current starting frame index for current block",
            ),
            InputParam(
                "start_frame",
                type_hint=int | None,
                description="Starting frame index that overrides current_start_frame",
            ),
            InputParam(
                "height",
                type_hint=int,
                description="Height of the video",
            ),
            InputParam(
                "width",
                type_hint=int,
                description="Width of the video",
            ),
            InputParam(
                "control_frames_buffer",
                type_hint=torch.Tensor | None,
                description=(
                    "Precomputed control frames buffer with shape "
                    "[B, 3, T_total, H, W]. If None, this block is a no-op."
                ),
            ),
            InputParam(
                "controlnet_weight",
                default=1.0,
                type_hint=float,
                description="Strength of ControlNet influence (0.0 to 2.0)",
            ),
            InputParam(
                "controlnet_stride",
                default=3,
                type_hint=int,
                description="Apply ControlNet residual every Nth transformer block",
            ),
            InputParam(
                "controlnet_compression_ratio",
                default=4,
                type_hint=int,
                description=(
                    "Temporal compression ratio between control frames and student "
                    "frames (4 for Wan 1.3B teacher)."
                ),
            ),
            InputParam(
                "denoising_step_list",
                type_hint=torch.Tensor | None,
                description="Denoising step list used for generator timesteps",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "controlnet_states",
                description=(
                    "Tuple of per-block ControlNet states to be consumed by the "
                    "denoising block (or None if disabled)."
                ),
            ),
            OutputParam(
                "controlnet_weight",
                type_hint=float,
                description="Strength of ControlNet influence",
            ),
            OutputParam(
                "controlnet_stride",
                type_hint=int,
                description="ControlNet stride in transformer blocks",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)

        # If no control frames are provided, leave controlnet_states unset.
        if block_state.control_frames_buffer is None:
            self.set_block_state(state, block_state)
            return components, state

        # Lazily import the WanControlnet teacher implementation.
        # This keeps the dependency scoped to experimental usage.
        from importlib import import_module

        project_root = Path(__file__).resolve().parents[3]
        controlnet_path = project_root / "notes" / "wan2.1-dilated-controlnet"
        if str(controlnet_path) not in sys.path:
            sys.path.insert(0, str(controlnet_path))

        WanControlnet = import_module("wan_controlnet").WanControlnet  # type: ignore[attr-defined]

        device = components.config.device
        dtype = components.config.dtype

        # Lazily initialize the teacher ControlNet and cache it on the block.
        if not hasattr(self, "_controlnet"):
            # Initialize WanControlnet with the real canny teacher weights.
            # This mirrors the configuration from the previous research branch.
            controlnet_config = {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "image_dim": None,
                "in_channels": 3,
                "num_attention_heads": 12,
                "num_layers": 8,
                "patch_size": (1, 2, 2),
                "qk_norm": "rms_norm_across_heads",
                "rope_max_seq_len": 1024,
                "text_dim": 4096,
                "downscale_coef": 8,
                "out_proj_dim": 12 * 128,
            }
            self._controlnet = WanControlnet(**controlnet_config)  # type: ignore[call-arg]

            # Load pretrained weights from HuggingFace.
            import safetensors.torch
            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(
                repo_id="TheDenk/wan2.1-t2v-1.3b-controlnet-canny-v1",
                filename="diffusion_pytorch_model.safetensors",
            )
            state_dict = safetensors.torch.load_file(model_path)
            self._controlnet.load_state_dict(state_dict)

            self._controlnet = self._controlnet.to(device=device, dtype=dtype)
            self._controlnet.eval()

        controlnet = self._controlnet

        noise_latents = block_state.latents
        batch_size, num_frames, _, _, _ = noise_latents.shape

        start_frame = block_state.current_start_frame
        if block_state.start_frame is not None:
            start_frame = block_state.start_frame

        # Map student frame indices into teacher control frame indices.
        compression_ratio = int(block_state.controlnet_compression_ratio)
        raw_start = start_frame * compression_ratio
        target_len = num_frames * compression_ratio
        raw_end = raw_start + target_len

        # Compute a window of exactly target_len frames whenever possible.
        total_control_frames = block_state.control_frames_buffer.shape[2]

        if total_control_frames >= target_len:
            # Prefer [raw_start, raw_start + target_len); if that would run past
            # the end of the buffer, shift the window backwards.
            if raw_end <= total_control_frames:
                start_idx = max(0, raw_start)
                end_idx = start_idx + target_len
            else:
                end_idx = total_control_frames
                start_idx = max(0, end_idx - target_len)
        else:
            # Not enough frames overall – use the full buffer and pad later.
            start_idx = 0
            end_idx = total_control_frames

        print(
            f"controlnet_conditioning: control_frames_buffer shape: "
            f"{block_state.control_frames_buffer.shape}"
        )
        print(
            f"controlnet_conditioning: raw_start: {raw_start}, raw_end: {raw_end}, "
            f"target_len: {target_len}"
        )
        print(
            f"controlnet_conditioning: total_control_frames: {total_control_frames}, "
            f"start_idx: {start_idx}, end_idx: {end_idx}"
        )

        control_frames = block_state.control_frames_buffer[
            :, :, start_idx:end_idx, :, :
        ]

        print(
            "controlnet_conditioning: control_frames shape (after slicing): "
            f"{control_frames.shape}"
        )

        # Ensure we always provide exactly target_len frames to the teacher
        # so that the temporal compression (4x) yields num_frames outputs.
        current_len = control_frames.shape[2]
        if current_len < target_len:
            if current_len == 0:
                raise RuntimeError(
                    "ControlNetConditioningBlock: control_frames_buffer slice is empty"
                )
            pad_needed = target_len - current_len
            last = control_frames[:, :, -1:, :, :].expand(-1, -1, pad_needed, -1, -1)
            control_frames = torch.cat([control_frames, last], dim=2)
            print(
                "controlnet_conditioning: padded control_frames to target_len, "
                f"new shape: {control_frames.shape}"
            )

        # Prepare inputs for ControlNet teacher.
        hidden_states = noise_latents.permute(0, 2, 1, 3, 4).to(
            device=device, dtype=dtype
        )
        control_frames = control_frames.to(device=device, dtype=dtype)
        prompt_embeds = block_state.prompt_embeds.to(device=device, dtype=dtype)

        print(f"controlnet_conditioning: noise_latents shape: {noise_latents.shape}")
        print(
            f"controlnet_conditioning: hidden_states shape (after permute): {hidden_states.shape}"
        )
        print(f"controlnet_conditioning: control_frames shape: {control_frames.shape}")
        print(f"controlnet_conditioning: prompt_embeds shape: {prompt_embeds.shape}")

        # Use the first denoising step as the teacher timestep for this block
        # to mirror the previous research behavior as closely as possible in
        # the modular pipeline.
        if block_state.denoising_step_list is not None:
            base_timestep = int(block_state.denoising_step_list[0].item())
        else:
            base_timestep = 0

        timestep = torch.full(
            (batch_size,),
            base_timestep,
            device=device,
            dtype=torch.long,
        )

        print(f"controlnet_conditioning: timestep shape: {timestep.shape}")
        print(
            f"controlnet_conditioning: batch_size: {batch_size}, num_frames: {num_frames}"
        )
        print(
            f"controlnet_conditioning: start_frame: {start_frame}, start_idx: {start_idx}, end_idx: {end_idx}"
        )
        print(f"controlnet_conditioning: compression_ratio: {compression_ratio}")
        print("controlnet_conditioning: About to call controlnet with:")
        print(f"  - hidden_states shape: {hidden_states.shape}")
        print(f"  - controlnet_states (control_frames) shape: {control_frames.shape}")
        print(f"  - encoder_hidden_states (prompt_embeds) shape: {prompt_embeds.shape}")
        print(f"  - timestep shape: {timestep.shape}")

        controlnet_hidden_states = controlnet(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            controlnet_states=control_frames,
            return_dict=False,
        )[0]

        # controlnet_hidden_states is a tuple of per-block tensors.
        if len(controlnet_hidden_states) > 0:
            print(
                "controlnet_conditioning: received "
                f"{len(controlnet_hidden_states)} control blocks, "
                f"first block shape: {controlnet_hidden_states[0].shape}"
            )

        block_state.controlnet_states = controlnet_hidden_states
        block_state.controlnet_weight = block_state.controlnet_weight
        block_state.controlnet_stride = block_state.controlnet_stride

        self.set_block_state(state, block_state)
        return components, state
