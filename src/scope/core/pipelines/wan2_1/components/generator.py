# Modified from https://github.com/guandeh17/Self-Forcing
import inspect
import json
import os
import types

import torch

from scope.core.pipelines.utils import load_state_dict

from .scheduler import FlowMatchScheduler, SchedulerInterface


def filter_causal_model_cls_config(causal_model_cls, config):
    # Filter config to only include parameters accepted by the model's __init__
    sig = inspect.signature(causal_model_cls.__init__)
    config = {k: v for k, v in config.items() if k in sig.parameters}
    return config


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
        self,
        causal_model_cls,
        model_name="Wan2.1-T2V-1.3B",
        timestep_shift=8.0,
        local_attn_size=-1,
        sink_size=0,
        model_dir: str | None = None,
        generator_path: str | None = None,
        generator_model_name: str | None = None,
        **model_kwargs,
    ):
        super().__init__()

        # Use provided model_dir or default to "wan_models"
        model_dir = model_dir if model_dir is not None else "wan_models"
        model_path = os.path.join(model_dir, f"{model_name}/")

        if generator_path:
            config_path = os.path.join(model_path, "config.json")
            with open(config_path) as f:
                config = json.load(f)

            config.update({"local_attn_size": local_attn_size, "sink_size": sink_size})
            # Merge in additional model-specific kwargs (e.g., vace_in_dim for VACE models)
            config.update(model_kwargs)

            state_dict = load_state_dict(generator_path)
            # Handle case where the dict with required keys is nested under a specific key
            # eg state_dict["generator"]
            if generator_model_name is not None:
                state_dict = state_dict[generator_model_name]

            # Remove 'model.' prefix if present (from wrapped models)
            if all(k.startswith("model.") for k in state_dict.keys()):
                state_dict = {
                    k.replace("model.", "", 1): v for k, v in state_dict.items()
                }

            with torch.device("meta"):
                self.model = causal_model_cls(
                    **filter_causal_model_cls_config(causal_model_cls, config)
                )

            # HACK!
            # Store freqs shape before it becomes problematic
            freqs_shape = (
                self.model.freqs.shape if hasattr(self.model, "freqs") else None
            )

            # Move model to CPU first to materialize all buffers and parameters
            self.model = self.model.to_empty(device="cpu")
            # Then load the state dict weights
            # Use strict=False to allow partial loading (e.g., VACE model with non-VACE checkpoint)
            self.model.load_state_dict(state_dict, assign=True, strict=False)

            # HACK!
            # Reinitialize self.freqs properly on CPU (it's not in state_dict)
            if freqs_shape is not None and hasattr(self.model, "freqs"):
                # Get model dimensions to recreate freqs
                d = self.model.dim // self.model.num_heads

                # From Wan2.1 model.py
                def rope_params(max_seq_len, dim, theta=10000):
                    assert dim % 2 == 0
                    freqs = torch.outer(
                        torch.arange(max_seq_len),
                        1.0
                        / torch.pow(
                            theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)
                        ),
                    )
                    freqs = torch.polar(torch.ones_like(freqs), freqs)
                    return freqs

                self.model.freqs = torch.cat(
                    [
                        rope_params(1024, d - 4 * (d // 6)),
                        rope_params(1024, 2 * (d // 6)),
                        rope_params(1024, 2 * (d // 6)),
                    ],
                    dim=1,
                )
        else:
            from_pretrained_config = {
                "local_attn_size": local_attn_size,
                "sink_size": sink_size,
            }
            # Merge in additional model-specific kwargs (e.g., vace_in_dim for VACE models)
            from_pretrained_config.update(model_kwargs)
            self.model = causal_model_cls.from_pretrained(
                model_path,
                **filter_causal_model_cls_config(
                    causal_model_cls,
                    from_pretrained_config,
                ),
            )

        self.model.eval()
        self.model.requires_grad_(False)

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = False

        # Cache configuration (set during setup)
        self._sink_size = 0
        self._frame_seq_length = None
        self._max_attention_size = None

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        # self.seq_len = 1560 * local_attn_size if local_attn_size != -1 else 32760 # [1, 21, 16, 60, 104]
        self.seq_len = (
            1560 * local_attn_size if local_attn_size > 21 else 32760
        )  # [1, 21, 16, 60, 104]
        self.post_init()

    def _convert_flow_pred_to_x0(
        self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = [
            x.double().to(flow_pred.device)
            for x in [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps]
        ]

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(
        scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = [
            x.double().to(x0_pred.device)
            for x in [x0_pred, xt, scheduler.sigmas, scheduler.timesteps]
        ]
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def _call_model(self, *args, **kwargs):
        # HACK!
        # __call__() and forward() accept *args, **kwargs so inspection doesn't tell us anything
        # As a workaround we inspect the internal _forward_inference() function to determine what the accepted params are
        # This allows us to filter out params that might not work with the underlying CausalWanModel impl
        sig = inspect.signature(self.model._forward_inference)

        # Check if the signature accepts **kwargs (VAR_KEYWORD), if so pass all parameters through
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if has_var_keyword:
            accepted = kwargs
        else:
            accepted = {
                name: value for name, value in kwargs.items() if name in sig.parameters
            }
        return self.model(*args, **accepted)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: list[dict] | None = None,
        crossattn_cache: list[dict] | None = None,
        global_end_index: int = 0,
        local_end_index: int = 0,
        current_start: int | None = None,
        current_end: int | None = None,
        update_cache: bool = False,
        vace_context: torch.Tensor | None = None,
        vace_context_scale: float = 1.0,
        # Training-only params
        clean_x: torch.Tensor | None = None,
        aug_t: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Forward pass through the diffusion model.

        Returns:
            flow_pred: Flow prediction tensor
            pred_x0: X0 prediction tensor
            new_global_end_index: Updated global end index (same as input if update_cache=False)
            new_local_end_index: Updated local end index (same as input if update_cache=False)
        """
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        # Compute start_frame from current_start
        if current_start is not None and self._frame_seq_length:
            start_frame = current_start // self._frame_seq_length
        else:
            start_frame = 0

        if kv_cache is not None:
            # Extract past K/V for model
            past_key_values = self._extract_past_key_values(
                kv_cache, global_end_index, local_end_index
            )

            # Call model
            result = self._call_model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep,
                context=prompt_embeds,
                seq_len=self.seq_len,
                past_key_values=past_key_values,
                crossattn_cache=crossattn_cache,
                start_frame=start_frame,
                vace_context=vace_context,
                vace_context_scale=vace_context_scale,
            )

            if isinstance(result, tuple):
                flow_pred, present_key_values = result
            else:
                flow_pred = result
                present_key_values = None

            flow_pred = flow_pred.permute(0, 2, 1, 3, 4)

            # Update cache if requested
            if update_cache and present_key_values is not None:
                new_global_end_index, new_local_end_index = self._update_kv_cache(
                    kv_cache,
                    present_key_values,
                    current_start,
                    current_end,
                    global_end_index,
                    local_end_index,
                )
            else:
                new_global_end_index = global_end_index
                new_local_end_index = local_end_index
        else:
            # No cache path (training, etc.)
            if clean_x is not None:
                # teacher forcing
                flow_pred = self._call_model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep,
                    context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),
                    aug_t=aug_t,
                    vace_context=vace_context,
                    vace_context_scale=vace_context_scale,
                ).permute(0, 2, 1, 3, 4)
            else:
                flow_pred = self._call_model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep,
                    context=prompt_embeds,
                    seq_len=self.seq_len,
                    vace_context=vace_context,
                    vace_context_scale=vace_context_scale,
                ).permute(0, 2, 1, 3, 4)
            new_global_end_index = 0
            new_local_end_index = 0

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        return flow_pred, pred_x0, new_global_end_index, new_local_end_index

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler
        )
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler
        )
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler
        )
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()

    def set_cache_config(
        self, sink_size: int, frame_seq_length: int, max_attention_size: int
    ):
        """Set cache configuration. Called by SetupCachesBlock."""
        self._sink_size = sink_size
        self._frame_seq_length = frame_seq_length
        self._max_attention_size = max_attention_size

    def _extract_past_key_values(
        self,
        kv_cache: list[dict],
        global_end_index: int,
        local_end_index: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract past K/V for attention computation.
        Returns sink tokens + sliding window tokens for each layer.
        """
        # If no data in cache yet, return empty tensors
        if local_end_index == 0:
            past_key_values = []
            for layer_cache in kv_cache:
                k = layer_cache["k"]
                batch_size, _, num_heads, head_dim = k.shape
                empty_k = k.new_empty(batch_size, 0, num_heads, head_dim)
                empty_v = k.new_empty(batch_size, 0, num_heads, head_dim)
                past_key_values.append((empty_k, empty_v))
            return past_key_values

        sink_tokens = self._sink_size * self._frame_seq_length
        local_budget = self._max_attention_size - sink_tokens

        past_key_values = []
        for layer_cache in kv_cache:
            k = layer_cache["k"]
            v = layer_cache["v"]

            if sink_tokens > 0 and local_end_index >= sink_tokens:
                # Extract sink tokens (only if we have enough data)
                k_sink = k[:, :sink_tokens]
                v_sink = v[:, :sink_tokens]

                if local_budget > 0 and local_end_index > sink_tokens:
                    # Extract sliding window tokens
                    window_start = max(sink_tokens, local_end_index - local_budget)
                    k_window = k[:, window_start:local_end_index]
                    v_window = v[:, window_start:local_end_index]

                    # Concatenate sink + window
                    past_k = torch.cat([k_sink, k_window], dim=1)
                    past_v = torch.cat([v_sink, v_window], dim=1)
                else:
                    past_k = k_sink
                    past_v = v_sink
            elif sink_tokens > 0 and local_end_index < sink_tokens:
                # We have some data but not enough to fill sink region yet
                # Just use what we have
                past_k = k[:, :local_end_index]
                past_v = v[:, :local_end_index]
            else:
                # No sink, just sliding window
                window_start = max(0, local_end_index - self._max_attention_size)
                past_k = k[:, window_start:local_end_index]
                past_v = v[:, window_start:local_end_index]

            past_key_values.append((past_k, past_v))

        return past_key_values

    def _update_kv_cache(
        self,
        kv_cache: list[dict],
        present_key_values: list[tuple[torch.Tensor, torch.Tensor]],
        current_start: int,
        current_end: int,
        global_end_index: int,
        local_end_index: int,
    ) -> tuple[int, int]:
        """
        Update KV cache with new K/V from model.
        Handles rolling eviction when cache is full.
        Returns updated (global_end_index, local_end_index).
        """
        sink_tokens = self._sink_size * self._frame_seq_length
        num_new_tokens = current_end - current_start
        kv_cache_size = kv_cache[0]["k"].shape[1]

        # Check if this is a recompute (e.g., recaching)
        is_recompute = current_end <= global_end_index and current_start > 0

        # Determine if we need to roll the cache
        need_roll = (
            self.model.local_attn_size != -1
            and current_end > global_end_index
            and num_new_tokens + local_end_index > kv_cache_size
        )

        if need_roll:
            # Calculate eviction
            num_evicted_tokens = num_new_tokens + local_end_index - kv_cache_size
            num_rolled_tokens = local_end_index - num_evicted_tokens - sink_tokens

            # Roll cache for each layer
            for layer_cache in kv_cache:
                k = layer_cache["k"]
                v = layer_cache["v"]

                # Shift tokens left (evict oldest non-sink tokens)
                k[:, sink_tokens : sink_tokens + num_rolled_tokens] = k[
                    :,
                    sink_tokens
                    + num_evicted_tokens : sink_tokens
                    + num_evicted_tokens
                    + num_rolled_tokens,
                ].clone()
                v[:, sink_tokens : sink_tokens + num_rolled_tokens] = v[
                    :,
                    sink_tokens
                    + num_evicted_tokens : sink_tokens
                    + num_evicted_tokens
                    + num_rolled_tokens,
                ].clone()

            # Compute new local indices
            new_local_end_index = (
                local_end_index + (current_end - global_end_index) - num_evicted_tokens
            )
            local_start_index = new_local_end_index - num_new_tokens
        else:
            # Direct insert (no rolling needed)
            new_local_end_index = local_end_index + (current_end - global_end_index)
            local_start_index = new_local_end_index - num_new_tokens

        # Determine write position (protect sink during recompute)
        write_start_index = (
            max(local_start_index, sink_tokens) if is_recompute else local_start_index
        )
        write_len = new_local_end_index - write_start_index
        roped_offset = write_start_index - local_start_index

        # Write new K/V to cache
        if write_len > 0:
            for layer_idx, (new_k, new_v) in enumerate(present_key_values):
                kv_cache[layer_idx]["k"][
                    :, write_start_index:new_local_end_index
                ] = new_k[:, roped_offset : roped_offset + write_len]
                kv_cache[layer_idx]["v"][
                    :, write_start_index:new_local_end_index
                ] = new_v[:, roped_offset : roped_offset + write_len]

        # Update indices (but not during recompute)
        if not is_recompute:
            new_global_end_index = current_end
        else:
            new_global_end_index = global_end_index

        return new_global_end_index, new_local_end_index
