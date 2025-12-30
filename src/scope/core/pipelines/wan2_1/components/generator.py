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
        kv_bank: list[dict] | None = None,
        update_bank: bool | None = False,
        q_bank: bool | None = False,
        update_cache: bool | None = False,
        is_recache: bool | None = False,
        current_start: int | None = None,
        current_end: int | None = None,
        classify_mode: bool | None = False,
        concat_time_embeddings: bool | None = False,
        clean_x: torch.Tensor | None = None,
        aug_t: torch.Tensor | None = None,
        cache_start: int | None = None,
        kv_cache_attention_bias: float = 1.0,
        vace_context: torch.Tensor | None = None,
        vace_context_scale: float = 1.0,
        sink_recache_after_switch: bool = False,
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        logits = None
        # X0 prediction
        if kv_cache is not None:
            flow_pred = self._call_model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep,
                context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                kv_bank=kv_bank,
                update_bank=update_bank,
                q_bank=q_bank,
                update_cache=update_cache,
                is_recache=is_recache,
                current_start=current_start,
                current_end=current_end,
                cache_start=cache_start,
                kv_cache_attention_bias=kv_cache_attention_bias,
                vace_context=vace_context,
                vace_context_scale=vace_context_scale,
                sink_recache_after_switch=sink_recache_after_switch,
            ).permute(0, 2, 1, 3, 4)
        else:
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
                if classify_mode:
                    flow_pred, logits = self._call_model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep,
                        context=prompt_embeds,
                        seq_len=self.seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings,
                        vace_context=vace_context,
                        vace_context_scale=vace_context_scale,
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    flow_pred = self._call_model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep,
                        context=prompt_embeds,
                        seq_len=self.seq_len,
                        vace_context=vace_context,
                        vace_context_scale=vace_context_scale,
                    ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0

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
