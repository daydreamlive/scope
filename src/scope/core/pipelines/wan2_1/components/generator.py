# Modified from https://github.com/guandeh17/Self-Forcing
import inspect
import json
import logging
import os
import types

import numpy as np
import torch

from scope.core.pipelines.utils import load_state_dict

from .scheduler import FlowMatchScheduler, SchedulerInterface

logger = logging.getLogger(__name__)

# IOBinding needs a numpy dtype for byte-size calculation.
# bf16 uses uint16 (same 2 bytes); ORT reads the real type from the ONNX graph.
_TORCH_TO_NP_DTYPE = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.bfloat16: np.uint16,
    torch.int64: np.int64,
    torch.int32: np.int32,
}


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

        # Copy model config values to generator so preprocessing doesn't
        # need to reach into self.model (which may be ONNX-compiled).
        self.patch_size = self.model.patch_size
        self.text_len = self.model.text_len
        self.num_layers = self.model.num_layers
        self.freqs = self.model.freqs

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

        # KV cache runtime state (managed by SetupCachesBlock / RecacheFramesBlock)
        self.fill_level = 0
        self.cache_tokens = 0
        self.sink_tokens = 0

        self.ort_session = None

        self.post_init()


    def export_onnx(self, save_dir="onnx_models"):
        """Export self.model to ONNX using latent-space dimensions directly."""
        import onnx

        os.makedirs(save_dir, exist_ok=True)
        onnx_path = os.path.join(save_dir, "model.onnx")

        model = self.model
        device = next(model.parameters()).device
        original_dtype = next(model.parameters()).dtype

        # ORT Conv doesn't support bf16; export in fp16 and cast inputs at runtime
        if original_dtype == torch.bfloat16:
            model.half()
        dtype = torch.float16

        ps = self.patch_size
        f = 3 // ps[0]
        h, w = 60 // ps[1], 104 // ps[2]
        seq = f * h * w
        cs = self.cache_tokens
        nl, nh, hd = model.num_layers, model.num_heads, model.dim // model.num_heads

        dummy = (
            torch.randn(1, model.in_dim, 3, 60, 104, device=device, dtype=dtype),
            torch.zeros(1, 3, device=device, dtype=torch.int64),
            torch.randn(1, self.text_len, model.text_dim, device=device, dtype=dtype),
            torch.randn(1, seq, 1, hd // 2, device=device, dtype=torch.float32),
            torch.randn(1, seq, 1, hd // 2, device=device, dtype=torch.float32),
            torch.zeros(nl, 1, cs, nh, hd, device=device, dtype=dtype),
            torch.zeros(nl, 1, cs, nh, hd, device=device, dtype=dtype),
            torch.zeros(1, 1, 1, cs + seq, device=device, dtype=dtype),
        )

        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=[
                "x", "t", "context", "freqs_cos", "freqs_sin",
                "cache_ks", "cache_vs", "mask",
            ],
            output_names=["output", "new_ks", "new_vs"],
            dynamo=False,
        )

        # Restore original dtype
        if original_dtype == torch.bfloat16:
            model.to(original_dtype)

        # Re-save with external data (model weights exceed 2 GB protobuf limit)
        onnx_model = onnx.load(onnx_path)
        onnx.save_model(
            onnx_model,
            onnx_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.pb",
            convert_attribute=False,
        )
        logger.info("ONNX model saved to %s", save_dir)
        return onnx_path

    def _ensure_onnx_session(self, x_video):
        """Auto-export + load ONNX model on first kv_cache forward call.

        Skipped when self.onnx_dir is None (ONNX mode disabled).
        The cache path includes the latent resolution so different
        resolutions get separate exports.
        """
        if self.ort_session is not None:
            return

        _, _, num_frames, lh, lw = x_video.shape
        res_dir = os.path.join("onnx_models", f"{lh}x{lw}x{num_frames}")
        onnx_path = os.path.join(res_dir, "model.onnx")

        if not os.path.exists(onnx_path):
            logger.info("ONNX model not found at %s, exporting...", res_dir)
            self.export_onnx(res_dir)

        self.load_onnx(res_dir)

    def load_onnx(self, save_dir):
        """Load an exported ONNX model for GPU inference via ORT.

        After calling this, forward() automatically routes through ONNX
        Runtime instead of the PyTorch model for the kv_cache path.
        """
        import onnxruntime as ort

        onnx_path = os.path.join(save_dir, "model.onnx")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.ort_session = ort.InferenceSession(
            onnx_path,
            sess_options=opts,
            providers=["CUDAExecutionProvider"],
        )
        logger.info("ONNX Runtime session loaded from %s", save_dir)

    def _run_onnx(self, x, t, context, freqs_cos, freqs_sin, cache_ks, cache_vs, mask):
        """Run model through ONNX Runtime with GPU IOBinding (zero-copy)."""
        io = self.ort_session.io_binding()
        device_id = x.device.index or 0

        def _cast(tensor):
            if tensor.dtype == torch.bfloat16:
                return tensor.half()
            return tensor

        for name, tensor in [
            ("x", x), ("t", t), ("context", context),
            ("freqs_cos", freqs_cos), ("freqs_sin", freqs_sin),
            ("cache_ks", cache_ks), ("cache_vs", cache_vs), ("mask", mask),
        ]:
            tensor = _cast(tensor).contiguous()
            io.bind_input(
                name, "cuda", device_id,
                _TORCH_TO_NP_DTYPE[tensor.dtype],
                tuple(tensor.shape), tensor.data_ptr(),
            )

        for name in ["output", "new_ks", "new_vs"]:
            io.bind_output(name, "cuda", device_id)

        self.ort_session.run_with_iobinding(io)
        return tuple(torch.from_dlpack(o) for o in io.get_outputs())

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

    # ------------------------------------------------------------------
    # Model call helpers
    # ------------------------------------------------------------------

    def _call_model(self, *args, **kwargs):
        # Inspect the model's forward to determine accepted params and filter out
        # kwargs that might not work with the underlying model implementation.
        # Prefer _forward_inference if available (legacy models), else use forward.
        if hasattr(self.model, "_forward_inference"):
            sig = inspect.signature(self.model._forward_inference)
        else:
            sig = inspect.signature(self.model.forward)

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

    def _preprocess_model_inputs(self, x, context, kv_cache, current_start):
        """Prepare tensor inputs for the CausalWanModel tensor-only forward.

        Handles RoPE precomputation (complex numbers), context padding,
        cache tensor stacking, and attention mask construction -- all the
        non-ONNX-friendly operations that live outside the model.

        Returns:
            (context, freqs_cos, freqs_sin, cache_ks, cache_vs, mask)
        """
        from scope.core.pipelines.longlive.modules.causal_model import (
            precompute_freqs_i,
        )

        self.freqs = self.freqs.to(x.device)

        f = x.shape[2] // self.patch_size[0]
        h = x.shape[3] // self.patch_size[1]
        w = x.shape[4] // self.patch_size[2]

        start_frame = current_start // (h * w)
        freqs_cos, freqs_sin = precompute_freqs_i(self.freqs, f, h, w, start_frame)

        if context.shape[1] < self.text_len:
            padded = torch.zeros(
                (context.shape[0], self.text_len, context.shape[2]),
                dtype=context.dtype,
                device=context.device,
            )
            padded[:, : context.shape[1], :] = context
            context = padded

        cache_ks = torch.stack([kv_cache[i]["k"] for i in range(self.num_layers)])
        cache_vs = torch.stack([kv_cache[i]["v"] for i in range(self.num_layers)])

        cache_size = kv_cache[0]["k"].shape[1]
        seq_len = f * h * w
        mask = torch.zeros(1, 1, 1, cache_size + seq_len, dtype=x.dtype, device=x.device)
        if self.fill_level < cache_size:
            mask[:, :, :, self.fill_level:cache_size] = float("-inf")

        return context, freqs_cos, freqs_sin, cache_ks, cache_vs, mask

    def _update_kv_cache(self, kv_cache, new_ks, new_vs):
        """Update KV cache dicts in-place with new keys/values from model output.

        During filling: appends at fill_level.
        Once full: rolls the non-sink portion and overwrites the tail.

        Args:
            kv_cache: List of dicts (one per layer), each with 'k' and 'v' tensors
            new_ks: Stacked new keys [num_layers, B, seq_len, num_heads, head_dim]
            new_vs: Stacked new values [num_layers, B, seq_len, num_heads, head_dim]
        """
        num_new = new_ks.shape[2]
        can_fit = self.cache_tokens - self.sink_tokens
        take = min(num_new, can_fit)

        for i in range(new_ks.shape[0]):
            if self.fill_level >= self.cache_tokens:
                kv_cache[i]["k"][:, self.sink_tokens :] = torch.roll(
                    kv_cache[i]["k"][:, self.sink_tokens :], -take, dims=1
                )
                kv_cache[i]["v"][:, self.sink_tokens :] = torch.roll(
                    kv_cache[i]["v"][:, self.sink_tokens :], -take, dims=1
                )
                kv_cache[i]["k"][:, -take:] = new_ks[i, :, -take:]
                kv_cache[i]["v"][:, -take:] = new_vs[i, :, -take:]
            else:
                kv_cache[i]["k"][
                    :, self.fill_level : self.fill_level + take
                ] = new_ks[i, :, -take:]
                kv_cache[i]["v"][
                    :, self.fill_level : self.fill_level + take
                ] = new_vs[i, :, -take:]

        self.fill_level = min(self.fill_level + num_new, self.cache_tokens)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

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
        current_start: int | None = None,
        current_end: int | None = None,
        classify_mode: bool | None = False,
        concat_time_embeddings: bool | None = False,
        clean_x: torch.Tensor | None = None,
        aug_t: torch.Tensor | None = None,
        kv_cache_attention_bias: float = 1.0,
        vace_context: torch.Tensor | None = None,
        vace_context_scale: float = 1.0,
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
            x_video = noisy_image_or_video.permute(0, 2, 1, 3, 4)
            self._ensure_onnx_session(x_video)
            context, freqs_cos, freqs_sin, cache_ks, cache_vs, mask = (
                self._preprocess_model_inputs(
                    x_video, prompt_embeds, kv_cache, current_start
                )
            )
            if self.ort_session is not None:
                flow_pred, new_ks, new_vs = self._run_onnx(
                    x_video, input_timestep, context,
                    freqs_cos, freqs_sin, cache_ks, cache_vs, mask,
                )
            else:
                flow_pred, new_ks, new_vs = self.model(
                    x_video,
                    t=input_timestep,
                    context=context,
                    freqs_cos=freqs_cos,
                    freqs_sin=freqs_sin,
                    cache_ks=cache_ks,
                    cache_vs=cache_vs,
                    mask=mask,
                )
            if update_cache:
                self._update_kv_cache(kv_cache, new_ks, new_vs)
            flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
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
