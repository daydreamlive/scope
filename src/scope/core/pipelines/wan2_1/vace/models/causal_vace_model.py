# Modified from https://github.com/ali-vilab/VACE/blob/48eb44f1c4be87cc65a98bff985a26976841e9f3/vace/models/wan/modules/model.py
# Adapted for causal/autoregressive generation with factory pattern
# Pipeline-agnostic using duck typing - works with any CausalWanModel
import math

import torch
import torch.nn as nn

from .attention_blocks import (
    create_base_attention_block_class,
    create_vace_attention_block_class,
)


# TODO: Consolidate this with other pipeline implementations into a shared wan2_1/utils module.
# This is a standard sinusoidal positional embedding - identical across all pipelines apart from krea which has forced dtype
def sinusoidal_embedding_1d(dim, position):
    """
    Standard sinusoidal positional embedding.

    Args:
        dim: Embedding dimension
        position: Position tensor of shape [B]

    Returns:
        Embeddings of shape [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=position.device) / half
    )
    args = position[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class CausalVaceWanModel(nn.Module):
    """
    VACE wrapper that adds reference image conditioning to any CausalWanModel.

    Uses composition to wrap an existing CausalWanModel instance.
    Pipeline-agnostic via duck typing - works with longlive, streamdiffusionv2,
    krea_realtime_video, reward_forcing, or any future CausalWanModel implementation.
    """

    def __init__(
        self,
        causal_wan_model,
        vace_in_dim=96,
        vace_layers=None,
    ):
        super().__init__()

        # Store wrapped model
        self.causal_wan_model = causal_wan_model

        # Extract configuration from wrapped model via duck typing
        self.num_layers = causal_wan_model.num_layers
        self.dim = causal_wan_model.dim
        self.ffn_dim = causal_wan_model.ffn_dim
        self.num_heads = causal_wan_model.num_heads
        self.qk_norm = causal_wan_model.qk_norm
        self.cross_attn_norm = causal_wan_model.cross_attn_norm
        self.eps = causal_wan_model.eps
        self.model_type = causal_wan_model.model_type
        self.patch_size = causal_wan_model.patch_size
        self.in_dim = causal_wan_model.in_dim

        # Pipeline-specific attributes (duck typed with defaults)
        self.local_attn_size = getattr(causal_wan_model, "local_attn_size", -1)
        self.window_size = getattr(causal_wan_model, "window_size", (-1, -1))
        if hasattr(causal_wan_model, "config") and hasattr(
            causal_wan_model.config, "sink_size"
        ):
            self.sink_size = causal_wan_model.config.sink_size
        else:
            self.sink_size = getattr(causal_wan_model, "sink_size", 0)

        # VACE configuration
        self.vace_layers = (
            list(range(0, self.num_layers, 2)) if vace_layers is None else vace_layers
        )
        self.vace_in_dim = vace_in_dim

        assert 0 in self.vace_layers
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # Get the original block class BEFORE replacing blocks
        self._original_block_class = type(causal_wan_model.blocks[0])
        import inspect

        block_forward_params = inspect.signature(self._original_block_class.forward).parameters
        self._block_forward_accepts_cache_start = "cache_start" in block_forward_params
        self._block_forward_accepts_current_end = "current_end" in block_forward_params
        self._block_forward_accepts_kv_cache_attention_bias = (
            "kv_cache_attention_bias" in block_forward_params
        )

        # Create factory-generated classes for this pipeline's block type
        self._BaseWanAttentionBlock = create_base_attention_block_class(
            self._original_block_class
        )
        self._VaceWanAttentionBlock = create_vace_attention_block_class(
            self._original_block_class
        )

        # Replace blocks with hint-injection-enabled versions
        self._replace_blocks_with_hint_injection_support()

        # Create VACE blocks (parallel processing path for reference images)
        self._create_vace_blocks()

        # VACE patch embedding
        self.vace_patch_embedding = nn.Conv3d(
            self.vace_in_dim,
            self.dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        # Cache: VACE patch-embedded context is constant across denoise steps within a chunk.
        # Cache the patch embedding + flatten/pad work to avoid re-running it per timestep.
        self._cached_vace_context_key: tuple | None = None
        self._cached_vace_context_tokens: torch.Tensor | None = None

    def _prepare_vace_context_tokens(
        self, vace_context: list[torch.Tensor], seq_len: int
    ) -> torch.Tensor:
        # Embed VACE context
        c = [self._vace_patch_embed(u) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]

        # Pad to seq_len
        c = torch.cat(
            [
                torch.cat(
                    [u, u.new_zeros(1, max(0, seq_len - u.size(1)), u.size(2))], dim=1
                )
                for u in c
            ]
        )
        return c

    def _get_cached_vace_context_tokens(
        self, vace_context: list[torch.Tensor], seq_len: int
    ) -> torch.Tensor:
        # Invalidate when the backing tensors change or the patch-embedding weights change.
        bias = self.vace_patch_embedding.bias
        key = (
            int(seq_len),
            int(getattr(self.vace_patch_embedding.weight, "_version", 0)),
            int(getattr(bias, "_version", 0)) if bias is not None else None,
            tuple(id(u) for u in vace_context),
        )
        if key == self._cached_vace_context_key and self._cached_vace_context_tokens is not None:
            return self._cached_vace_context_tokens
        c = self._prepare_vace_context_tokens(vace_context, seq_len)
        self._cached_vace_context_key = key
        self._cached_vace_context_tokens = c
        return c

    def _get_block_init_kwargs(self):
        """Get initialization kwargs for creating new blocks.

        Uses duck typing to determine which parameters the block class expects.
        """
        cross_attn_type = (
            "t2v_cross_attn" if self.model_type == "t2v" else "i2v_cross_attn"
        )

        # Base kwargs that all blocks should have
        kwargs = {
            "cross_attn_type": cross_attn_type,
            "dim": self.dim,
            "ffn_dim": self.ffn_dim,
            "num_heads": self.num_heads,
            "qk_norm": self.qk_norm,
            "cross_attn_norm": self.cross_attn_norm,
            "eps": self.eps,
        }

        # Add pipeline-specific kwargs based on what the original block class expects
        import inspect

        sig = inspect.signature(self._original_block_class.__init__)
        params = sig.parameters

        if "local_attn_size" in params:
            kwargs["local_attn_size"] = self.local_attn_size
        if "sink_size" in params:
            kwargs["sink_size"] = self.sink_size
        if "window_size" in params:
            kwargs["window_size"] = self.window_size

        return kwargs

    def _replace_blocks_with_hint_injection_support(self):
        """Replace blocks with BaseWanAttentionBlock to support hint injection.

        Creates new block instances of the factory-generated class and copies
        weights from the original blocks. Uses proper inheritance (not composition),
        so state_dict paths are preserved.

        Memory-optimized: replaces blocks one at a time to avoid doubling memory
        usage when wrapping large models (e.g. 14B).
        """
        original_blocks = self.causal_wan_model.blocks

        # Get device and dtype from original blocks
        orig_dtype = next(original_blocks[0].parameters()).dtype
        orig_device = next(original_blocks[0].parameters()).device

        # Get initialization kwargs
        block_kwargs = self._get_block_init_kwargs()

        # Replace blocks one-at-a-time to minimize peak memory usage.
        new_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block_id = self.vace_layers_mapping[i] if i in self.vace_layers else None
            orig_block = original_blocks[i]

            with torch.device("cpu"):
                new_block = self._BaseWanAttentionBlock(
                    **block_kwargs,
                    block_id=block_id,
                )

            orig_state = orig_block.state_dict()
            new_state = new_block.state_dict()
            saved_block_id = new_block.block_id

            for key in orig_state.keys():
                if key in new_state:
                    new_state[key] = orig_state[key].detach().to("cpu")

            new_block.load_state_dict(new_state, strict=False, assign=True)
            new_block.block_id = saved_block_id

            # Drop the original block reference early so its parameters can be freed.
            # This avoids a full-model 2x peak during wrapping.
            original_blocks[i] = nn.Identity()
            del orig_block
            del orig_state
            del new_state

            new_block = new_block.to(device=orig_device, dtype=orig_dtype)
            new_block.eval()
            new_blocks.append(new_block)

            if torch.cuda.is_available() and i % 10 == 0:
                torch.cuda.empty_cache()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Replace blocks in wrapped model
        self.causal_wan_model.blocks = new_blocks

        # Also register blocks on self for LoRA compatibility
        self.blocks = new_blocks

    def _create_vace_blocks(self):
        """Create VACE blocks for parallel processing of reference images.

        Create on CPU by default; the owning pipeline can move these to the
        target (device, dtype) before loading VACE weights.
        """
        # Get dtype from existing blocks
        orig_dtype = next(self.blocks[0].parameters()).dtype

        # Get initialization kwargs
        block_kwargs = self._get_block_init_kwargs()

        # Create VACE blocks on CPU to minimize peak memory usage during init.
        vace_blocks = nn.ModuleList()
        with torch.device("cpu"):
            for block_id in range(len(self.vace_layers)):
                vace_block = self._VaceWanAttentionBlock(
                    **block_kwargs,
                    block_id=block_id,
                )
                vace_blocks.append(vace_block)

        vace_blocks.to(dtype=orig_dtype)

        self.vace_blocks = vace_blocks

    def forward_vace(
        self,
        x,
        vace_context,
        seq_len,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        crossattn_cache,
    ):
        """Process VACE context to generate hints."""
        c = self._get_cached_vace_context_tokens(vace_context, seq_len)

        # Process through VACE blocks
        for _block_idx, block in enumerate(self.vace_blocks):
            c = block.forward_vace(
                c,
                x,
                e,
                seq_lens,
                grid_sizes,
                freqs,
                context,
                context_lens,
                block_mask,
                crossattn_cache,
            )

        # Extract hints
        hints = torch.unbind(c)[:-1]
        return hints

    def _patch_embed(self, u: torch.Tensor) -> torch.Tensor:
        """Patch-embed a single latent sample, preferring pipeline fastpaths.

        Some pipelines (e.g. krea_realtime_video) provide a Conv3d(t=1) → Conv2d
        fastpath to avoid slow Conv3d implementations on some backends. When present, prefer it.
        """
        patch_embed = getattr(self.causal_wan_model, "_patch_embed", None)
        if callable(patch_embed):
            return patch_embed(u)
        return self.causal_wan_model.patch_embedding(u.unsqueeze(0))

    def _vace_patch_embed(self, u: torch.Tensor) -> torch.Tensor:
        """Patch-embed a single VACE context sample.

        VACE uses a Conv3d patch embedding with the same (t,h,w) patch size as the
        base model. When t==1, apply an equivalent Conv2d per frame to avoid slow
        Conv3d paths on some backends.
        """
        u = u.unsqueeze(0)  # [1, C, F, H, W]
        try:
            t_patch, h_patch, w_patch = self.patch_size
        except Exception:
            return self.vace_patch_embedding(u)

        if int(t_patch) != 1:
            return self.vace_patch_embedding(u)

        b, c, f, h, w = u.shape
        u2 = u.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)  # [B*F, C, H, W]

        out2 = torch.nn.functional.conv2d(
            u2,
            self.vace_patch_embedding.weight.squeeze(2),
            bias=self.vace_patch_embedding.bias,
            stride=(int(h_patch), int(w_patch)),
            padding=0,
        )

        out = out2.reshape(b, f, out2.shape[1], out2.shape[2], out2.shape[3]).permute(
            0, 2, 1, 3, 4
        )
        return out

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        vace_context=None,
        vace_context_scale=1.0,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        current_end=0,
        cache_start=0,
        kv_cache_attention_bias=1.0,
        **block_kwargs,
    ):
        """Forward pass with optional VACE conditioning."""
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None

        device = self.causal_wan_model.patch_embedding.weight.device
        if self.causal_wan_model.freqs.device != device:
            self.causal_wan_model.freqs = self.causal_wan_model.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y, strict=False)]

        # Embeddings
        x = [self._patch_embed(u) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)

        # Time embeddings
        e = self.causal_wan_model.time_embedding(
            sinusoidal_embedding_1d(
                self.causal_wan_model.freq_dim, t.flatten()
            ).type_as(x)
        )
        e0 = (
            self.causal_wan_model.time_projection(e)
            .unflatten(1, (6, self.dim))
            .unflatten(dim=0, sizes=t.shape)
        )

        # Context
        context_lens = None
        context = self.causal_wan_model.text_embedding(
            torch.stack(
                [
                    torch.cat(
                        [
                            u,
                            u.new_zeros(
                                self.causal_wan_model.text_len - u.size(0), u.size(1)
                            ),
                        ]
                    )
                    for u in context
                ]
            )
        )

        if clip_fea is not None:
            context_clip = self.causal_wan_model.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        # Generate VACE hints
        hints = None
        if vace_context is not None:
            hints = self.forward_vace(
                x,
                vace_context,
                seq_len,
                e0,
                seq_lens,
                grid_sizes,
                self.causal_wan_model.freqs,
                context,
                context_lens,
                self.causal_wan_model.block_mask,
                crossattn_cache,
            )

        # Arguments for transformer blocks
        kwargs = {
            "e": e0,
            "seq_lens": seq_lens,
            "grid_sizes": grid_sizes,
            "freqs": self.causal_wan_model.freqs,
            "context": context,
            "context_lens": context_lens,
            "block_mask": self.causal_wan_model.block_mask,
            "hints": hints,
            "context_scale": vace_context_scale,
        }

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)

            return custom_forward

        # Process through blocks
        cache_update_infos = []
        for block_index, block in enumerate(self.blocks):
            block_call_kwargs = {
                "kv_cache": kv_cache[block_index],
                "current_start": current_start,
                **block_kwargs,
            }
            if self._block_forward_accepts_current_end:
                block_call_kwargs["current_end"] = current_end
            if self._block_forward_accepts_cache_start:
                block_call_kwargs["cache_start"] = cache_start
            if self._block_forward_accepts_kv_cache_attention_bias:
                block_call_kwargs["kv_cache_attention_bias"] = kv_cache_attention_bias

            if torch.is_grad_enabled() and self.causal_wan_model.gradient_checkpointing:
                result = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    **block_call_kwargs,
                    use_reentrant=False,
                )
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                else:
                    x = result
            else:
                block_call_kwargs["crossattn_cache"] = crossattn_cache[block_index]
                result = block(x, **kwargs, **block_call_kwargs)
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                else:
                    x = result

        if kv_cache is not None and cache_update_infos:
            self.causal_wan_model._apply_cache_updates(kv_cache, cache_update_infos)

        x = self.causal_wan_model.head(
            x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2)
        )
        x = self.causal_wan_model.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(self, *args, **kwargs):
        if kwargs.get("kv_cache", None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self.causal_wan_model._forward_train(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.causal_wan_model, name)
