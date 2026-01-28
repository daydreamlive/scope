# Modified from https://github.com/ali-vilab/VACE/blob/48eb44f1c4be87cc65a98bff985a26976841e9f3/vace/models/wan/modules/model.py
# Adapted for causal/autoregressive generation with factory pattern
# Pipeline-agnostic using duck typing - works with any CausalWanModel
import inspect
import logging
import math

import torch
import torch.nn as nn

from .attention_blocks import (
    create_base_attention_block_class,
    create_vace_attention_block_class,
)

logger = logging.getLogger(__name__)


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
        # Created on CPU by default for memory efficiency
        self._create_vace_blocks()

        # VACE patch embedding - create on CPU for memory efficiency
        # Will be populated with weights via load_vace_weights_only()
        with torch.device("cpu"):
            self.vace_patch_embedding = nn.Conv3d(
                self.vace_in_dim,
                self.dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )

        # Cache block forward signature for dynamic parameter filtering
        # This allows the VACE model to work with any CausalWanModel implementation
        self._block_forward_params = self._get_block_forward_params()

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
        sig = inspect.signature(self._original_block_class.__init__)
        params = sig.parameters

        if "local_attn_size" in params:
            kwargs["local_attn_size"] = self.local_attn_size
        if "sink_size" in params:
            kwargs["sink_size"] = self.sink_size
        if "window_size" in params:
            kwargs["window_size"] = self.window_size

        return kwargs

    def _get_block_forward_params(self):
        """Get the set of parameter names accepted by the block's forward method.

        Inspects the original block class's forward signature to determine which
        parameters should be passed through to blocks. This allows the VACE model
        to work with any CausalWanModel implementation without hardcoding parameter names.

        Returns:
            set: Parameter names accepted by block.forward(), or None if the block
                 accepts **kwargs (VAR_KEYWORD) and can handle any parameters.
        """
        sig = inspect.signature(self._original_block_class.forward)

        # If block accepts **kwargs, return None to indicate all params are accepted
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if has_var_keyword:
            return None

        return set(sig.parameters.keys())

    def _filter_block_kwargs(self, block_kwargs, block_index):
        """Filter and prepare kwargs for a specific block.

        Handles two types of parameters:
        1. Per-block indexed: Lists with length matching num_blocks (e.g., kv_bank)
           These get indexed with block_index.
        2. Shared: Scalar/other values passed to all blocks as-is

        Only includes parameters that the block's forward method accepts.

        Args:
            block_kwargs: Dict of additional kwargs from _forward_inference
            block_index: Index of the current block

        Returns:
            Dict of kwargs filtered and prepared for this specific block
        """
        if not block_kwargs:
            return {}

        filtered = {}
        for key, value in block_kwargs.items():
            # Skip if block doesn't accept this parameter
            if (
                self._block_forward_params is not None
                and key not in self._block_forward_params
            ):
                continue

            # Check if this is a per-block indexed parameter (list matching block count)
            if isinstance(value, list | tuple) and len(value) == self.num_layers:
                filtered[key] = value[block_index]
            else:
                filtered[key] = value

        return filtered

    def _replace_blocks_with_hint_injection_support(self):
        """Replace blocks with BaseWanAttentionBlock to support hint injection.

        Creates new block instances of the factory-generated class and copies
        weights from the original blocks. Uses proper inheritance (not composition),
        so state_dict paths are preserved.

        Memory-optimized: replaces blocks one at a time to avoid doubling memory usage.
        """
        original_blocks = self.causal_wan_model.blocks

        # Get device and dtype from original blocks
        orig_dtype = next(original_blocks[0].parameters()).dtype
        orig_device = next(original_blocks[0].parameters()).device

        # Get initialization kwargs
        block_kwargs = self._get_block_init_kwargs()

        # Replace blocks ONE AT A TIME to minimize peak memory usage
        # This avoids having both old and new blocks in memory simultaneously
        new_blocks = nn.ModuleList()

        for i in range(self.num_layers):
            block_id = self.vace_layers_mapping[i] if i in self.vace_layers else None
            orig_block = original_blocks[i]

            # Create new block on CPU first to avoid GPU memory spike
            with torch.device("cpu"):
                new_block = self._BaseWanAttentionBlock(
                    **block_kwargs,
                    block_id=block_id,
                )

            # Get state dict from original (on GPU) and copy to new block (on CPU)
            orig_state = orig_block.state_dict()
            new_state = new_block.state_dict()
            saved_block_id = new_block.block_id

            for key in orig_state.keys():
                if key in new_state:
                    # Copy to CPU (where new_block is)
                    new_state[key] = orig_state[key].to("cpu")

            new_block.load_state_dict(new_state, strict=False, assign=True)
            new_block.block_id = saved_block_id

            # Move original block to CPU first to free GPU memory immediately
            # This is safe because we've already copied the weights to CPU
            orig_block.to("cpu")

            # Now move new block to target device/dtype (reusing freed GPU memory)
            new_block = new_block.to(device=orig_device, dtype=orig_dtype)
            new_block.eval()

            new_blocks.append(new_block)

            # Clear CUDA cache periodically to help with fragmentation
            if i % 10 == 0:
                torch.cuda.empty_cache()

        # Final cache clear
        torch.cuda.empty_cache()

        # Replace blocks in wrapped model
        self.causal_wan_model.blocks = new_blocks

        # Also register blocks on self for LoRA compatibility
        self.blocks = new_blocks

    def _create_vace_blocks(self):
        """Create VACE blocks for parallel processing of reference images.

        Creates blocks on CPU by default to save GPU memory.
        They will be populated with weights later via load_vace_weights_only().
        """
        # Get dtype from existing blocks (but create on CPU for memory efficiency)
        orig_dtype = next(self.blocks[0].parameters()).dtype

        # Get initialization kwargs
        block_kwargs = self._get_block_init_kwargs()

        # Create VACE blocks on CPU to save GPU memory
        # They will receive weights via load_vace_weights_only() and can optionally
        # be moved to GPU later if memory permits
        vace_blocks = nn.ModuleList()
        with torch.device("cpu"):
            for block_id in range(len(self.vace_layers)):
                vace_block = self._VaceWanAttentionBlock(
                    **block_kwargs,
                    block_id=block_id,
                )
                vace_blocks.append(vace_block)

        # Keep on CPU, use orig_dtype
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
        # Get target dtype from vace_patch_embedding parameters
        target_dtype = next(self.vace_patch_embedding.parameters()).dtype

        # Convert all VACE context to model dtype first
        vace_context_converted = [u.to(dtype=target_dtype) for u in vace_context]

        # Embed VACE context
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context_converted]
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
        x = [self.causal_wan_model.patch_embedding(u.unsqueeze(0)) for u in x]
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

        # Base arguments for transformer blocks (shared across all blocks)
        base_kwargs = {
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

        # --------------------------------------------------------------
        # MagCache support (for VACE-wrapped causal Wan models)
        #
        # Same strategy as the base LongLive CausalWanModel:
        # cache token-level residual right before the head, optionally skip blocks.
        # --------------------------------------------------------------

        def _magcache_reset():
            pass
            '''
            self._magcache_step = 0
            self._magcache_accumulated_ratio = 1.0
            self._magcache_accumulated_err = 0.0
            self._magcache_accumulated_steps = 0
            self._magcache_residual_cache = None
            self._magcache_stats = {"skipped": 0, "computed": 0}
            '''
        if not hasattr(self, "_magcache_reset"):
            self._magcache_reset = _magcache_reset  # type: ignore[attr-defined]

        magcache_cfg = getattr(self, "_magcache_config", None)
        magcache_enabled = bool(getattr(magcache_cfg, "enabled", False))
        magcache_num_steps = getattr(self, "_magcache_num_steps", None)

        if magcache_enabled and not hasattr(self, "_magcache_step"):
            _magcache_reset()

        skip_forward = False
        ori_x = x

        # Debug: track MagCache statistics
        if not hasattr(self, "_magcache_stats"):
            self._magcache_stats = {"skipped": 0, "computed": 0}

        if magcache_enabled and magcache_num_steps is not None:
            from scope.core.pipelines.wan2_1.magcache import (
                wan21_t2v_13b_mag_ratios,
            )

            retention_ratio = float(getattr(magcache_cfg, "retention_ratio", 0.2))
            retain_steps = int(float(magcache_num_steps) * retention_ratio)
            step_idx = int(getattr(self, "_magcache_step", 0))
            residual_cache = getattr(self, "_magcache_residual_cache", None)

            # Match reference implementation: only consider skipping after retention phase
            # The retention phase ensures early steps (which are most important for quality)
            # are always computed fully.
            if step_idx >= retain_steps:
                ratios = getattr(self, "_magcache_ratios", None)
                if ratios is None or int(ratios.shape[0]) != int(magcache_num_steps):
                    ratios = wan21_t2v_13b_mag_ratios(int(magcache_num_steps))
                    setattr(self, "_magcache_ratios", ratios)

                if step_idx < int(ratios.shape[0]):
                    cur_ratio = float(ratios[step_idx])
                    accumulated_ratio = float(
                        getattr(self, "_magcache_accumulated_ratio", 1.0)
                    )
                    accumulated_steps = int(
                        getattr(self, "_magcache_accumulated_steps", 0)
                    )
                    accumulated_err = float(
                        getattr(self, "_magcache_accumulated_err", 0.0)
                    )

                    # Accumulate error estimate for this step
                    # (matches reference: accumulate BEFORE deciding to skip)
                    accumulated_ratio *= cur_ratio
                    accumulated_steps += 1
                    cur_skip_err = abs(1.0 - accumulated_ratio)
                    accumulated_err += cur_skip_err

                    thresh = float(getattr(magcache_cfg, "thresh", 0.12))
                    K = int(getattr(magcache_cfg, "K", 1))

                    # Check if we can skip: need a cached residual, error below threshold,
                    # and haven't exceeded max consecutive skips
                    if (
                        residual_cache is not None
                        and accumulated_err < thresh
                        and accumulated_steps <= K
                    ):
                        skip_forward = True
                        # Save updated accumulation state for next step
                        setattr(self, "_magcache_accumulated_ratio", accumulated_ratio)
                        setattr(self, "_magcache_accumulated_steps", accumulated_steps)
                        setattr(self, "_magcache_accumulated_err", accumulated_err)
                        logger.debug(
                            f"MagCache (VACE) SKIP: step={step_idx}, acc_err={accumulated_err:.4f} < thresh={thresh}, acc_steps={accumulated_steps} <= K={K}"
                        )
                    else:
                        # Force compute: reset accumulation state
                        # (matches reference: reset when NOT skipping)
                        setattr(self, "_magcache_accumulated_ratio", 1.0)
                        setattr(self, "_magcache_accumulated_steps", 0)
                        setattr(self, "_magcache_accumulated_err", 0.0)
                        logger.debug(
                            f"MagCache (VACE) COMPUTE: step={step_idx}, acc_err={accumulated_err:.4f}, thresh={thresh}, "
                            f"acc_steps={accumulated_steps}, K={K}, cache={'exists' if residual_cache is not None else 'None'}"
                        )
            else:
                logger.debug(
                    f"MagCache (VACE) RETAIN: step={step_idx} < retain_steps={retain_steps}"
                )

        # Process through blocks (or skip)
        cache_update_infos = []

        # Safety check: if we're about to skip but the KV cache expects updates,
        # we must not skip to avoid cache index misalignment.
        # This can happen when magcache_K is set too high relative to num_steps.
        if skip_forward and kv_cache is not None:
            # Check if the first block's cache indicates we need an update
            # by comparing current_start with the cache's expected position
            first_cache = kv_cache[0] if kv_cache else None
            if first_cache is not None:
                cache_global_end = first_cache.get("global_end_index")
                if cache_global_end is not None:
                    cache_end_val = (
                        cache_global_end.item()
                        if hasattr(cache_global_end, "item")
                        else int(cache_global_end)
                    )
                    # If current_start is beyond what cache expects, force compute
                    if current_start is not None and current_start > cache_end_val:
                        logger.warning(
                            f"MagCache (VACE): forcing compute due to KV cache misalignment "
                            f"(current_start={current_start}, cache_end={cache_end_val})"
                        )
                        skip_forward = False

        if skip_forward:
            x = ori_x + getattr(self, "_magcache_residual_cache")
            self._magcache_stats["skipped"] += 1
        else:
            self._magcache_stats["computed"] += 1
            for block_index, block in enumerate(self.blocks):
                # Build per-block kwargs:
                # - kv_cache/crossattn_cache are always per-block indexed
                # - Additional block_kwargs are dynamically filtered based on block's signature
                #   and automatically indexed if they're per-block lists
                filtered_block_kwargs = self._filter_block_kwargs(
                    block_kwargs, block_index
                )
                per_block_kwargs = {
                    "kv_cache": kv_cache[block_index],
                    "current_start": current_start,
                    **filtered_block_kwargs,
                }

                if torch.is_grad_enabled() and self.causal_wan_model.gradient_checkpointing:
                    kwargs = {**base_kwargs, **per_block_kwargs}
                    result = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        **kwargs,
                        use_reentrant=False,
                    )
                    if kv_cache is not None and isinstance(result, tuple):
                        x, block_cache_update_info = result
                        cache_update_infos.append((block_index, block_cache_update_info))
                    else:
                        x = result
                else:
                    per_block_kwargs["crossattn_cache"] = crossattn_cache[block_index]
                    kwargs = {**base_kwargs, **per_block_kwargs}
                    result = block(x, **kwargs)
                    if kv_cache is not None and isinstance(result, tuple):
                        x, block_cache_update_info = result
                        cache_update_infos.append((block_index, block_cache_update_info))
                    else:
                        x = result

        if kv_cache is not None and cache_update_infos:
            self.causal_wan_model._apply_cache_updates(
                kv_cache, cache_update_infos, **block_kwargs
            )

        if magcache_enabled and not skip_forward:
            setattr(self, "_magcache_residual_cache", x - ori_x)

        if magcache_enabled and magcache_num_steps is not None:
            next_step = int(getattr(self, "_magcache_step", 0)) + 1
            setattr(self, "_magcache_step", next_step)
            if next_step >= int(magcache_num_steps):
                # Log MagCache stats at end of each chunk
                stats = getattr(self, "_magcache_stats", {"skipped": 0, "computed": 0})
                if stats["skipped"] > 0 or stats["computed"] > 0:
                    total = stats["skipped"] + stats["computed"]
                    skip_pct = 100 * stats["skipped"] / total if total > 0 else 0
                    logger.info(
                        f"MagCache (VACE): skipped {stats['skipped']}/{total} steps ({skip_pct:.1f}%)"
                    )
                # Reset ALL state between chunks.
                # CRITICAL: The residual cache MUST be cleared because each chunk has
                # different latent content. Reusing a residual from chunk N on chunk N+1
                # causes severe artifacts ("looping noise") since the cached residual
                # was computed for completely different spatial content.
                self._magcache_step = 0
                self._magcache_accumulated_ratio = 1.0
                self._magcache_accumulated_err = 0.0
                self._magcache_accumulated_steps = 0
                self._magcache_stats = {"skipped": 0, "computed": 0}
                self._magcache_residual_cache = None  # Clear to prevent cross-chunk artifacts

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
