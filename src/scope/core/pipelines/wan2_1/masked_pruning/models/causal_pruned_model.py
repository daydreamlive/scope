import inspect
import math

import torch
import torch.nn as nn

from .attention_blocks import create_pruned_block_class


# Duplicated from CausalVaceWanModel to avoid coupling
def sinusoidal_embedding_1d(dim, position):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=position.device) / half
    )
    args = position[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class CausalPrunedWanModel(nn.Module):
    """Masked pruning wrapper that adds token pruning to any CausalWanModel.

    Uses composition to wrap an existing model instance (which may itself be
    a CausalVaceWanModel). No new weights are introduced; the wrapper reuses
    existing weights and only replaces attention blocks with pruning-aware versions.

    Composition order: LoRA -> MaskedPruning -> VACE -> CausalWanModel (base)
    """

    def __init__(self, causal_wan_model):
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
        if hasattr(causal_wan_model, "config") and hasattr(
            causal_wan_model.config, "sink_size"
        ):
            self.sink_size = causal_wan_model.config.sink_size
        else:
            self.sink_size = getattr(causal_wan_model, "sink_size", 0)

        # Get the original block class
        self._original_block_class = type(causal_wan_model.blocks[0])

        # Create factory-generated pruned block class
        self._PrunedBlock = create_pruned_block_class(self._original_block_class)

        # Replace blocks with pruning-aware versions
        self._replace_blocks_with_pruning_support()

        # Cache block forward signature for dynamic parameter filtering
        self._block_forward_params = self._get_block_forward_params()

    def _get_block_init_kwargs(self):
        """Get initialization kwargs for creating new blocks."""
        cross_attn_type = (
            "t2v_cross_attn" if self.model_type == "t2v" else "i2v_cross_attn"
        )

        kwargs = {
            "cross_attn_type": cross_attn_type,
            "dim": self.dim,
            "ffn_dim": self.ffn_dim,
            "num_heads": self.num_heads,
            "qk_norm": self.qk_norm,
            "cross_attn_norm": self.cross_attn_norm,
            "eps": self.eps,
        }

        sig = inspect.signature(self._original_block_class.__init__)
        params = sig.parameters

        if "local_attn_size" in params:
            kwargs["local_attn_size"] = self.local_attn_size
        if "sink_size" in params:
            kwargs["sink_size"] = self.sink_size

        return kwargs

    def _get_block_forward_params(self):
        """Get the set of parameter names accepted by the block's forward method."""
        sig = inspect.signature(self._original_block_class.forward)

        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if has_var_keyword:
            return None

        return set(sig.parameters.keys())

    def _filter_block_kwargs(self, block_kwargs, block_index):
        """Filter and prepare kwargs for a specific block."""
        if not block_kwargs:
            return {}

        filtered = {}
        for key, value in block_kwargs.items():
            if (
                self._block_forward_params is not None
                and key not in self._block_forward_params
            ):
                continue

            if isinstance(value, list | tuple) and len(value) == self.num_layers:
                filtered[key] = value[block_index]
            else:
                filtered[key] = value

        return filtered

    def _replace_blocks_with_pruning_support(self):
        """Replace blocks with pruning-aware versions.

        Memory-optimized: replaces blocks one at a time.
        """
        original_blocks = self.causal_wan_model.blocks

        orig_dtype = next(original_blocks[0].parameters()).dtype
        orig_device = next(original_blocks[0].parameters()).device

        block_kwargs = self._get_block_init_kwargs()

        new_blocks = nn.ModuleList()

        for i in range(self.num_layers):
            orig_block = original_blocks[i]

            with torch.device("cpu"):
                new_block = self._PrunedBlock(**block_kwargs)

            orig_state = orig_block.state_dict()
            new_state = new_block.state_dict()

            for key in orig_state.keys():
                if key in new_state:
                    new_state[key] = orig_state[key].to("cpu")

            new_block.load_state_dict(new_state, strict=False, assign=True)

            orig_block.to("cpu")
            new_block = new_block.to(device=orig_device, dtype=orig_dtype)
            new_block.eval()

            new_blocks.append(new_block)

            if i % 10 == 0:
                torch.cuda.empty_cache()

        torch.cuda.empty_cache()

        self.causal_wan_model.blocks = new_blocks
        self.blocks = new_blocks

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        prune_mask=None,
        vace_context=None,
        vace_context_scale=1.0,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        **block_kwargs,
    ):
        """Forward pass with optional masked pruning."""
        from ..utils import nan_fill_pruned, prune_tokens

        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None

        device = self.causal_wan_model.patch_embedding.weight.device
        if self.causal_wan_model.freqs.device != device:
            self.causal_wan_model.freqs = self.causal_wan_model.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y, strict=False)]

        # Patch embedding
        x = [self.causal_wan_model.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)

        # Prune tokens after patch embedding
        prune_active = prune_mask is not None
        if prune_active:
            f = grid_sizes[0][0].item()
            x = prune_tokens(x, prune_mask, f)

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

        # Generate VACE hints if the wrapped model supports it
        hints = None
        if vace_context is not None and hasattr(self.causal_wan_model, "forward_vace"):
            hints = self.causal_wan_model.forward_vace(
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

            # If pruning, also prune each hint
            if prune_active and hints is not None:
                pruned_hints = []
                for hint in hints:
                    pruned_hints.append(prune_tokens(hint, prune_mask, f))
                hints = pruned_hints

        # Base arguments for transformer blocks
        base_kwargs = {
            "e": e0,
            "seq_lens": seq_lens,
            "grid_sizes": grid_sizes,
            "freqs": self.causal_wan_model.freqs,
            "context": context,
            "context_lens": context_lens,
            "block_mask": self.causal_wan_model.block_mask,
            "prune_mask": prune_mask,
        }

        # Add VACE hints if present
        if hints is not None:
            base_kwargs["hints"] = hints
            base_kwargs["context_scale"] = vace_context_scale

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)

            return custom_forward

        # Process through blocks
        cache_update_infos = []
        for block_index, block in enumerate(self.blocks):
            filtered_block_kwargs = self._filter_block_kwargs(block_kwargs, block_index)
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

        # Expand pruned tokens back to full resolution with NaN fill
        if prune_active:
            f = grid_sizes[0][0].item()
            x = nan_fill_pruned(x, prune_mask, f)

        # Head + unpatchify (delegated to wrapped model's layers)
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
