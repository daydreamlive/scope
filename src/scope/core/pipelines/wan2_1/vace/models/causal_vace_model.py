# Modified from notes/VACE/vace/models/wan/modules/model.py
# Adapted for causal/autoregressive generation with composition pattern
import torch
import torch.nn as nn

from ....longlive.modules.model import sinusoidal_embedding_1d
from .attention_blocks import BaseWanAttentionBlock, VaceWanAttentionBlock


class CausalVaceWanModel(nn.Module):
    """
    VACE wrapper that adds reference image conditioning to any CausalWanModel.
    Uses composition to wrap an existing CausalWanModel instance.
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
        self.local_attn_size = getattr(causal_wan_model, "local_attn_size", -1)
        if hasattr(causal_wan_model, "config") and hasattr(
            causal_wan_model.config, "sink_size"
        ):
            self.sink_size = causal_wan_model.config.sink_size
        else:
            self.sink_size = getattr(causal_wan_model, "sink_size", 0)
        self.qk_norm = causal_wan_model.qk_norm
        self.cross_attn_norm = causal_wan_model.cross_attn_norm
        self.eps = causal_wan_model.eps
        self.model_type = causal_wan_model.model_type
        self.patch_size = causal_wan_model.patch_size
        self.in_dim = causal_wan_model.in_dim

        # VACE configuration
        self.vace_layers = (
            list(range(0, self.num_layers, 2)) if vace_layers is None else vace_layers
        )
        self.vace_in_dim = vace_in_dim

        assert 0 in self.vace_layers
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # Replace wrapped model's blocks with BaseWanAttentionBlock to support hint injection
        self._replace_blocks_with_hint_injection_support()

        # Add VACE-specific components
        cross_attn_type = (
            "t2v_cross_attn" if self.model_type == "t2v" else "i2v_cross_attn"
        )

        # VACE blocks (parallel processing path for reference images)
        self.vace_blocks = nn.ModuleList(
            [
                VaceWanAttentionBlock(
                    cross_attn_type,
                    self.dim,
                    self.ffn_dim,
                    self.num_heads,
                    self.local_attn_size,
                    self.sink_size,
                    self.qk_norm,
                    self.cross_attn_norm,
                    self.eps,
                    block_id=i,
                )
                for i in range(len(self.vace_layers))
            ]
        )

        # VACE patch embedding (separate encoder for reference images)
        self.vace_patch_embedding = nn.Conv3d(
            self.vace_in_dim,
            self.dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def _replace_blocks_with_hint_injection_support(self):
        """Replace wrapped model's blocks with BaseWanAttentionBlock to support hint injection."""
        cross_attn_type = (
            "t2v_cross_attn" if self.model_type == "t2v" else "i2v_cross_attn"
        )

        original_blocks = self.causal_wan_model.blocks

        # Create new blocks with hint injection support
        new_blocks = nn.ModuleList(
            [
                BaseWanAttentionBlock(
                    cross_attn_type,
                    self.dim,
                    self.ffn_dim,
                    self.num_heads,
                    self.local_attn_size,
                    self.sink_size,
                    self.qk_norm,
                    self.cross_attn_norm,
                    self.eps,
                    block_id=self.vace_layers_mapping[i]
                    if i in self.vace_layers
                    else None,
                )
                for i in range(self.num_layers)
            ]
        )

        # Set new blocks to eval mode
        new_blocks.eval()

        # Move new blocks to same device/dtype as original blocks
        orig_dtype = next(original_blocks[0].parameters()).dtype
        orig_device = next(original_blocks[0].parameters()).device
        new_blocks.to(device=orig_device, dtype=orig_dtype)

        # Copy weights from original blocks to new blocks
        for _i, (orig_block, new_block) in enumerate(
            zip(original_blocks, new_blocks, strict=False)
        ):
            orig_state = orig_block.state_dict()
            new_state = new_block.state_dict()
            saved_block_id = new_block.block_id

            for key in orig_state.keys():
                if key in new_state:
                    new_state[key] = orig_state[key].clone()

            new_block.load_state_dict(new_state, strict=False, assign=True)
            new_block.block_id = saved_block_id

        # Replace the blocks in wrapped model
        self.causal_wan_model.blocks = new_blocks

        # CRITICAL: Also register blocks directly on self so that named_modules()
        # sees them at "blocks.0", "blocks.1", etc. This ensures LoRA checkpoint
        # keys (saved with inheritance-style paths like "blocks.0.self_attn.q")
        # match correctly when loaded via peft.set_peft_model_state_dict().
        self.blocks = new_blocks

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
        # Embed VACE context
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
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
        vace_regenerate_hints=True,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=0,
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
        if vace_context is not None and vace_regenerate_hints:
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
            if torch.is_grad_enabled() and self.causal_wan_model.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                    }
                )
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
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                    }
                )
                result = block(x, **kwargs)
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
