# Modified from notes/VACE/vace/models/wan/modules/model.py
# Adapted for causal/autoregressive generation with Longlive
import torch
import torch.nn as nn
from diffusers.configuration_utils import register_to_config

from ....longlive.modules.causal_model import CausalWanModel
from ....longlive.modules.model import sinusoidal_embedding_1d
from .attention_blocks import BaseWanAttentionBlock, VaceWanAttentionBlock


class CausalVaceWanModel(CausalWanModel):
    """
    Causal Wan model with VACE support for reference image conditioning.

    This model extends CausalWanModel with VACE blocks that process reference images
    and generate hints for injection into the main transformer blocks.

    Key differences from standard VACE:
    - Causal processing: Reference images are processed with causal attention
    - KV cache compatible: Works with Longlive's incremental generation
    - Hint caching: VACE hints are computed once and reused across frames
    """

    @register_to_config
    def __init__(
        self,
        vace_layers=None,
        vace_in_dim=None,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    ):
        # Initialize base model
        super().__init__(
            model_type,
            patch_size,
            text_len,
            in_dim,
            dim,
            ffn_dim,
            freq_dim,
            text_dim,
            out_dim,
            num_heads,
            num_layers,
            local_attn_size,
            sink_size,
            qk_norm,
            cross_attn_norm,
            eps,
        )

        # VACE configuration
        self.vace_layers = (
            list(range(0, self.num_layers, 2)) if vace_layers is None else vace_layers
        )
        self.vace_in_dim = self.in_dim if vace_in_dim is None else vace_in_dim

        assert 0 in self.vace_layers
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # Replace standard blocks with BaseWanAttentionBlock (supports hint injection)
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                BaseWanAttentionBlock(
                    cross_attn_type,
                    self.dim,
                    self.ffn_dim,
                    self.num_heads,
                    self.local_attn_size,
                    sink_size,
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

        # VACE blocks (separate processing path)
        self.vace_blocks = nn.ModuleList(
            [
                VaceWanAttentionBlock(
                    cross_attn_type,
                    self.dim,
                    self.ffn_dim,
                    self.num_heads,
                    self.local_attn_size,
                    sink_size,
                    self.qk_norm,
                    self.cross_attn_norm,
                    self.eps,
                    block_id=i,
                )
                for i in range(len(self.vace_layers))
            ]
        )

        # VACE patch embedding (separate from main patch embedding)
        self.vace_patch_embedding = nn.Conv3d(
            self.vace_in_dim,
            self.dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

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
        """
        Process VACE context (reference images + frames) to generate hints.

        Args:
            x: Main latent input (used for shape reference in block 0)
            vace_context: List of VAE-encoded reference images/frames concatenated with masks
            seq_len: Maximum sequence length
            Other args: Standard transformer block arguments needed for VACE blocks

        Returns:
            List of hints to be injected at specified transformer layers
        """
        # Embed VACE context
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]

        c = [u.flatten(2).transpose(1, 2) for u in c]

        # Pad to seq_len (only if context is shorter; reference frames may exceed seq_len)
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

        # Extract hints (all but the last accumulated context)
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
        """
        Forward pass with optional VACE conditioning.

        Args:
            vace_context: List of VAE-encoded conditioning (reference images, depth, flow, pose, etc.)
            vace_context_scale: Scaling factor for VACE hint injection
            vace_regenerate_hints: Whether to regenerate hints for this chunk.
                - True: Generate fresh hints (for per-chunk conditioning like depth/flow)
                - False: Skip hint generation (for static reference images after first chunk)
        """
        # Standard preprocessing
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y, strict=False)]

        # Embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        # Don't pad x - causal model needs unpadded sequences for KV cache
        x = torch.cat(x)

        # Time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x)
        )
        e0 = (
            self.time_projection(e)
            .unflatten(1, (6, self.dim))
            .unflatten(dim=0, sizes=t.shape)
        )

        # Context
        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [
                    torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]
            )
        )

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        # Generate VACE hints if vace_context provided and regeneration is requested
        hints = None
        if vace_context is not None and vace_regenerate_hints:
            hints = self.forward_vace(
                x,
                vace_context,
                seq_len,
                e0,
                seq_lens,
                grid_sizes,
                self.freqs,
                context,
                context_lens,
                self.block_mask,
                crossattn_cache,
            )

        # Arguments for transformer blocks
        kwargs = {
            "e": e0,
            "seq_lens": seq_lens,
            "grid_sizes": grid_sizes,
            "freqs": self.freqs,
            "context": context,
            "context_lens": context_lens,
            "block_mask": self.block_mask,
            "hints": hints,
            "context_scale": vace_context_scale,
        }

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)

            return custom_forward

        cache_update_infos = []
        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
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

        # Apply cache updates
        if kv_cache is not None and cache_update_infos:
            self._apply_cache_updates(kv_cache, cache_update_infos)

        # Head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # Unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)
