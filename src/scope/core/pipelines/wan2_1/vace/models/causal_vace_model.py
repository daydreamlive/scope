# Modified from notes/VACE/vace/models/wan/modules/model.py
# Adapted for causal/autoregressive generation with factory pattern
# Pipeline-agnostic using duck typing - works with any CausalWanModel
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
        self._create_vace_blocks()

        # VACE patch embedding (separate encoder for reference images)
        self.vace_patch_embedding = nn.Conv3d(
            self.vace_in_dim,
            self.dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

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
        """
        original_blocks = self.causal_wan_model.blocks

        # Get device and dtype from original blocks
        orig_dtype = next(original_blocks[0].parameters()).dtype
        orig_device = next(original_blocks[0].parameters()).device

        # Get initialization kwargs
        block_kwargs = self._get_block_init_kwargs()

        # Create new blocks with hint injection support
        new_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block_id = self.vace_layers_mapping[i] if i in self.vace_layers else None
            new_block = self._BaseWanAttentionBlock(
                **block_kwargs,
                block_id=block_id,
            )
            new_blocks.append(new_block)

        # Set to eval mode and move to correct device/dtype
        new_blocks.eval()
        new_blocks.to(device=orig_device, dtype=orig_dtype)

        # Copy weights from original blocks
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

        # Replace blocks in wrapped model
        self.causal_wan_model.blocks = new_blocks

        # Also register blocks on self for LoRA compatibility
        self.blocks = new_blocks

    def _create_vace_blocks(self):
        """Create VACE blocks for parallel processing of reference images."""
        # Get device and dtype from existing blocks
        orig_dtype = next(self.blocks[0].parameters()).dtype
        orig_device = next(self.blocks[0].parameters()).device

        # Get initialization kwargs
        block_kwargs = self._get_block_init_kwargs()

        # Create VACE blocks
        vace_blocks = nn.ModuleList()
        for block_id in range(len(self.vace_layers)):
            vace_block = self._VaceWanAttentionBlock(
                **block_kwargs,
                block_id=block_id,
            )
            vace_blocks.append(vace_block)

        # Move to correct device/dtype
        vace_blocks.to(device=orig_device, dtype=orig_dtype)

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
        **block_kwargs,
    ):
        """
        Forward pass with optional VACE conditioning.

        Reference-to-Video (R2V) Sink Token Architecture:
        -------------------------------------------------
        On first chunk (current_start==0), reference images are injected as sink tokens
        that persist in KV cache across all subsequent chunks. This provides continuous
        reference appearance conditioning to combat temporal drift.

        Key design decisions:
        1. Reference spatial tokens preserved and tiled to fill one frame (frame_seqlen tokens)
        2. Reference "frame" prepended to sequence, integrated into frame structure (seq_lens updated)
        3. Hints inject only on video tokens, not reference prefix
        4. Reference tokens stripped before unpatchify (only video frames decoded)
        5. With sink_size mechanism, reference tokens never evicted from cache

        This combines two conditioning paths:
        - Persistent: Reference tokens in KV cache with spatial structure (always visible via attention)
        - Ephemeral: VACE hints (injected per denoising step, chunk-dependent)
        """
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

        # Reference frame injection as sink tokens (first chunk only)
        # This makes reference appearance persistently available in KV cache across all chunks
        reference_prefix_length = 0
        if current_start == 0 and vace_context is not None and kv_cache is not None:
            # Calculate frame_seqlen from video tokens to ensure proper frame alignment
            # e.g., for 3 frames with 1024 tokens/frame: frame_seqlen = 1024
            frame_seqlen = x.size(1) // t.shape[1]

            # Encode reference tokens with full spatial structure and tile to frame_seqlen
            # This preserves spatial information from reference images for better conditioning
            # vace_context is a list of tensors, each with shape [C, num_refs, H, W]
            ref_latents = []
            for ref_ctx in vace_context:
                # ref_ctx shape: [C, num_refs, H, W] where C=96 (vace_in_dim)

                # Use VACE patch embedding (trained for 96-channel VACE context)
                # This is critical - vace_patch_embedding was trained to extract features
                # from the full 96-channel reference context (VAE latent + depth + edges + etc)
                ref_embedded = self.vace_patch_embedding(ref_ctx.unsqueeze(0))
                # ref_embedded shape: [1, dim, num_refs, h, w]

                # Flatten spatial dimensions: [1, dim, num_refs, h, w] -> [1, dim, num_refs*h*w]
                ref_embedded = ref_embedded.flatten(2)

                # Transpose: [1, dim, num_tokens] -> [1, num_tokens, dim]
                ref_embedded = ref_embedded.transpose(1, 2)

                # Tile/repeat reference tokens to fill frame_seqlen
                # This preserves spatial structure while maintaining frame alignment
                num_ref_tokens = ref_embedded.size(1)
                if num_ref_tokens < frame_seqlen:
                    # Repeat tokens cyclically to fill frame_seqlen
                    num_repeats = (frame_seqlen + num_ref_tokens - 1) // num_ref_tokens
                    ref_tiled = ref_embedded.repeat(1, num_repeats, 1)[
                        :, :frame_seqlen, :
                    ]
                elif num_ref_tokens > frame_seqlen:
                    # Downsample if too many tokens (interpolate to preserve all spatial info)
                    ref_tiled = torch.nn.functional.interpolate(
                        ref_embedded.transpose(1, 2),
                        size=frame_seqlen,
                        mode="linear",
                        align_corners=False,
                    ).transpose(1, 2)
                else:
                    ref_tiled = ref_embedded

                ref_latents.append(ref_tiled)

            # Concatenate across batch: list of [1, frame_seqlen, dim] -> [batch_size, frame_seqlen, dim]
            ref_tokens = torch.cat(ref_latents, dim=0)
            reference_prefix_length = ref_tokens.size(1)  # Should equal frame_seqlen

            # Diagnostic: Check token distribution statistics
            logger.info(
                f"_forward_inference: Reference token statistics - "
                f"mean: {ref_tokens.mean().item():.4f}, "
                f"std: {ref_tokens.std().item():.4f}, "
                f"min: {ref_tokens.min().item():.4f}, "
                f"max: {ref_tokens.max().item():.4f}"
            )
            logger.info(
                f"_forward_inference: Video token statistics - "
                f"mean: {x.mean().item():.4f}, "
                f"std: {x.std().item():.4f}, "
                f"min: {x.min().item():.4f}, "
                f"max: {x.max().item():.4f}"
            )

            logger.info(
                f"_forward_inference: Injecting {reference_prefix_length} reference tokens as 1 reference frame "
                f"(batch_size={ref_tokens.size(0)}, dim={ref_tokens.size(2)}, frame_seqlen={frame_seqlen}, "
                f"spatial_structure_preserved=True)"
            )

            # Amplify reference signal to prevent it from being drowned out during denoising
            # Video tokens undergo large variance changes (std grows ~2.6x during transformer)
            # Scale reference tokens to maintain influence throughout denoising process
            reference_amplification = 2.0
            ref_tokens = ref_tokens * reference_amplification
            logger.info(
                f"_forward_inference: Amplified reference tokens by {reference_amplification}x to combat signal dilution"
            )

            # Prepend reference tokens to sequence as sink tokens (one reference "frame")
            # x: [batch_size, num_frames*frame_seqlen, dim], ref_tokens: [batch_size, frame_seqlen, dim]
            # Result: [batch_size, (num_frames+1)*frame_seqlen, dim]
            x = torch.cat([ref_tokens, x], dim=1)

            # Update seq_lens to reflect the additional reference "frame"
            # This maintains proper frame structure for unflatten operations
            seq_lens = seq_lens + reference_prefix_length

            # Save original grid_sizes for forward_vace (VACE context has different spatial dims)
            grid_sizes_original = grid_sizes.clone()

            # Update grid_sizes to reflect additional reference frame
            # grid_sizes shape: [B, 3] where columns are (F, H, W)
            # Increment frame count by 1 for the reference frame
            grid_sizes = grid_sizes.clone()
            grid_sizes[:, 0] += 1  # Add 1 to frame count

        # Time embeddings
        # Save original t for head operation later
        t_original = t

        # If reference frame was added, prepend a time embedding for it (timestep = 0)
        if reference_prefix_length > 0:
            # Create time embedding for reference frame (timestep = 0)
            ref_timestep = torch.zeros((t.shape[0], 1), dtype=t.dtype, device=t.device)
            # Concatenate: [ref_timestep, video_timesteps]
            t_with_ref = torch.cat([ref_timestep, t], dim=1)
        else:
            t_with_ref = t

        e = self.causal_wan_model.time_embedding(
            sinusoidal_embedding_1d(
                self.causal_wan_model.freq_dim, t_with_ref.flatten()
            ).type_as(x)
        )
        e0 = (
            self.causal_wan_model.time_projection(e)
            .unflatten(1, (6, self.dim))
            .unflatten(dim=0, sizes=t_with_ref.shape)
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
        # Hints are generated from VACE context, independent of reference token injection
        hints = None
        if vace_context is not None and vace_regenerate_hints:
            # Extract video tokens for forward_vace (skip reference prefix if present)
            x_for_vace = (
                x[:, reference_prefix_length:, :] if reference_prefix_length > 0 else x
            )

            hints = self.forward_vace(
                x_for_vace,
                vace_context,
                seq_len,
                e0,
                seq_lens - reference_prefix_length,  # Use original video seq_lens
                grid_sizes_original
                if reference_prefix_length > 0
                else grid_sizes,  # Use original grid_sizes
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
            "reference_prefix_length": reference_prefix_length,
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
                        **block_kwargs,
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
                        **block_kwargs,
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

        # Strip reference frame before output processing (only actual video frames should be decoded)
        if reference_prefix_length > 0:
            # Diagnostic: Check how tokens changed after transformer blocks
            ref_final = x[:, :reference_prefix_length, :]
            video_final = x[:, reference_prefix_length:, :]

            logger.info(
                f"_forward_inference: Reference tokens after transformer blocks - "
                f"mean: {ref_final.mean().item():.4f}, "
                f"std: {ref_final.std().item():.4f}, "
                f"min: {ref_final.min().item():.4f}, "
                f"max: {ref_final.max().item():.4f}"
            )
            logger.info(
                f"_forward_inference: Video tokens after transformer blocks - "
                f"mean: {video_final.mean().item():.4f}, "
                f"std: {video_final.std().item():.4f}, "
                f"min: {video_final.min().item():.4f}, "
                f"max: {video_final.max().item():.4f}"
            )

            logger.debug(
                f"_forward_inference: Stripping {reference_prefix_length} reference tokens from output "
                f"(x.shape before: {x.shape})"
            )
            x = x[:, reference_prefix_length:, :]

            # Also strip reference frame from grid_sizes for unpatchify
            # These operations expect only video frames
            grid_sizes_video = grid_sizes.clone()
            grid_sizes_video[:, 0] -= 1  # Remove reference frame from count

            # For head, we need e in original shape (before projection to e0)
            # Recompute e for video frames only using t_original
            e_video = self.causal_wan_model.time_embedding(
                sinusoidal_embedding_1d(
                    self.causal_wan_model.freq_dim, t_original.flatten()
                ).type_as(x)
            )
        else:
            grid_sizes_video = grid_sizes
            e_video = e

        x = self.causal_wan_model.head(
            x, e_video.unflatten(dim=0, sizes=t_original.shape).unsqueeze(2)
        )
        x = self.causal_wan_model.unpatchify(x, grid_sizes_video)
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
