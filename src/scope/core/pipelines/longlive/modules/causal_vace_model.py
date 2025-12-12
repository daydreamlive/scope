# Modified from notes/VACE/vace/models/wan/modules/model.py
# Adapted for causal/autoregressive generation with Longlive
import torch
import torch.nn as nn
from diffusers.configuration_utils import register_to_config

from .causal_model import CausalWanModel, CausalWanAttentionBlock
from .model import sinusoidal_embedding_1d


class VaceWanAttentionBlock(CausalWanAttentionBlock):
    """VACE attention block with zero-initialized projection layers for hint injection."""

    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        block_id=0,
    ):
        super().__init__(
            cross_attn_type,
            dim,
            ffn_dim,
            num_heads,
            local_attn_size,
            sink_size,
            qk_norm,
            cross_attn_norm,
            eps,
        )
        self.block_id = block_id

        # Initialize projection layers for hint accumulation
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)

        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward_vace(
        self,
        c,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        crossattn_cache=None,
    ):
        """
        Forward pass for VACE blocks.

        Args:
            c: Accumulated VACE context from previous blocks (stacked hints + current)
            x: Input latent features
            Other args: Standard transformer block arguments

        Returns:
            Updated VACE context stack with new hint appended
        """
        # Unpack accumulated hints
        if self.block_id == 0:
            # c is padded to seq_len, but x may be shorter (unpadded for causal KV cache)
            # Slice c to match x's size for residual addition
            c_sliced = c[:, :x.size(1), :]

            print(f"forward_vace VaceBlock[{self.block_id}]: Mixing VACE context (shape={c_sliced.shape}) with current input x (shape={x.shape})")
            print(f"forward_vace VaceBlock[{self.block_id}]: This mixing causes reference image re-injection - should only happen on first chunk!")

            before_proj_out = self.before_proj(c_sliced)
            c = before_proj_out + x
            print(f"forward_vace VaceBlock[{self.block_id}]: After mixing, c.shape={c.shape}")

            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)

        # Run standard transformer block on current context
        # VACE blocks don't use caching since they process reference images once
        c = super().forward(
            c,
            e,
            seq_lens,
            grid_sizes,
            freqs,
            context,
            context_lens,
            block_mask,
            kv_cache=None,
            crossattn_cache=None,
            current_start=0,
            cache_start=None,
        )

        # Debug after transformer
        # c_nan = c.isnan().any().item()
        # c_min, c_max = c.min().item(), c.max().item()
        # print(f"VaceBlock[{self.block_id}] after transformer: nan={c_nan}, range=[{c_min:.2f},{c_max:.2f}]")

        # Generate hint for injection
        c_skip = self.after_proj(c)

        # Debug after_proj
        # cs_nan = c_skip.isnan().any().item()
        # cs_min, cs_max = c_skip.min().item(), c_skip.max().item()
        # print(f"VaceBlock[{self.block_id}] after_proj output: nan={cs_nan}, range=[{cs_min:.2f},{cs_max:.2f}]")

        all_c += [c_skip, c]

        # Stack and return
        return torch.stack(all_c)


class BaseWanAttentionBlock(CausalWanAttentionBlock):
    """Base attention block with VACE hint injection support."""

    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        block_id=None,
    ):
        super().__init__(
            cross_attn_type,
            dim,
            ffn_dim,
            num_heads,
            local_attn_size,
            sink_size,
            qk_norm,
            cross_attn_norm,
            eps,
        )
        self.block_id = block_id

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        hints=None,
        context_scale=1.0,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
    ):
        """
        Forward pass with optional VACE hint injection.

        Args:
            hints: List of VACE hints, one per injection layer
            context_scale: Scaling factor for hint injection
        """
        # Standard forward pass
        result = super().forward(
            x,
            e,
            seq_lens,
            grid_sizes,
            freqs,
            context,
            context_lens,
            block_mask,
            kv_cache,
            crossattn_cache,
            current_start,
            cache_start,
        )

        # Handle cache updates if present
        if kv_cache is not None and isinstance(result, tuple):
            x, cache_update_info = result
        else:
            x = result
            cache_update_info = None

        # Inject VACE hint if this block has one
        if hints is not None and self.block_id is not None:
            hint = hints[self.block_id]
            # Slice hint to match x's sequence length (x is unpadded, hint may be padded to seq_len)
            if hint.shape[1] > x.shape[1]:
                hint = hint[:, :x.shape[1], :]

            x = x + hint * context_scale

            # if not x_before_nan and x.isnan().any().item():
            #     print(f"VACEBlock[{self.block_id}]: WARNING - NaN introduced by hint injection!")

        # Return with cache info if applicable
        if cache_update_info is not None:
            return x, cache_update_info
        else:
            return x


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
            [i for i in range(0, self.num_layers, 2)]
            if vace_layers is None
            else vace_layers
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
                    block_id=self.vace_layers_mapping[i] if i in self.vace_layers else None,
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
            self.vace_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size
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
        print(f"forward_vace: Processing VACE context - x.shape={x.shape}, seq_len={seq_len}, num_contexts={len(vace_context)}")
        # Debug: Check input vace_context
        for i, vc in enumerate(vace_context):
            vc_nan = vc.isnan().any().item()
            vc_inf = vc.isinf().any().item()
            vc_min, vc_max = vc.min().item(), vc.max().item()
            print(f"forward_vace: vace_context[{i}] shape={vc.shape}, nan={vc_nan}, inf={vc_inf}, range=[{vc_min:.2f},{vc_max:.2f}]")

        # Embed VACE context
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]

        # Debug: Check after patch embedding
        # for i, emb in enumerate(c):
        #     emb_nan = emb.isnan().any().item()
        #     emb_inf = emb.isinf().any().item()
        #     emb_min, emb_max = emb.min().item(), emb.max().item()
        #     print(f"forward_vace: after patch_embed[{i}] shape={emb.shape}, nan={emb_nan}, inf={emb_inf}, range=[{emb_min:.2f},{emb_max:.2f}]")

        # Check patch embedding weights
        # patch_weight = self.vace_patch_embedding.weight
        # pw_nan = patch_weight.isnan().any().item()
        # pw_inf = patch_weight.isinf().any().item()
        # pw_min, pw_max = patch_weight.min().item(), patch_weight.max().item()
        # print(f"forward_vace: vace_patch_embedding.weight nan={pw_nan}, inf={pw_inf}, range=[{pw_min:.6f},{pw_max:.6f}]")

        c = [u.flatten(2).transpose(1, 2) for u in c]

        # Pad to seq_len (only if context is shorter; reference frames may exceed seq_len)
        c = torch.cat(
            [
                torch.cat([u, u.new_zeros(1, max(0, seq_len - u.size(1)), u.size(2))], dim=1)
                for u in c
            ]
        )

        # Debug: Check after padding
        # c_nan = c.isnan().any().item()
        # c_inf = c.isinf().any().item()
        # c_min, c_max = c.min().item(), c.max().item()
        # print(f"forward_vace: after padding shape={c.shape}, nan={c_nan}, inf={c_inf}, range=[{c_min:.2f},{c_max:.2f}]")

        # Process through VACE blocks
        for block_idx, block in enumerate(self.vace_blocks):
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
            # Debug: Check after each block
            # c_nan = c.isnan().any().item()
            # c_inf = c.isinf().any().item()
            # c_min, c_max = c.min().item(), c.max().item()
            # print(f"forward_vace: after vace_block[{block_idx}] shape={c.shape}, nan={c_nan}, inf={c_inf}, range=[{c_min:.2f},{c_max:.2f}]")

        # Extract hints (all but the last accumulated context)
        hints = torch.unbind(c)[:-1]
        print(f"forward_vace: Generated {len(hints)} hints for injection")
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
        vace_guidance_mode=None,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=0,
    ):
        """
        Forward pass with optional VACE conditioning.

        Args:
            vace_context: List of VAE-encoded reference images/frames or depth maps
            vace_context_scale: Scaling factor for VACE hint injection
            vace_guidance_mode: VACE guidance mode ('r2v' or 'depth')
                - r2v: Generate hints once (current_start==0), cache and reuse
                - depth: Generate hints every chunk, no caching
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

        # Generate VACE hints if vace_context provided
        # Behavior depends on guidance_mode:
        # - R2V: Generate hints once (current_start==0), cache and reuse
        # - Depth: Generate hints every chunk (no caching)
        hints = None
        if vace_context is not None:
            # Default to R2V mode if not specified
            if vace_guidance_mode is None:
                vace_guidance_mode = "r2v"

            if vace_guidance_mode == "r2v":
                # R2V mode: Only generate hints on first chunk
                if current_start == 0:
                    print(f"_forward_inference: Generating VACE hints for R2V mode (first chunk only, current_start={current_start}, seq_len={seq_len})")
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
                    print(f"_forward_inference: Generated {len(hints)} VACE hints for first chunk - subsequent chunks will not use VACE hints")
                else:
                    print(f"_forward_inference: Skipping VACE hint generation for R2V chunk (current_start={current_start}) - hints only applied to first chunk")
                    hints = None
            elif vace_guidance_mode == "depth":
                # Depth mode: Generate hints every chunk
                print(f"_forward_inference: Generating VACE hints for depth mode (chunk starting at frame {current_start}, seq_len={seq_len})")
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
                print(f"_forward_inference: Generated {len(hints)} VACE hints for depth chunk at frame {current_start}")
            else:
                raise ValueError(
                    f"CausalVaceWanModel._forward_inference: Unknown vace_guidance_mode '{vace_guidance_mode}', "
                    f"expected 'r2v' or 'depth'"
                )

            # Debug: Check if hints contain NaN
            # nan_status = [f"{i}:{'NaN' if hint.isnan().any().item() else 'OK'}" for i, hint in enumerate(hints)]
            # print(f"forward_vace: Generated {len(hints)} hints - {', '.join(nan_status)}")

        # Arguments for transformer blocks
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            hints=hints,
            context_scale=vace_context_scale,
        )

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
