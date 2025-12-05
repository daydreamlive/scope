# Modified from https://github.com/JaydenLu666/Reward-Forcing
# Implements EMA sink mechanism for real-time streaming
import copy
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

from scope.core.pipelines.wan2_1.modules.attention import attention
from scope.core.pipelines.longlive.modules.model import (
    WAN_CROSSATTENTION_CLASSES,
    MLPProj,
    WanLayerNorm,
    WanRMSNorm,
    rope_apply,
    rope_params,
    sinusoidal_embedding_1d,
)

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
)


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    """Apply causal rotary position embedding."""
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freqs[0][start_frame : start_frame + f]
                .view(f, 1, 1, -1)
                .expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class RewardForcingSelfAttention(nn.Module):
    """Self-attention with EMA sink mechanism for Reward-Forcing.

    Key difference from LongLive: when tokens are evicted from the KV cache,
    they are compressed into sink tokens using Exponential Moving Average (EMA)
    instead of being discarded.

    Args:
        dim: Hidden dimension
        num_heads: Number of attention heads
        local_attn_size: Local attention window size (-1 for global)
        sink_size: Number of sink frames to keep
        qk_norm: Whether to use QK normalization
        eps: Epsilon for layer norm
        compression_alpha: EMA coefficient for sink compression (default 0.999)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        eps: float = 1e-6,
        compression_alpha: float = 0.999,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.compression_alpha = compression_alpha
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def incremental_update(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        current_sink_k: torch.Tensor,
        current_sink_v: torch.Tensor,
        alpha: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """EMA update for sink tokens.

        When tokens are evicted from the local attention window, they are
        compressed into the sink tokens using exponential moving average.

        Args:
            new_k: Evicted key tokens [B, sink_tokens, num_heads, head_dim]
            new_v: Evicted value tokens [B, sink_tokens, num_heads, head_dim]
            current_sink_k: Current sink keys [B, sink_tokens, num_heads, head_dim]
            current_sink_v: Current sink values [B, sink_tokens, num_heads, head_dim]
            alpha: EMA coefficient (default: self.compression_alpha)

        Returns:
            Updated sink keys and values
        """
        if alpha is None:
            alpha = self.compression_alpha

        # EMA update: updated = α * current + (1-α) * new
        updated_sink_k = alpha * current_sink_k + (1 - alpha) * new_k
        updated_sink_v = alpha * current_sink_v + (1 - alpha) * new_v

        return updated_sink_k, updated_sink_v

    def forward(
        self,
        x: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        block_mask: BlockMask | None,
        kv_cache: dict | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
    ) -> torch.Tensor:
        """Forward pass with EMA sink mechanism.

        Args:
            x: Input tensor [B, L, C]
            seq_lens: Sequence lengths [B]
            grid_sizes: Grid sizes [B, 3] containing (F, H, W)
            freqs: RoPE frequencies [1024, C / num_heads / 2]
            block_mask: Attention mask for flex attention
            kv_cache: KV cache dict with 'k', 'v', 'global_end_index', 'local_end_index'
            current_start: Current start position in sequence
            cache_start: Cache start position (defaults to current_start)

        Returns:
            Output tensor [B, L, C]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # Query, key, value projection
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        current_start_frame = current_start // frame_seqlen
        current_end = current_start + q.shape[1]
        total_sink_tokens = self.sink_size * frame_seqlen

        # KV cache management
        kv_cache_size = kv_cache["k"].shape[1]
        num_new_tokens = q.shape[1]

        evicted_tokens_exist = False
        evicted_k = None
        evicted_v = None

        # Check if we need to evict tokens (cache overflow)
        if (
            self.local_attn_size != -1
            and (current_end > kv_cache["global_end_index"].item())
            and (num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size)
        ):
            # Calculate eviction parameters
            num_evicted_tokens = (
                num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
            )
            num_rolled_tokens = (
                kv_cache["local_end_index"].item()
                - num_evicted_tokens
                - total_sink_tokens
            )

            # Extract evicted tokens before rolling
            evicted_start = total_sink_tokens + num_rolled_tokens
            evicted_end = total_sink_tokens + num_rolled_tokens + num_evicted_tokens
            evicted_k = kv_cache["k"][:, evicted_start:evicted_end].clone()
            evicted_v = kv_cache["v"][:, evicted_start:evicted_end].clone()
            evicted_tokens_exist = True

            # Roll cache: shift non-sink, non-evicted tokens to the left
            kv_cache["k"][:, total_sink_tokens : total_sink_tokens + num_rolled_tokens] = (
                kv_cache["k"][
                    :,
                    total_sink_tokens + num_evicted_tokens : total_sink_tokens
                    + num_evicted_tokens
                    + num_rolled_tokens,
                ].clone()
            )
            kv_cache["v"][:, total_sink_tokens : total_sink_tokens + num_rolled_tokens] = (
                kv_cache["v"][
                    :,
                    total_sink_tokens + num_evicted_tokens : total_sink_tokens
                    + num_evicted_tokens
                    + num_rolled_tokens,
                ].clone()
            )

            # Update local indices
            local_end_index = (
                kv_cache["local_end_index"].item()
                + current_end
                - kv_cache["global_end_index"].item()
                - num_evicted_tokens
            )
            local_start_index = local_end_index - num_new_tokens

            # Insert new tokens
            kv_cache["k"][:, local_start_index:local_end_index] = k
            kv_cache["v"][:, local_start_index:local_end_index] = v
        else:
            # No eviction needed, just append
            local_end_index = (
                kv_cache["local_end_index"].item()
                + current_end
                - kv_cache["global_end_index"].item()
            )
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = k
            kv_cache["v"][:, local_start_index:local_end_index] = v

        # EMA sink compression: compress evicted tokens into sink tokens
        if evicted_tokens_exist and evicted_k is not None and evicted_k.shape[1] > 0:
            if evicted_k.shape[1] == total_sink_tokens:
                # Full frame eviction - apply EMA compression
                current_sink_k = kv_cache["k"][:, :total_sink_tokens]
                current_sink_v = kv_cache["v"][:, :total_sink_tokens]

                updated_sink_k, updated_sink_v = self.incremental_update(
                    evicted_k, evicted_v, current_sink_k, current_sink_v
                )

                kv_cache["k"][:, :total_sink_tokens] = updated_sink_k
                kv_cache["v"][:, :total_sink_tokens] = updated_sink_v
            else:
                # Partial eviction - directly copy (for initial frames)
                min_len = min(evicted_k.shape[1], total_sink_tokens)
                kv_cache["k"][:, :min_len] = evicted_k[:, :min_len]
                kv_cache["v"][:, :min_len] = evicted_v[:, :min_len]

        # Prepare key and value tensors for attention
        kv_start_index = max(
            total_sink_tokens, local_end_index - self.max_attention_size + total_sink_tokens
        )
        kv_end_index = local_end_index

        # Calculate the actual number of tokens to use
        actual_kv_tokens = kv_end_index - kv_start_index

        # Extract KV segments
        k_segment = kv_cache["k"][:, kv_start_index:kv_end_index]
        v_segment = kv_cache["v"][:, kv_start_index:kv_end_index]

        # Compute attention with sink tokens
        if self.sink_size > 0:
            sink_k = kv_cache["k"][:, :total_sink_tokens]
            sink_v = kv_cache["v"][:, :total_sink_tokens]

            k_combined = torch.cat([sink_k, k_segment], dim=1)
            v_combined = torch.cat([sink_v, v_segment], dim=1)

            # Update grid_sizes for KV
            grid_sizes_kv = copy.deepcopy(grid_sizes)
            total_combined_tokens = total_sink_tokens + actual_kv_tokens
            total_frames = total_combined_tokens // frame_seqlen
            grid_sizes_kv[:, 0] = total_frames

            # Calculate start frame for query RoPE
            query_start_frame = (
                total_sink_tokens
                + (
                    local_end_index
                    - max(
                        total_sink_tokens,
                        local_end_index - self.max_attention_size + total_sink_tokens,
                    )
                )
                - q.shape[1]
            ) // frame_seqlen

            x = attention(
                causal_rope_apply(q, grid_sizes, freqs, start_frame=query_start_frame).type_as(
                    v
                ),
                causal_rope_apply(k_combined, grid_sizes_kv, freqs, start_frame=0).type_as(v),
                v_combined,
            )
        else:
            # No sink tokens
            grid_sizes_kv = copy.deepcopy(grid_sizes)
            grid_sizes_kv[:, 0] = actual_kv_tokens // frame_seqlen

            query_start_frame = (
                local_end_index - self.max_attention_size - q.shape[1]
            ) // frame_seqlen

            x = attention(
                causal_rope_apply(q, grid_sizes, freqs, start_frame=query_start_frame).type_as(
                    v
                ),
                causal_rope_apply(k_segment, grid_sizes_kv, freqs, start_frame=0).type_as(v),
                v_segment,
            )

        # Update cache indices
        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end_index)

        # Output projection
        x = x.flatten(2)
        x = self.o(x)
        return x


class RewardForcingAttentionBlock(nn.Module):
    """Attention block with EMA sink for Reward-Forcing."""

    def __init__(
        self,
        cross_attn_type: str,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        compression_alpha: float = 0.999,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = RewardForcingSelfAttention(
            dim, num_heads, local_attn_size, sink_size, qk_norm, eps, compression_alpha
        )
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor | None,
        block_mask: BlockMask | None,
        kv_cache: dict | None = None,
        crossattn_cache: dict | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
    ) -> torch.Tensor:
        """Forward pass with EMA sink mechanism."""
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # Self-attention with EMA sink
        y = self.self_attn(
            (
                self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                * (1 + e[1])
                + e[0]
            ).flatten(1, 2),
            seq_lens,
            grid_sizes,
            freqs,
            block_mask,
            kv_cache,
            current_start,
            cache_start,
        )

        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # Cross-attention & FFN
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(
                self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache
            )
            y = self.ffn(
                (
                    self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                    * (1 + e[4])
                    + e[3]
                ).flatten(1, 2)
            )
            x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(
                1, 2
            )
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class RewardForcingHead(nn.Module):
    """Output head for Reward-Forcing model."""

    def __init__(self, dim: int, out_dim: int, patch_size: tuple, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim_total = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim_total)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = self.head(
            self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]
        )
        return x


class RewardForcingCausalModel(ModelMixin, ConfigMixin):
    """Causal diffusion model with EMA sink for Reward-Forcing.

    This model is based on Wan2.1-T2V-1.3B architecture with the following
    key additions for Reward-Forcing:

    1. EMA sink mechanism: When tokens are evicted from the local attention
       window, they are compressed into sink tokens using Exponential Moving
       Average instead of being discarded. This enables bounded memory usage
       while retaining long-term context.

    2. 4-step denoising: The model is trained to generate high-quality videos
       in just 4 denoising steps (1000, 750, 500, 250) through reward-based
       distillation.

    Reference: https://github.com/JaydenLu666/Reward-Forcing
    """

    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim"]
    _no_split_modules = ["RewardForcingAttentionBlock"]
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type: str = "t2v",
        patch_size: tuple = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 1536,  # Wan2.1-T2V-1.3B
        ffn_dim: int = 8960,  # Wan2.1-T2V-1.3B
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 12,  # Wan2.1-T2V-1.3B
        num_layers: int = 30,  # Wan2.1-T2V-1.3B
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        compression_alpha: float = 0.999,
    ):
        """Initialize Reward-Forcing causal model.

        Args:
            model_type: Model variant - 't2v' or 'i2v'
            patch_size: 3D patch dimensions (t_patch, h_patch, w_patch)
            text_len: Fixed length for text embeddings
            in_dim: Input video channels
            dim: Hidden dimension (1536 for 1.3B, 5120 for 14B)
            ffn_dim: FFN intermediate dimension (8960 for 1.3B, 13824 for 14B)
            freq_dim: Dimension for sinusoidal time embeddings
            text_dim: Input dimension for text embeddings
            out_dim: Output video channels
            num_heads: Number of attention heads (12 for 1.3B, 40 for 14B)
            num_layers: Number of transformer blocks (30 for 1.3B, 40 for 14B)
            local_attn_size: Window size for local attention (-1 for global)
            sink_size: Number of sink frames to keep
            qk_norm: Enable query/key normalization
            cross_attn_norm: Enable cross-attention normalization
            eps: Epsilon for normalization layers
            compression_alpha: EMA coefficient for sink compression (default 0.999)
        """
        super().__init__()

        assert model_type in ["t2v", "i2v"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.compression_alpha = compression_alpha

        # embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks with EMA sink
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                RewardForcingAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    local_attn_size,
                    sink_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    compression_alpha,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = RewardForcingHead(dim, out_dim, patch_size, eps)

        # buffers
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

        if model_type == "i2v":
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False
        self.block_mask = None
        self.block_mask_cache: dict[tuple, BlockMask] = {}
        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def _get_block_mask(
        self,
        device: torch.device | str,
        num_frames: int,
        frame_seqlen: int,
        num_frame_per_block: int,
        local_attn_size: int,
    ) -> BlockMask:
        """Get or create cached block mask."""
        cache_key = (num_frames, frame_seqlen, num_frame_per_block, local_attn_size)

        if cache_key not in self.block_mask_cache:
            if self.independent_first_frame:
                mask = self._prepare_blockwise_causal_attn_mask_i2v(
                    device, num_frames, frame_seqlen, num_frame_per_block, local_attn_size
                )
            else:
                mask = self._prepare_blockwise_causal_attn_mask(
                    device, num_frames, frame_seqlen, num_frame_per_block, local_attn_size
                )
            self.block_mask_cache[cache_key] = mask

        return self.block_mask_cache[cache_key]

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str,
        num_frames: int = 21,
        frame_seqlen: int = 1560,
        num_frame_per_block: int = 1,
        local_attn_size: int = -1,
    ) -> BlockMask:
        """Prepare blockwise causal attention mask."""
        total_length = num_frames * frame_seqlen
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device,
        )

        for tmp in frame_indices:
            ends[tmp : tmp + frame_seqlen * num_frame_per_block] = (
                tmp + frame_seqlen * num_frame_per_block
            )

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return (
                    (kv_idx < ends[q_idx])
                    & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))
                ) | (q_idx == kv_idx)

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str,
        num_frames: int = 21,
        frame_seqlen: int = 1560,
        num_frame_per_block: int = 4,
        local_attn_size: int = -1,
    ) -> BlockMask:
        """Prepare blockwise causal attention mask for I2V."""
        total_length = num_frames * frame_seqlen
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Special handling for first frame
        ends[:frame_seqlen] = frame_seqlen

        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device,
        )

        for tmp in frame_indices:
            ends[tmp : tmp + frame_seqlen * num_frame_per_block] = (
                tmp + frame_seqlen * num_frame_per_block
            )

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return (
                    (kv_idx < ends[q_idx])
                    & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))
                ) | (q_idx == kv_idx)

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )

        return block_mask

    def _forward_inference(
        self,
        x: list[torch.Tensor],
        t: torch.Tensor,
        context: list[torch.Tensor],
        seq_len: int,
        clip_fea: torch.Tensor | None = None,
        y: list[torch.Tensor] | None = None,
        kv_cache: list[dict] | None = None,
        crossattn_cache: list[dict] | None = None,
        current_start: int = 0,
        cache_start: int = 0,
    ) -> torch.Tensor:
        """Forward pass for inference with KV caching and EMA sink."""
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)

        # time embeddings
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        # context
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

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)

            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                    }
                )
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(self, *args, **kwargs):
        """Forward pass - routes to inference or training."""
        if kwargs.get("kv_cache", None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            # Training not implemented for now - use inference only
            raise NotImplementedError("Training forward not implemented for RewardForcingCausalModel")

    def unpatchify(self, x: torch.Tensor, grid_sizes: torch.Tensor) -> list[torch.Tensor]:
        """Reconstruct video tensors from patch embeddings."""
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        """Initialize model parameters using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
