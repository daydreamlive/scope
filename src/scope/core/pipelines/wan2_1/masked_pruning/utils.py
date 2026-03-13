import torch

# Number of recent cache frames used for M-Degree Approximation attention.
# Per arXiv:2603.05811 Section 3.2, using m recent KV entries exploits
# RoPE temporal alignment to approximate full softmax.
M_DEGREE = 6


def prune_tokens(
    x: torch.Tensor, spatial_mask: torch.Tensor, num_frames: int
) -> torch.Tensor:
    """Remove pruned spatial positions from token sequence.

    Args:
        x: [B, F*H'*W', dim]
        spatial_mask: [H'*W'] boolean, True=keep
        num_frames: number of frames F

    Returns:
        pruned_x: [B, F*kept, dim]
    """
    full_mask = spatial_mask.unsqueeze(0).expand(num_frames, -1).flatten()  # [F*H'*W']
    return x[:, full_mask]


def nan_fill_pruned(
    pruned_x: torch.Tensor,
    spatial_mask: torch.Tensor,
    num_frames: int,
) -> torch.Tensor:
    """Expand pruned token sequence back to full resolution, filling gaps with NaN.

    Args:
        pruned_x: [B, F*kept, dim]
        spatial_mask: [H'*W'] boolean, True=keep
        num_frames: number of frames F

    Returns:
        full_x: [B, F*H'*W', dim] with NaN at pruned positions
    """
    full_mask = spatial_mask.unsqueeze(0).expand(num_frames, -1).flatten()
    full_len = full_mask.shape[0]
    output = torch.full(
        (pruned_x.shape[0], full_len, pruned_x.shape[-1]),
        float("nan"),
        device=pruned_x.device,
        dtype=pruned_x.dtype,
    )
    output[:, full_mask] = pruned_x
    return output


def restore_latent_space(
    denoised_pred: torch.Tensor,
    source_latent: torch.Tensor,
    prune_mask: torch.Tensor,
    patch_size: int = 2,
) -> torch.Tensor:
    """Override pruned spatial positions in the model's x0 prediction with source latent.

    For V2V, unchanged regions should predict the original clean input. This corrects
    the feature-space restoration mismatch caused by mixing clean-pass (t=0) and
    denoise-pass (t>0) features through timestep-modulated head.

    Args:
        denoised_pred: [B, F, C, H, W] model x0 prediction
        source_latent: [B, F, C, H, W] clean VAE-encoded latent
        prune_mask: [H'*W'] boolean, True=keep (changed), False=pruned (unchanged)
        patch_size: spatial patch size used for mask computation

    Returns:
        corrected: [B, F, C, H, W] with pruned positions replaced by source_latent
    """
    B, F, C, H, W = denoised_pred.shape
    H_p, W_p = H // patch_size, W // patch_size

    # Reshape flat mask to 2D and expand to full latent resolution
    mask_2d = prune_mask.view(H_p, W_p)
    mask_full = mask_2d.repeat_interleave(patch_size, dim=0).repeat_interleave(
        patch_size, dim=1
    )
    # True=keep model output, False=use source latent
    mask_expanded = mask_full[None, None, None, :, :]
    return torch.where(mask_expanded, denoised_pred, source_latent)


def select_m_degree_kv(
    kv_cache: dict,
    m_degree: int,
    frame_seqlen: int,
    sink_tokens: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select m most recent frames of K/V from cache for M-Degree Approximation.

    Args:
        kv_cache: Cache dict with 'k', 'v', 'local_end_index'
        m_degree: Number of recent frames to include
        frame_seqlen: Tokens per frame at full spatial resolution
        sink_tokens: Number of sink tokens at start of cache

    Returns:
        (cache_k, cache_v): Selected K/V tensors from cache
    """
    local_end = kv_cache["local_end_index"].item()
    if local_end == 0:
        return kv_cache["k"][:, :0], kv_cache["v"][:, :0]

    m_tokens = m_degree * frame_seqlen

    if sink_tokens > 0 and sink_tokens < local_end:
        k_sink = kv_cache["k"][:, :sink_tokens]
        v_sink = kv_cache["v"][:, :sink_tokens]
        window_start = max(sink_tokens, local_end - m_tokens)
        return (
            torch.cat([k_sink, kv_cache["k"][:, window_start:local_end]], dim=1),
            torch.cat([v_sink, kv_cache["v"][:, window_start:local_end]], dim=1),
        )

    window_start = max(0, local_end - m_tokens)
    return (
        kv_cache["k"][:, window_start:local_end],
        kv_cache["v"][:, window_start:local_end],
    )


def causal_rope_apply_masked(x, grid_sizes, freqs, start_frame=0, spatial_mask=None):
    """Apply RoPE to tokens that may be spatially pruned, using correct positional encodings.

    When spatial_mask is provided, x contains only the kept tokens but positional
    encodings are computed for the full grid, then selected by the mask.

    Args:
        x: [B, pruned_seq_len, num_heads, head_dim]
        grid_sizes: [B, 3] with (F, H, W) at full resolution
        freqs: RoPE frequencies
        start_frame: starting frame index
        spatial_mask: [F*H*W] boolean, True=kept. If None, falls back to standard rope.
    """
    if spatial_mask is None:
        # Import the standard causal_rope_apply from the model module
        from scope.core.pipelines.longlive.modules.causal_model import causal_rope_apply

        return causal_rope_apply(x, grid_sizes, freqs, start_frame)

    n, c = x.size(2), x.size(3) // 2
    freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        full_seq_len = f * h * w
        kept_len = spatial_mask.sum().item()

        # Compute full-resolution freqs
        freqs_i = torch.cat(
            [
                freqs_split[0][start_frame : start_frame + f]
                .view(f, 1, 1, -1)
                .expand(f, h, w, -1),
                freqs_split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs_split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(full_seq_len, 1, -1)

        # Select only kept positions
        freqs_i = freqs_i[spatial_mask]

        x_i = torch.view_as_complex(
            x[i, :kept_len].to(torch.float64).reshape(kept_len, n, -1, 2)
        )
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, kept_len:]])
        output.append(x_i)

    return torch.stack(output).type_as(x)
