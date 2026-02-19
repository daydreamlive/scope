import torch

TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    pass


if TRITON_AVAILABLE:

    @triton.jit
    def _extract_heads_kernel(
        roped_query_ptr,
        roped_key_ptr,
        v_ptr,
        q1_ptr,
        k1_ptr,
        v1_ptr,
        q2_ptr,
        k2_ptr,
        v2_ptr,
        headgroup_last_ptr,
        headgroup_first_mid_ptr,
        B,
        L,
        num_heads,
        C,
        num_last,
        num_first_mid,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b_idx = pid // (L * num_heads)
        remainder = pid % (L * num_heads)
        l_idx = remainder // num_heads
        h_idx = remainder % num_heads

        if b_idx >= B or l_idx >= L or h_idx >= num_heads:
            return

        is_last = 0
        out_h_last = 0
        for i in range(num_last):
            head_id = tl.load(headgroup_last_ptr + i)
            if h_idx == head_id:
                is_last = 1
                out_h_last = i

        is_first_mid = 0
        out_h_first_mid = 0
        for i in range(num_first_mid):
            head_id = tl.load(headgroup_first_mid_ptr + i)
            if h_idx == head_id:
                is_first_mid = 1
                out_h_first_mid = i

        base_offset = b_idx * L * num_heads * C + l_idx * num_heads * C + h_idx * C
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < C

        if is_last == 1:
            q_data = tl.load(roped_query_ptr + base_offset + offsets, mask=mask)
            k_data = tl.load(roped_key_ptr + base_offset + offsets, mask=mask)
            v_data = tl.load(v_ptr + base_offset + offsets, mask=mask)
            out_offset = (
                b_idx * L * num_last * C + l_idx * num_last * C + out_h_last * C
            )
            tl.store(q1_ptr + out_offset + offsets, q_data, mask=mask)
            tl.store(k1_ptr + out_offset + offsets, k_data, mask=mask)
            tl.store(v1_ptr + out_offset + offsets, v_data, mask=mask)

        if is_first_mid == 1:
            q_data = tl.load(roped_query_ptr + base_offset + offsets, mask=mask)
            k_data = tl.load(roped_key_ptr + base_offset + offsets, mask=mask)
            v_data = tl.load(v_ptr + base_offset + offsets, mask=mask)
            out_offset = (
                b_idx * L * num_first_mid * C
                + l_idx * num_first_mid * C
                + out_h_first_mid * C
            )
            tl.store(q2_ptr + out_offset + offsets, q_data, mask=mask)
            tl.store(k2_ptr + out_offset + offsets, k_data, mask=mask)
            tl.store(v2_ptr + out_offset + offsets, v_data, mask=mask)

    def _extract_heads_triton(
        roped_query, roped_key, v, headgroup_mid, headgroup_sink_dummy
    ):
        B, L, num_heads, C = roped_query.shape
        num_sd = len(headgroup_sink_dummy)
        num_mid = len(headgroup_mid)

        q1 = torch.empty(
            B, L, num_sd, C, device=roped_query.device, dtype=roped_query.dtype
        )
        k1 = torch.empty(
            B, L, num_sd, C, device=roped_key.device, dtype=roped_key.dtype
        )
        v1 = torch.empty(B, L, num_sd, C, device=v.device, dtype=v.dtype)
        q2 = torch.empty(
            B, L, num_mid, C, device=roped_query.device, dtype=roped_query.dtype
        )
        k2 = torch.empty(
            B, L, num_mid, C, device=roped_key.device, dtype=roped_key.dtype
        )
        v2 = torch.empty(B, L, num_mid, C, device=v.device, dtype=v.dtype)

        sd_tensor = torch.tensor(
            headgroup_sink_dummy, device=roped_query.device, dtype=torch.int32
        )
        mid_tensor = torch.tensor(
            headgroup_mid, device=roped_query.device, dtype=torch.int32
        )

        grid = (B * L * num_heads,)
        _extract_heads_kernel[grid](
            roped_query,
            roped_key,
            v,
            q1,
            k1,
            v1,
            q2,
            k2,
            v2,
            sd_tensor,
            mid_tensor,
            B,
            L,
            num_heads,
            C,
            num_sd,
            num_mid,
            BLOCK_SIZE=128,
        )
        return q1, k1, v1, q2, k2, v2


def _extract_heads_torch(
    roped_query, roped_key, v, headgroup_mid, headgroup_sink_dummy
):
    q1 = roped_query[:, :, headgroup_sink_dummy, :]
    k1 = roped_key[:, :, headgroup_sink_dummy, :]
    v1 = v[:, :, headgroup_sink_dummy, :]
    q2 = roped_query[:, :, headgroup_mid, :]
    k2 = roped_key[:, :, headgroup_mid, :]
    v2 = v[:, :, headgroup_mid, :]
    return q1, k1, v1, q2, k2, v2


def extract_heads(roped_query, roped_key, v, headgroup_mid, headgroup_sink_dummy):
    if TRITON_AVAILABLE and roped_query.is_cuda:
        return _extract_heads_triton(
            roped_query, roped_key, v, headgroup_mid, headgroup_sink_dummy
        )
    return _extract_heads_torch(
        roped_query, roped_key, v, headgroup_mid, headgroup_sink_dummy
    )
