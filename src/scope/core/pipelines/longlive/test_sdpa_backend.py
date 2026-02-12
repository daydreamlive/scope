"""Quick test to check which SDPA backend gets used for VACE-like shapes."""

import torch
import torch.nn.functional as F

# VACE shapes: B=1, heads=12, seq_len=3072, head_dim=128
# (1.3B model: dim=1536, num_heads=12, head_dim=128)
b, n, s, d = 1, 12, 3072, 128
dtype = torch.bfloat16

q = torch.randn(b, n, s, d, device="cuda", dtype=dtype)
k = torch.randn(b, n, s, d, device="cuda", dtype=dtype)
v = torch.randn(b, n, s, d, device="cuda", dtype=dtype)

# Check which backends are available
print(f"Flash attention available: {torch.backends.cuda.flash_sdp_enabled()}")
print(
    f"Mem-efficient attention available: {torch.backends.cuda.mem_efficient_sdp_enabled()}"
)
print(f"Math attention available: {torch.backends.cuda.math_sdp_enabled()}")

# Test with context manager to see which one actually runs
with torch.backends.cuda.sdp_kernel(
    enable_flash=True, enable_math=False, enable_mem_efficient=False
):
    try:
        out = F.scaled_dot_product_attention(q, k, v)
        print("Flash SDP: works")
    except Exception as e:
        print(f"Flash SDP: FAILED - {e}")

with torch.backends.cuda.sdp_kernel(
    enable_flash=False, enable_math=False, enable_mem_efficient=True
):
    try:
        out = F.scaled_dot_product_attention(q, k, v)
        print("Mem-efficient SDP: works")
    except Exception as e:
        print(f"Mem-efficient SDP: FAILED - {e}")

# Benchmark all three

torch.cuda.synchronize()
# Warmup
for _ in range(10):
    F.scaled_dot_product_attention(q, k, v)
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

runs = 50
start.record()
for _ in range(runs):
    F.scaled_dot_product_attention(q, k, v)
end.record()
torch.cuda.synchronize()
print(f"\nSDPA (auto): {start.elapsed_time(end) / runs:.3f}ms per call")

# Compare with flash-only
try:
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_math=False, enable_mem_efficient=False
    ):
        for _ in range(10):
            F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        start.record()
        for _ in range(runs):
            F.scaled_dot_product_attention(q, k, v)
        end.record()
        torch.cuda.synchronize()
        print(f"SDPA (flash-only): {start.elapsed_time(end) / runs:.3f}ms per call")
except Exception:
    print("Flash-only: not available for this config")

# Compare with mem-efficient only
try:
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_math=False, enable_mem_efficient=True
    ):
        for _ in range(10):
            F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        start.record()
        for _ in range(runs):
            F.scaled_dot_product_attention(q, k, v)
        end.record()
        torch.cuda.synchronize()
        print(f"SDPA (mem-efficient): {start.elapsed_time(end) / runs:.3f}ms per call")
except Exception:
    print("Mem-efficient: not available for this config")
