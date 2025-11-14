# LoRA Performance Analysis for WAN Models with FP8 Quantization

## Summary

Three LoRA merge strategies tested with WAN 14B + FP8 quantization. None are perfect.

| Strategy | Inference FPS | Update Time | Init Time | Production Ready? |
|----------|--------------|-------------|-----------|-------------------|
| permanent_merge | 9.15 | N/A (no updates) | ~70s | ✅ YES |
| runtime_peft | 4.34 | <1s | ~53s | ⚠️ DEV ONLY |
| gpu_reconstruct | 10.30 | ~60s | ~130s | ❌ NO |

## Strategy 1: permanent_merge (RECOMMENDED FOR PRODUCTION)

**Implementation:** Merges LoRA weights directly into model at load time: `W_final = W_base + scale * (lora_B @ lora_A)`

**Performance:**
- Inference: 9.15 FPS (baseline performance)
- Updates: Not supported (scale baked in at load)
- Init: ~70s (one-time merge)

**Why It's Fast:**
- Zero overhead per frame (LoRA is gone after merge)
- No additional memory or computation

**Limitation:**
- Cannot change LoRA scale at runtime
- To change scale: must reload entire pipeline (~70s)

**Use When:**
- Scale is predetermined
- Maximum FPS is critical
- Production deployment

## Strategy 2: runtime_peft (RECOMMENDED FOR DEVELOPMENT)

**Implementation:** Uses PEFT's LoraLayer to wrap nn.Linear modules. LoRA applied in forward pass: `output = base(x) + scale * lora(x)`

**Performance:**
- Inference: 4.34 FPS (52% slower than baseline)
- Updates: <1s (instant scale changes)
- Init: ~53s (PEFT wrapping overhead)

**Why It's Slow:**
- Every forward pass computes both base and LoRA paths
- Extra matmuls per layer: `lora_B @ (lora_A @ x)`
- FP8 quantization adds conversion overhead

**Why Updates Are Fast:**
- Scale is just a scalar multiplication
- No weight reconstruction needed

**Use When:**
- Tuning/experimenting with LoRA scales
- Interactive development
- FPS hit is acceptable

## Strategy 3: gpu_reconstruct (NOT RECOMMENDED)

**Implementation:** Stores original weights and LoRA diffs on GPU. Reconstructs on scale change: `W_new = W_original + scale * diff`

**Performance:**
- Inference: 10.30 FPS (best!)
- Updates: ~60s (terrible)
- Init: ~130s (stores copies)

**Why Inference Is Fast:**
- Zero per-frame overhead (weights are pre-merged)
- Same as permanent_merge after reconstruction

**Why Updates Are Slow (THE FUNDAMENTAL PROBLEM):**
1. FP8 tensors don't support arithmetic operations
2. Each weight update requires: FP8 → float32 → add → float32 → FP8
3. 400 weights × 150ms conversion overhead = 60s
4. Python loop forces CPU-GPU synchronization per weight
5. Cannot batch operations effectively

**Failed Optimization Attempts:**
- Storing in float32: WORSE (99s updates, cloning overhead)
- Batch operations: No improvement (still synchronizes per weight)
- In-place ops: Breaks FP8 quantization

**Root Cause:**
Cannot batch FP8 conversions without custom CUDA kernel. PyTorch's overhead is unavoidable.

**Use When:**
- Never. Use permanent_merge or runtime_peft instead.

## Recommendations

### Production: Use permanent_merge
- Best FPS (9.15)
- Predetermined LoRA scales
- Reload pipeline if scale needs to change

### Development: Use runtime_peft
- Instant scale updates (<1s)
- 52% FPS hit is acceptable for iteration
- Switch to permanent_merge for deployment

### Avoid: gpu_reconstruct
- 60s updates make it unusable for real-time
- Only marginally faster than permanent_merge (10.30 vs 9.15 FPS)
- Not worth the complexity and slow updates (for now)
