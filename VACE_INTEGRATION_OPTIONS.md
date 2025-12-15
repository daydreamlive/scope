# VACE Integration Options for Wan Pipelines

## Context

VACE POC is currently tightly integrated with LongLive. Key architectural elements:

**Current VACE Implementation:**
- `CausalVaceWanModel` extends `CausalWanModel`
- Adds `vace_blocks` (16 parallel blocks for hint generation) and `vace_patch_embedding` (Conv3D encoder)
- Replaces ALL `CausalWanAttentionBlock` with `BaseWanAttentionBlock` (adds hint injection)
- Weight loading order: Base → VACE → LoRA (critical for PEFT compatibility)
- LoRAs target attention blocks via `ModuleTargetedLoRAStrategy` (all Linear layers in attention blocks)

**Other Wan Pipelines:**
- RewardForcing, KreaRealtimeVideo, StreamDiffusionV2 all use `CausalWanModel`
- All use `LoRAEnabledPipeline` mixin
- All use modular blocks architecture (sequential pipeline stages)

**Critical LoRA Insight:**
- VACE weights MUST load before LoRA wrapping to avoid PEFT unwrapping complexity
- `BaseWanAttentionBlock` extends `CausalWanAttentionBlock` → LoRA targeting works unchanged
- LongLive has built-in performance LoRA loaded separately from user LoRAs

**What VACE Actually Is:**
- An architectural variant of CausalWanModel (not a behavior that changes at runtime)
- Fixed at model initialization time (not dynamic)
- Single implementation (no multiple "strategies" for applying VACE)

---

## Option D: Composition-Based Wrapper (Single VACE Model)

**Architecture:**
Create a single `CausalVaceWanModel` that wraps ANY `CausalWanModel` implementation using composition, rather than creating per-pipeline VACE variants.

**Core Insight:**
VACE logic is identical regardless of base model. The wrapper receives an already-initialized `CausalWanModel` from any pipeline and adds VACE capability by:
1. Replacing blocks with hint-injection-capable blocks
2. Adding VACE-specific components (vace_blocks, vace_patch_embedding)
3. Wrapping forward pass to inject hints

**Implementation:**
```python
# In src/scope/core/pipelines/shared/vace/causal_vace_model.py
class CausalVaceWanModel(torch.nn.Module):
    """
    Single VACE wrapper that works with any CausalWanModel implementation.

    Uses composition to add VACE conditioning capability to any Wan-based model
    without requiring per-pipeline VACE implementations.
    """

    def __init__(
        self,
        causal_wan_model,
        vace_in_dim=96,
        vace_layers=None,
    ):
        """
        Wrap a CausalWanModel with VACE conditioning support.

        Args:
            causal_wan_model: Any CausalWanModel instance (from any pipeline)
            vace_in_dim: Input channels for VACE context (96 for R2V, 16 for depth)
            vace_layers: Layers to inject hints at (default: every 2nd layer)
        """
        super().__init__()

        # Store wrapped model
        self.causal_wan_model = causal_wan_model

        # Extract configuration from wrapped model
        self.num_layers = causal_wan_model.num_layers
        self.dim = causal_wan_model.dim
        self.ffn_dim = causal_wan_model.ffn_dim
        self.num_heads = causal_wan_model.num_heads
        self.local_attn_size = getattr(causal_wan_model, 'local_attn_size', -1)
        self.sink_size = getattr(causal_wan_model, 'sink_size', 0)
        self.qk_norm = causal_wan_model.qk_norm
        self.cross_attn_norm = causal_wan_model.cross_attn_norm
        self.eps = causal_wan_model.eps
        self.model_type = causal_wan_model.model_type
        self.patch_size = causal_wan_model.patch_size

        # VACE configuration
        self.vace_layers = (
            [i for i in range(0, self.num_layers, 2)]
            if vace_layers is None
            else vace_layers
        )
        self.vace_in_dim = vace_in_dim

        assert 0 in self.vace_layers
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # Replace wrapped model's blocks to support hint injection
        # This is the same operation the current inheritance-based approach does,
        # just applied to a passed-in model instead of self after super().__init__()
        self._replace_blocks_with_hint_injection_support()

        # Add VACE-specific components
        cross_attn_type = "t2v_cross_attn" if self.model_type == "t2v" else "i2v_cross_attn"

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
            self.vace_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size
        )

    def _replace_blocks_with_hint_injection_support(self):
        """
        Replace wrapped model's blocks with BaseWanAttentionBlock to support hint injection.

        This is the same operation the current inheritance-based CausalVaceWanModel does
        (creates blocks via super().__init__() then immediately replaces them), just made
        explicit as a post-initialization modification of the wrapped model.
        """
        cross_attn_type = "t2v_cross_attn" if self.model_type == "t2v" else "i2v_cross_attn"

        self.causal_wan_model.blocks = nn.ModuleList(
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
                    block_id=self.vace_layers_mapping[i] if i in self.vace_layers else None,
                )
                for i in range(self.num_layers)
            ]
        )

    def forward_vace(self, x, vace_context, seq_len, e, seq_lens, grid_sizes,
                     freqs, context, context_lens, block_mask, crossattn_cache):
        """Process VACE context to generate hints (same as current implementation)."""
        # ... existing forward_vace implementation ...
        pass

    def forward(self, *args, vace_context=None, vace_context_scale=1.0,
                vace_regenerate_hints=True, **kwargs):
        """
        Forward pass with optional VACE conditioning.

        Delegates to wrapped model, adding VACE parameters if context provided.
        """
        # Add VACE parameters to kwargs if VACE context provided
        if vace_context is not None:
            kwargs['vace_context'] = vace_context
            kwargs['vace_context_scale'] = vace_context_scale
            kwargs['vace_regenerate_hints'] = vace_regenerate_hints

        # Delegate to wrapped model (which now has hint-injection-capable blocks)
        return self.causal_wan_model(*args, **kwargs)

    def __getattr__(self, name):
        """Proxy attribute access to wrapped model for compatibility."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.causal_wan_model, name)
```

**Pipeline Integration:**
```python
# Any pipeline (LongLive, RewardForcing, StreamDiffusionV2, KreaRealtimeVideo)
from .modules.causal_model import CausalWanModel
from scope.core.pipelines.shared.vace import CausalVaceWanModel

# Load base model (pipeline-specific CausalWanModel)
base_model = CausalWanModel(
    **filter_causal_model_cls_config(CausalWanModel, config)
)

# Wrap with VACE if configured
vace_path = getattr(config, "vace_path", None)
if vace_path is not None:
    vace_in_dim = getattr(config, "vace_in_dim", 96)
    model = CausalVaceWanModel(base_model, vace_in_dim=vace_in_dim)

    # Load VACE weights
    from scope.core.pipelines.shared.vace import load_vace_weights_only
    load_vace_weights_only(model, vace_path)
else:
    model = base_model

# Wrap in WanDiffusionWrapper
generator = WanDiffusionWrapper(model, ...)

# Apply LoRAs (same as before)
generator.model = self._init_loras(config, generator.model)
```
