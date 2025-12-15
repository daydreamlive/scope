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

## TL;DR - Recommended Approach

**Option D: Composition-Based Wrapper**

Create a single `CausalVaceWanModel` wrapper that accepts ANY `CausalWanModel` implementation from any pipeline:

```python
class CausalVaceWanModel(torch.nn.Module):
    def __init__(self, causal_wan_model, vace_in_dim=96, vace_layers=None):
        self.causal_wan_model = causal_wan_model  # Any pipeline's CausalWanModel
        # Replace blocks to support hint injection (same as current approach does)
        # Add VACE components (vace_blocks, vace_patch_embedding)
```

**Why:** Single VACE implementation for all pipelines, no refactoring required, composition over inheritance, automatically supports future pipelines with unique parameters.

**Avoids:** Per-pipeline VACE class duplication (Option A), extensive refactoring to merge different CausalWanModel implementations (Option B), mixin complexity that still requires per-pipeline VACE classes (Option C).

---

## Why Only 3 Options?

After critical analysis, several initially proposed options were eliminated:

- ❌ **Strategy Pattern** - LoRA needs strategies (permanent merge, runtime PEFT, module-targeted). VACE has ONE way to apply: add blocks, replace attention blocks, load weights. Strategy pattern is over-engineering.
- ❌ **Decorator Pattern** - Elegant in theory, but VACE is determined at initialization and never changes. We're not leveraging dynamic composition.
- ❌ **Injection System** - Runtime model surgery is fragile and overly complex.
- ❌ **Component Wrapper** - Redundant with decorator, less elegant.

**The Core Question:** Do we want ONE model class or TWO model classes?

---

## Option A: Two Separate Classes (Minimal Change from Current)

**Architecture:**
Keep `CausalWanModel` and `CausalVaceWanModel` as separate classes. Extend current LongLive pattern to other pipelines.

**Pipeline Integration:**
```python
# In each pipeline __init__:
vace_path = getattr(config, "vace_path", None)
model_class = CausalVaceWanModel if vace_path is not None else CausalWanModel

# Configure VACE if needed
if vace_path is not None:
    base_model_kwargs = dict(base_model_kwargs) if base_model_kwargs else {}
    if "vace_in_dim" not in base_model_kwargs:
        base_model_kwargs["vace_in_dim"] = 96

generator = WanDiffusionWrapper(
    model_class,
    model_name=base_model_name,
    model_dir=model_dir,
    generator_path=generator_path,
    **base_model_kwargs,
)

# Load VACE weights before LoRA (same order as LongLive)
if vace_path:
    from .vace_weight_loader import load_vace_weights_only
    load_vace_weights_only(generator.model, vace_path)

# Then apply LoRAs
generator.model = self._init_loras(config, generator.model)
```

**Modular Blocks:**
- Each pipeline can optionally add `VaceEncodingBlock` to their block list
- `DenoiseBlock` already handles VACE params (backward compatible when None)
- Other pipelines add VACE blocks when needed:

```python
# In modular_blocks.py:
ALL_BLOCKS = InsertableDict([
    ("text_conditioning", TextConditioningBlock),
    # ... other blocks ...
    ("vace_encoding", VaceEncodingBlock),  # Add if VACE support desired
    ("denoise", DenoiseBlock),
    # ... rest of blocks ...
])
```

**Pros:**
- **Minimal change** - Extends existing proven pattern
- **Clear separation** - Two distinct classes for two purposes
- **Low risk** - Just copy LongLive's approach to other pipelines
- **No refactoring** - Existing code continues to work

**Cons:**
- **Code duplication** - Each pipeline has similar VACE initialization logic
- **Maintenance** - Two model classes to maintain
- **Not DRY** - Same pattern repeated across pipelines

---

## Option B: Single Model Class with Optional Components

**Architecture:**
Merge into one `CausalWanModel` class that conditionally initializes VACE components based on configuration.

**Implementation:**
```python
class CausalWanModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        # ... existing params ...
        num_layers=32,
        # New: VACE configuration
        vace_enabled=False,
        vace_in_dim=None,
        vace_layers=None,
        **kwargs,
    ):
        super().__init__()

        # ... existing initialization ...

        # Standard attention blocks
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(...) if not vace_enabled
            else BaseWanAttentionBlock(..., block_id=...)
            for i in range(num_layers)
        ])

        # Conditionally add VACE components
        if vace_enabled:
            self._init_vace_components(vace_in_dim, vace_layers)

    def _init_vace_components(self, vace_in_dim, vace_layers):
        """Initialize VACE-specific components."""
        self.vace_in_dim = vace_in_dim or self.in_dim
        self.vace_layers = vace_layers or [i for i in range(0, self.num_layers, 2)]
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # VACE blocks (parallel processing)
        self.vace_blocks = nn.ModuleList([...])

        # VACE patch embedding
        self.vace_patch_embedding = nn.Conv3d(...)

    def forward_vace(self, ...):
        """Process VACE hints if VACE enabled."""
        if not self.config.vace_enabled:
            return None
        # ... VACE processing ...
```

**Pipeline Integration:**
```python
# In each pipeline __init__:
vace_path = getattr(config, "vace_path", None)

# Configure model with VACE if needed
base_model_kwargs = dict(base_model_kwargs) if base_model_kwargs else {}
if vace_path is not None:
    base_model_kwargs["vace_enabled"] = True
    base_model_kwargs["vace_in_dim"] = 96  # or from config

generator = WanDiffusionWrapper(
    CausalWanModel,  # Always the same class
    model_name=base_model_name,
    model_dir=model_dir,
    generator_path=generator_path,
    **base_model_kwargs,
)

# Load VACE weights if enabled
if vace_path:
    from .vace_weight_loader import load_vace_weights_only
    load_vace_weights_only(generator.model, vace_path)

# Then apply LoRAs
generator.model = self._init_loras(config, generator.model)
```

**Pros:**
- **Single model class** - One class to maintain
- **Configuration-driven** - VACE is just a config parameter
- **DRY** - No code duplication
- **Discoverable** - VACE config visible in model config
- **Extensible** - Easy to add other optional features (IP-Adapter, etc.)

**Cons:**
- **Requires refactoring** - Need to merge two classes
- **Branch complexity** - Conditional logic in model __init__
- **Testing complexity** - Need to test both paths in single class

---

## Option C: VACEEnabledPipeline Mixin (Shared Initialization Logic)

**Architecture:**
Create a `VACEEnabledPipeline` mixin (parallel to `LoRAEnabledPipeline`) that handles VACE model selection and initialization.

**Implementation:**
```python
class VACEEnabledPipeline:
    """Shared VACE integration for WAN-based pipelines."""

    _vace_enabled: bool = False

    def _init_vace_model(self, config, base_model_kwargs):
        """
        Select appropriate model class and configure VACE.

        Returns:
            Tuple of (model_class, updated_kwargs)
        """
        vace_path = getattr(config, "vace_path", None)

        if not vace_path:
            # No VACE - use standard model
            return CausalWanModel, base_model_kwargs

        # VACE enabled
        self._vace_enabled = True

        # Option 1: Use separate class (if keeping two classes)
        base_model_kwargs = dict(base_model_kwargs) if base_model_kwargs else {}
        base_model_kwargs["vace_in_dim"] = base_model_kwargs.get("vace_in_dim", 96)
        return CausalVaceWanModel, base_model_kwargs

        # Option 2: Use single class with config (if using Option B)
        # base_model_kwargs["vace_enabled"] = True
        # base_model_kwargs["vace_in_dim"] = 96
        # return CausalWanModel, base_model_kwargs

    def _load_vace_weights(self, model, config):
        """Load VACE weights if configured."""
        vace_path = getattr(config, "vace_path", None)
        if vace_path and self._vace_enabled:
            from .vace_weight_loader import load_vace_weights_only
            load_vace_weights_only(model, vace_path)
```

**Pipeline Integration:**
```python
class LongLivePipeline(Pipeline, LoRAEnabledPipeline, VACEEnabledPipeline):
    def __init__(self, config, ...):
        # ... load model config ...

        # Select model class and configure VACE (one line!)
        model_class, base_model_kwargs = self._init_vace_model(config, base_model_kwargs)

        # Load generator
        generator = WanDiffusionWrapper(
            model_class,
            model_name=base_model_name,
            model_dir=model_dir,
            generator_path=generator_path,
            **base_model_kwargs,
        )

        # Load VACE weights (one line!)
        self._load_vace_weights(generator.model, config)

        # Apply LoRAs (after VACE)
        generator.model = self._init_loras(config, generator.model)

        # ... rest of initialization ...
```

**All Other Pipelines:**
```python
class RewardForcingPipeline(Pipeline, LoRAEnabledPipeline, VACEEnabledPipeline):
    def __init__(self, config, ...):
        # Same pattern - VACE support with 2 lines
        model_class, base_model_kwargs = self._init_vace_model(config, base_model_kwargs)
        generator = WanDiffusionWrapper(model_class, ...)
        self._load_vace_weights(generator.model, config)
        generator.model = self._init_loras(config, generator.model)
        # ...
```

**Benefits:**
- **DRY** - Initialization logic shared across all pipelines
- **Consistent** - Same pattern as LoRA (developers expect mixins)
- **Composable** - Works with multiple mixins (LoRA + VACE + future)
- **Minimal changes** - Just add mixin and 2 lines per pipeline
- **Flexible** - Works with both Option A (two classes) or Option B (single class) underneath

**Pros:**
- **Minimal per-pipeline code** - Just 2 lines per pipeline
- **Consistent with LoRA pattern** - Follows existing mixin approach
- **Easy to adopt** - Other pipelines just add mixin
- **Flexible** - Can use either two classes or single class underneath

**Cons:**
- **Multiple inheritance** - Adds another mixin
- **Abstraction layer** - One more concept to understand
- **Still needs Option A or B** - Mixin is just about sharing logic, not about model architecture

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

**Key Points:**

1. **Single implementation**: One `CausalVaceWanModel` works with ALL pipelines
2. **No per-pipeline variants**: When StreamDiffusionV2 adds new parameters, VACE automatically supports them
3. **Block replacement**: Same operation as current inheritance approach, just explicit
4. **Clear separation**: VACE components (BaseWanAttentionBlock, VaceWanAttentionBlock, vace_patch_embedding) live in one shared location
5. **Attribute proxying**: `__getattr__` ensures wrapped model attributes accessible

**Pros:**
- **Single source of truth** - One VACE implementation for entire codebase
- **Zero per-pipeline maintenance** - New pipelines automatically get VACE support
- **No refactoring** - Each pipeline keeps its own `CausalWanModel` with unique parameters
- **Composition over inheritance** - Clean wrapper pattern
- **Future-proof** - Works with any `CausalWanModel` variant
- **Same safety as current** - Block replacement happens once at initialization (identical to inheritance approach)

**Cons:**
- **Modifies wrapped model** - Replaces blocks (but current approach does same thing)
- **Requires shared location** - VACE components need common module (minor)
- **New pattern** - Team needs to understand composition approach (one-time learning)

---

## Comparison

| Criterion | Option A: Two Classes | Option B: Single Class | Option C: Mixin | Option D: Composition |
|-----------|----------------------|------------------------|-----------------|----------------------|
| **Code Changes** | Low - Copy pattern | Moderate - Refactor model | Low - Add mixin | Low - Single wrapper |
| **DRY** | Poor - Duplication | Excellent - Single source | Fair - Shared init only | Excellent - Single VACE impl |
| **SOLID** | Fair - Separate classes | Excellent - Config-driven | Good - Composition | Excellent - Composition |
| **Maintainability** | Poor - N classes per pipeline | Good - Single class | Fair - Still N VACE classes | Excellent - One VACE class total |
| **Risk** | Low - Proven pattern | Moderate - Refactoring | Low - Additive only | Low - Same as current |
| **Elegance** | Fair - Pragmatic | Excellent - Architectural | Fair - Adds abstraction | Excellent - Clean wrapper |
| **Extensibility** | Poor - Copy code | High - Add config params | Moderate - Per-pipeline | Excellent - Any base model |
| **Per-Pipeline VACE** | Yes - One per pipeline | No - Single unified | Yes - One per pipeline | No - Single wrapper |
| **Refactoring Required** | None | Extensive - Merge models | None | None |

---

## Recommended Approach

**Best Overall: Option D (Composition-Based Wrapper)**

Why:
1. **Single VACE implementation** - One `CausalVaceWanModel` works with ALL pipelines
2. **Zero refactoring required** - Each pipeline keeps its unique `CausalWanModel` implementation
3. **Composition over inheritance** - Clean wrapper pattern, doesn't inherit from specific pipeline
4. **Future-proof** - Automatically works with new pipelines and their unique parameters
5. **Same safety as current** - Block replacement happens once at init (identical to inheritance approach)
6. **Zero per-pipeline maintenance** - When any pipeline adds new features, VACE automatically supports them
7. **Clean separation** - VACE logic lives in one shared location

Your boss correctly identified that:
- VACE logic is identical regardless of base model
- Each pipeline has different `CausalWanModel` parameters (window_size vs local_attn_size, etc.)
- Refactoring all pipelines to share one `CausalWanModel` is messy (Option B)
- A single wrapper that accepts ANY `CausalWanModel` is the cleanest solution

Implementation path:
1. Create shared `src/scope/core/pipelines/shared/vace/causal_vace_model.py`
2. Implement composition-based `CausalVaceWanModel` that wraps any `CausalWanModel`
3. Each pipeline conditionally wraps its `CausalWanModel` with VACE wrapper
4. Load VACE weights from shared `vace_weight_loader.py`

**Alternative if Option D Feels Risky: Option A (Two Classes)**

If composition wrapper feels unfamiliar:
1. Copy LongLive's `CausalVaceWanModel` pattern to each pipeline
2. Each pipeline gets its own `CausalVaceWanModel` that inherits from its `CausalWanModel`
3. Proven to work, minimal risk, but requires per-pipeline VACE maintenance

This gives you proven pattern but requires maintaining N VACE implementations.

---

## Why Not Other Options?

**Option B (Single Unified Model)**: Requires extensive refactoring to merge 4+ different `CausalWanModel` implementations (StreamDiffusionV2 has `window_size`, KreaRealtimeVideo has `kv_cache_attention_bias`, etc.). This is exactly what your boss wanted to avoid.

**Option C (Mixin)**: Can't actually share VACE model classes across pipelines because each pipeline needs its own `CausalVaceWanModel` that inherits from its specific `CausalWanModel`. The mixin would only share initialization boilerplate, not the actual VACE logic. You'd still maintain N VACE classes.

---

## Implementation Notes

Regardless of which option chosen:

1. **Weight Loading Order MUST be preserved:**
   - Base model weights → VACE weights → LoRA weights
   - Critical for PEFT compatibility

2. **Modular Blocks remain unchanged:**
   - `VaceEncodingBlock` already exists and is platform-agnostic
   - Other pipelines can add it to their block lists when VACE support desired
   - `DenoiseBlock` already handles VACE params (backward compatible)

3. **LoRA Compatibility:**
   - `BaseWanAttentionBlock` extends `CausalWanAttentionBlock`
   - LoRAs target the same Linear layers in both block types
   - No changes needed to LoRA strategies

4. **Testing Strategy:**
   - Ensure VACE disabled = identical to current CausalWanModel
   - Ensure VACE enabled = identical to current CausalVaceWanModel
   - Test LoRA compatibility in both modes
   - Test each pipeline with and without VACE
