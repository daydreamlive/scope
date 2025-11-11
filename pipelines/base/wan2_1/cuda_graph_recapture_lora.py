"""
CUDA Graph Re-capture LoRA manager for WAN models.

This implementation combines PEFT LoRA loading with CUDA Graph optimization.
Captures the entire forward pass into a CUDA Graph after loading LoRAs via PEFT,
enabling near-zero Python overhead during inference through graph replay.

PERFORMANCE TARGET:
- Inference: ~9-10 FPS (graph replay with minimal overhead)
- Updates: 1-5s (PEFT scale update + graph re-capture)
- Init time: 5-10s (PEFT loading + initial capture)
- Memory: Minimal overhead (PEFT layers + graph storage)

KEY STRATEGY:
1. Load LoRAs using PEFT (instant scale updates capability)
2. Capture model forward pass into CUDA Graph (eliminates Python overhead)
3. Stream using graph replay (fast, predictable latency)
4. On scale update: update PEFT scales + re-capture graph (moderate cost)

ADVANTAGES:
- Near-optimal inference speed (graph replay)
- Fast updates compared to weight reconstruction
- Clean separation of concerns (PEFT for flexibility, graphs for speed)

LIMITATIONS:
- Requires static input shapes (captured at init)
- Requires PyTorch 2.1+ with CUDA
- No dynamic control flow during forward pass
- Re-capture cost on scale updates (acceptable for real-time video)
"""
from typing import Dict, Any, List, Optional
import os
import time
import logging
from pathlib import Path
import torch
import torch.nn as nn
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

__all__ = ["CudaGraphRecaptureLoRAManager"]


class CudaGraphRecaptureLoRAManager:
    """
    Manages LoRA adapters using PEFT with CUDA Graph optimization.

    Combines PEFT's runtime flexibility with CUDA Graph's performance:
    - LoRAs loaded via PEFT for instant scale parameter updates
    - Forward pass captured into CUDA Graph for fast replay
    - Scale updates trigger graph re-capture (1-5s vs 60s for gpu_reconstruct)

    Compatible with FP8 quantization via PEFT's torchao support.
    """

    # Store state per model instance
    _model_states = {}

    @staticmethod
    def _get_model_state(model: nn.Module) -> Dict[str, Any]:
        """Get or create state dict for a model instance."""
        model_id = id(model)
        if model_id not in CudaGraphRecaptureLoRAManager._model_states:
            CudaGraphRecaptureLoRAManager._model_states[model_id] = {
                "peft_model": None,  # PEFT-wrapped model
                "original_model": None,  # Original unwrapped model (for reference tracking)
                "original_call": None,  # Original __call__ method before wrapping
                "wrapper": None,  # WanDiffusionWrapper reference (if applicable)
                "graph": None,  # Captured CUDA Graph
                "static_args": None,  # Static positional args for graph
                "static_kwargs": None,  # Static keyword args for graph
                "static_output": None,  # Static output from graph
                "captured": False,  # Whether graph has been captured
                "capture_on_next_call": True,  # Lazy capture on first forward pass
            }
        return CudaGraphRecaptureLoRAManager._model_states[model_id]

    @staticmethod
    def _sanitize_adapter_name(adapter_name: str) -> str:
        """
        Sanitize adapter name to be valid for PyTorch module names.

        PyTorch module names cannot contain periods (.), so we replace them
        with underscores.
        """
        sanitized = adapter_name.replace('.', '_')
        sanitized = sanitized.replace('/', '_').replace('\\', '_')
        return sanitized

    @staticmethod
    def _normalize_lora_key(lora_base_key: str) -> str:
        """
        Normalize LoRA base key to match model state dict format.

        Handles various LoRA naming conventions:
        - lora_unet_blocks_0_cross_attn_k -> blocks.0.cross_attn.k
        - diffusion_model.blocks.0.cross_attn.k -> blocks.0.cross_attn.k
        - blocks.0.cross_attn.k -> blocks.0.cross_attn.k
        """
        if lora_base_key.startswith("lora_unet_"):
            key = lora_base_key[len("lora_unet_"):]
            import re
            key = re.sub(r'_(\d+)_', r'.\1.', key)
            key = key.replace('_', '.')
            return key

        if lora_base_key.startswith("diffusion_model."):
            return lora_base_key[len("diffusion_model."):]

        return lora_base_key

    @staticmethod
    def _load_lora_weights(lora_path: str) -> Dict[str, torch.Tensor]:
        """Load LoRA weights from .safetensors or .bin file."""
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"_load_lora_weights: LoRA file not found: {lora_path}")

        if lora_path.endswith('.safetensors'):
            return load_file(lora_path)
        else:
            return torch.load(lora_path, map_location='cpu')

    @staticmethod
    def _parse_lora_weights(
        lora_state: Dict[str, torch.Tensor],
        model_state: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Parse LoRA weights and match them to model parameters.

        Returns dict mapping model parameter names to LoRA info.
        """
        lora_mapping = {}
        processed_keys = set()

        # Build model key map
        model_key_map = {}
        for key in model_state.keys():
            if key.endswith('.weight'):
                base_key = key[:-len('.weight')]
                model_key_map[base_key] = key
                model_key_map[f"diffusion_model.{base_key}"] = key

        # Iterate through LoRA keys to find A/B or up/down pairs
        for lora_key in lora_state.keys():
            if lora_key in processed_keys:
                continue

            lora_A, lora_B, alpha_key = None, None, None
            base_key = None

            if '.lora_up.weight' in lora_key:
                base_key = lora_key.replace('.lora_up.weight', '')
                lora_B_key = f"{base_key}.lora_down.weight"
                alpha_key = f"{base_key}.alpha"
                if lora_B_key in lora_state:
                    lora_B = lora_state[lora_key]
                    lora_A = lora_state[lora_B_key]
                    processed_keys.add(lora_key)
                    processed_keys.add(lora_B_key)

            elif '.lora_B.weight' in lora_key:
                base_key = lora_key.replace('.lora_B.weight', '')
                lora_A_key = f"{base_key}.lora_A.weight"
                alpha_key = f"{base_key}.alpha"
                if lora_A_key in lora_state:
                    lora_B = lora_state[lora_key]
                    lora_A = lora_state[lora_A_key]
                    processed_keys.add(lora_key)
                    processed_keys.add(lora_A_key)

            else:
                continue

            if base_key is None or lora_A is None or lora_B is None:
                continue

            # Normalize the base key
            normalized_key = CudaGraphRecaptureLoRAManager._normalize_lora_key(base_key)

            # Find matching model key
            model_key = model_key_map.get(normalized_key)
            if model_key is None:
                model_key = model_key_map.get(f"diffusion_model.{normalized_key}")

            if model_key is None:
                continue

            # Extract alpha and rank
            alpha = None
            if alpha_key and alpha_key in lora_state:
                alpha = lora_state[alpha_key].item()

            rank = lora_A.shape[0]

            lora_mapping[model_key] = {
                "lora_A": lora_A,
                "lora_B": lora_B,
                "alpha": alpha,
                "rank": rank
            }

        return lora_mapping

    @staticmethod
    def get_wrapped_model(model: nn.Module) -> Optional[nn.Module]:
        """
        Get the PEFT-wrapped model if it exists, otherwise return None.

        This should be called after load_adapter to get the wrapped model
        for replacing the original model reference in parent objects.

        Args:
            model: Original model passed to load_adapter

        Returns:
            PEFT-wrapped model if it exists, None otherwise
        """
        state = CudaGraphRecaptureLoRAManager._get_model_state(model)
        return state.get("peft_model")

    @staticmethod
    def _inject_lora_layers(
        model: nn.Module,
        lora_mapping: Dict[str, Dict[str, Any]],
        adapter_name: str,
        strength: float = 1.0
    ) -> nn.Module:
        """
        Inject PEFT LoRA layers into the model and wrap with lazy graph capture.

        Returns the PEFT-wrapped model with graph capture wrapper.
        """
        from peft import LoraConfig, get_peft_model
        from peft.tuners.lora import LoraLayer

        state = CudaGraphRecaptureLoRAManager._get_model_state(model)

        # Determine target modules from lora_mapping
        target_modules = []
        for param_name in lora_mapping.keys():
            if param_name.endswith('.weight'):
                module_path = param_name[:-len('.weight')]

                # Verify this is actually a Linear layer in the model
                parts = module_path.split('.')
                try:
                    current = model
                    for part in parts:
                        current = getattr(current, part)

                    if isinstance(current, nn.Linear):
                        target_modules.append(module_path)
                except AttributeError:
                    logger.debug(f"_inject_lora_layers: Module {module_path} not found in model")
                    continue

        if not target_modules:
            logger.warning("_inject_lora_layers: No target modules found in LoRA mapping")
            return model

        logger.info(f"_inject_lora_layers: Targeting {len(target_modules)} Linear modules")

        # Infer rank from first LoRA in mapping
        first_lora = next(iter(lora_mapping.values()))
        rank = first_lora['rank']
        alpha = first_lora['alpha']
        if alpha is None:
            alpha = rank

        # Create PEFT config
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            init_lora_weights=False,
            modules_to_save=None,
        )

        # Check if model already has PEFT adapters
        if state["peft_model"] is not None:
            logger.info(f"_inject_lora_layers: Adding adapter '{adapter_name}' to existing PEFT model")
            state["peft_model"].add_adapter(adapter_name, lora_config)
            peft_model = state["peft_model"]
        else:
            logger.info(f"_inject_lora_layers: Creating new PEFT model with adapter '{adapter_name}'")
            peft_model = get_peft_model(model, lora_config, adapter_name=adapter_name)

            # Store references
            state["peft_model"] = peft_model
            state["original_model"] = model

            # Store the original forward for graph capture
            original_peft_forward = peft_model.forward
            state["original_call"] = original_peft_forward

            # CRITICAL FIX: Override forward() method instead of __call__
            # nn.Module's __call__ is a class-level descriptor, but it calls self.forward()
            # which IS looked up at the instance level. This is the proper way to intercept.
            def wrapped_forward(*args, **kwargs):
                logger.debug(f"_inject_lora_layers.wrapped_forward: Intercepted PEFT model forward")
                # Get the state using peft_model as key
                peft_state = CudaGraphRecaptureLoRAManager._get_model_state(peft_model)

                # Delegate to lazy graph forward logic
                return CudaGraphRecaptureLoRAManager._lazy_graph_forward(
                    peft_model, original_peft_forward, peft_state, *args, **kwargs
                )

            peft_model.forward = wrapped_forward
            logger.info(f"_inject_lora_layers: Wrapped PEFT model's forward() for lazy CUDA Graph capture")

        # Load LoRA weights into PEFT layers
        loaded_count = 0
        for param_name, lora_info in lora_mapping.items():
            module_path = param_name[:-len('.weight')] if param_name.endswith('.weight') else param_name
            parts = module_path.split('.')

            try:
                # Navigate to the PEFT-wrapped module
                current = peft_model.base_model.model
                for part in parts:
                    current = getattr(current, part)

                if not isinstance(current, LoraLayer):
                    logger.debug(f"_inject_lora_layers: {module_path} is not a LoraLayer, skipping")
                    continue

                lora_A_weight = lora_info['lora_A']
                lora_B_weight = lora_info['lora_B']

                if adapter_name in current.lora_A:
                    current.lora_A[adapter_name].weight.data = lora_A_weight.to(
                        device=current.lora_A[adapter_name].weight.device,
                        dtype=current.lora_A[adapter_name].weight.dtype
                    )
                    current.lora_B[adapter_name].weight.data = lora_B_weight.to(
                        device=current.lora_B[adapter_name].weight.device,
                        dtype=current.lora_B[adapter_name].weight.dtype
                    )

                    # Set initial scaling
                    current.scaling[adapter_name] = strength

                    loaded_count += 1
                else:
                    logger.debug(f"_inject_lora_layers: Adapter '{adapter_name}' not found in {module_path}")

            except AttributeError as e:
                logger.debug(f"_inject_lora_layers: Could not find module {module_path}: {e}")
                continue

        logger.info(f"_inject_lora_layers: Loaded {loaded_count} LoRA weight pairs")

        # Activate the adapter
        peft_model.set_adapter(adapter_name)

        return peft_model

    @staticmethod
    def _clone_tensor_structure(obj):
        """
        Recursively clone tensors in nested structures (lists, tuples, dicts).
        Creates new tensors with same shape/dtype/device for CUDA graph capture.
        """
        if isinstance(obj, torch.Tensor):
            return obj.clone().detach()
        elif isinstance(obj, list):
            return [CudaGraphRecaptureLoRAManager._clone_tensor_structure(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(CudaGraphRecaptureLoRAManager._clone_tensor_structure(item) for item in obj)
        elif isinstance(obj, dict):
            return {k: CudaGraphRecaptureLoRAManager._clone_tensor_structure(v) for k, v in obj.items()}
        else:
            # Non-tensor types (int, float, str, etc.) are returned as-is
            return obj

    @staticmethod
    def _copy_tensor_data(dest, src):
        """
        Recursively copy tensor data from src to dest structure.
        Used to update static graph inputs with new data.
        """
        if isinstance(dest, torch.Tensor) and isinstance(src, torch.Tensor):
            dest.copy_(src)
        elif isinstance(dest, list) and isinstance(src, list):
            for d, s in zip(dest, src):
                CudaGraphRecaptureLoRAManager._copy_tensor_data(d, s)
        elif isinstance(dest, tuple) and isinstance(src, tuple):
            for d, s in zip(dest, src):
                CudaGraphRecaptureLoRAManager._copy_tensor_data(d, s)
        elif isinstance(dest, dict) and isinstance(src, dict):
            for key in dest:
                if key in src:
                    CudaGraphRecaptureLoRAManager._copy_tensor_data(dest[key], src[key])

    @staticmethod
    def _capture_cuda_graph(
        model: nn.Module,
        original_call,
        args: tuple,
        kwargs: dict
    ) -> None:
        """
        Capture model forward pass into CUDA Graph with all arguments.

        Args:
            model: Model to capture (should be PEFT-wrapped)
            original_call: Original __call__ method to use during capture
            args: Positional arguments to capture
            kwargs: Keyword arguments to capture
        """
        state = CudaGraphRecaptureLoRAManager._get_model_state(model)

        logger.info(f"_capture_cuda_graph: Capturing graph with {len(args)} args and {len(kwargs)} kwargs")

        # Set model to eval mode (required for static graph)
        model.eval()

        # Clone all arguments to create static versions
        static_args = CudaGraphRecaptureLoRAManager._clone_tensor_structure(args)
        static_kwargs = CudaGraphRecaptureLoRAManager._clone_tensor_structure(kwargs)
        static_output = None

        # IMPORTANT: Use original_call to avoid triggering our wrapper during capture
        # This prevents infinite recursion and ensures we capture the actual model forward pass

        # Warmup: run forward pass a few times to initialize CUDA context
        with torch.no_grad():
            for _ in range(3):
                _ = original_call(*static_args, **static_kwargs)

        # Synchronize before capture
        torch.cuda.synchronize()

        # Capture the graph
        graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(graph):
            static_output = original_call(*static_args, **static_kwargs)

        # Synchronize after capture
        torch.cuda.synchronize()

        # Store in state
        state["graph"] = graph
        state["static_args"] = static_args
        state["static_kwargs"] = static_kwargs
        state["static_output"] = static_output
        state["captured"] = True

        logger.info(f"_capture_cuda_graph: Graph captured successfully")

    @staticmethod
    def _replay_cuda_graph(
        model: nn.Module,
        args: tuple,
        kwargs: dict
    ):
        """
        Replay captured CUDA Graph with new input data.

        Args:
            model: Model with captured graph
            args: New positional arguments
            kwargs: New keyword arguments

        Returns:
            Output from graph replay
        """
        state = CudaGraphRecaptureLoRAManager._get_model_state(model)

        if not state["captured"]:
            raise RuntimeError("_replay_cuda_graph: No graph has been captured yet")

        # Copy new input data to static tensors
        CudaGraphRecaptureLoRAManager._copy_tensor_data(state["static_args"], args)
        CudaGraphRecaptureLoRAManager._copy_tensor_data(state["static_kwargs"], kwargs)

        # Replay graph
        state["graph"].replay()

        # Return output (clone if it's a tensor to avoid overwrites on next replay)
        output = state["static_output"]
        if isinstance(output, torch.Tensor):
            return output.clone()
        else:
            return CudaGraphRecaptureLoRAManager._clone_tensor_structure(output)

    @staticmethod
    def _lazy_graph_forward(peft_model: nn.Module, original_call, state: Dict[str, Any], *args, **kwargs):
        """
        Lazy graph capture wrapper for forward pass.

        On first call: captures CUDA Graph with actual input shape
        Subsequent calls: replays captured graph

        NO FALLBACKS - if capture fails, the application must stop.

        Args:
            peft_model: The PEFT-wrapped model
            original_call: The original __call__ method to use for non-graph execution
            state: The model state dict containing graph and capture info
            *args, **kwargs: Arguments to forward to the model
        """
        logger.debug(f"_lazy_graph_forward: Called with args={len(args)}, kwargs={list(kwargs.keys())}")
        logger.debug(f"_lazy_graph_forward: State - captured={state['captured']}, capture_on_next_call={state.get('capture_on_next_call', False)}")

        # If we should capture on next call and haven't captured yet
        if state.get("capture_on_next_call", False) and not state["captured"]:
            # First call - capture the graph with all arguments
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "_lazy_graph_forward: CUDA Graph capture requires CUDA, but CUDA is not available. "
                    "Cannot proceed with cuda_graph_recapture mode."
                )

            # Verify at least one tensor argument is on CUDA
            has_cuda_tensor = False
            if args:
                for arg in args:
                    if isinstance(arg, torch.Tensor) and arg.is_cuda:
                        has_cuda_tensor = True
                        break
            if not has_cuda_tensor and kwargs:
                for v in kwargs.values():
                    if isinstance(v, torch.Tensor) and v.is_cuda:
                        has_cuda_tensor = True
                        break

            if not has_cuda_tensor:
                raise RuntimeError(
                    "_lazy_graph_forward: CUDA Graph capture requires CUDA tensors, but no CUDA tensors found in inputs. "
                    "Cannot proceed with cuda_graph_recapture mode."
                )

            logger.info(f"_lazy_graph_forward: Attempting CUDA Graph capture on first forward pass")
            logger.info(f"_lazy_graph_forward: Args: {len(args)}, Kwargs keys: {list(kwargs.keys())}")

            # Capture graph - if it fails, raise the exception
            CudaGraphRecaptureLoRAManager._capture_cuda_graph(
                peft_model, original_call, args, kwargs
            )
            state["capture_on_next_call"] = False
            logger.info(f"_lazy_graph_forward: CUDA Graph captured successfully")

        # If graph is captured, use it
        if state["captured"]:
            logger.debug(f"_lazy_graph_forward: Using captured CUDA Graph for inference")
            return CudaGraphRecaptureLoRAManager._replay_cuda_graph(peft_model, args, kwargs)

        # Should not reach here
        raise RuntimeError("_lazy_graph_forward: Unexpected state - graph not captured and capture not scheduled")

    @staticmethod
    def load_adapter(
        model: nn.Module,
        lora_path: str,
        strength: float = 1.0,
        adapter_name: Optional[str] = None,
        capture_shape: Optional[tuple] = None
    ) -> str:
        """
        Load LoRA adapter using PEFT and capture CUDA Graph.

        Args:
            model: PyTorch model
            lora_path: Path to LoRA file (.safetensors or .bin)
            strength: Initial strength multiplier (default 1.0)
            adapter_name: Optional adapter name (defaults to filename)
            capture_shape: Shape for CUDA Graph capture (e.g., (1, 16, 512))
                          If None, graph capture is skipped (can be done later)

        Returns:
            The adapter name used

        Example:
            >>> adapter_name = CudaGraphRecaptureLoRAManager.load_adapter(
            ...     model=pipeline.transformer,
            ...     lora_path="models/lora/my-style.safetensors",
            ...     strength=1.0,
            ...     capture_shape=(1, 16, 512)
            ... )
        """
        start_time = time.time()

        if adapter_name is None:
            adapter_name = Path(lora_path).stem

        # Sanitize adapter name
        original_adapter_name = adapter_name
        adapter_name = CudaGraphRecaptureLoRAManager._sanitize_adapter_name(adapter_name)
        if adapter_name != original_adapter_name:
            logger.debug(f"load_adapter: Sanitized adapter name '{original_adapter_name}' -> '{adapter_name}'")

        logger.info(f"load_adapter: Loading LoRA from {lora_path} as adapter '{adapter_name}'")

        # Load LoRA weights
        lora_state = CudaGraphRecaptureLoRAManager._load_lora_weights(lora_path)
        logger.debug(f"load_adapter: Loaded {len(lora_state)} tensors from file")

        # Get model state dict
        model_state = model.state_dict()

        # Parse and map LoRA weights to model parameters
        lora_mapping = CudaGraphRecaptureLoRAManager._parse_lora_weights(lora_state, model_state)
        logger.info(f"load_adapter: Mapped {len(lora_mapping)} LoRA layers to model parameters")

        if not lora_mapping:
            logger.warning(f"load_adapter: No LoRA layers matched model parameters")
            return adapter_name

        # Inject PEFT LoRA layers
        peft_model = CudaGraphRecaptureLoRAManager._inject_lora_layers(
            model, lora_mapping, adapter_name, strength
        )

        # Capture CUDA Graph if shape is provided
        if capture_shape is not None:
            if not torch.cuda.is_available():
                logger.warning("load_adapter: CUDA not available, skipping graph capture")
            else:
                device = next(peft_model.parameters()).device
                CudaGraphRecaptureLoRAManager._capture_cuda_graph(
                    peft_model, capture_shape, device
                )

        elapsed = time.time() - start_time
        logger.info(f"load_adapter: Loaded adapter '{adapter_name}' in {elapsed:.3f}s")

        return adapter_name

    @staticmethod
    def load_adapters_from_list(
        model: nn.Module,
        lora_configs: List[Dict[str, Any]],
        logger_prefix: str = "",
        capture_shape: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """
        Load multiple LoRA adapters using PEFT and capture CUDA Graph.

        Args:
            model: PyTorch model
            lora_configs: List of dicts with keys:
                - path (str, required)
                - scale (float, optional, default=1.0)
                - adapter_name (str, optional)
            logger_prefix: Prefix for log messages
            capture_shape: Shape for CUDA Graph capture (only captured after all LoRAs loaded)

        Returns:
            List of loaded adapter info dicts with keys: adapter_name, path, scale

        Example:
            >>> loaded = CudaGraphRecaptureLoRAManager.load_adapters_from_list(
            ...     model=pipeline.transformer,
            ...     lora_configs=[{"path": "models/lora/style.safetensors", "scale": 1.0}],
            ...     capture_shape=(1, 16, 512)
            ... )
        """
        loaded_adapters = []

        if not lora_configs:
            return loaded_adapters

        # Load all LoRAs without capturing (capture once at the end)
        for lora_config in lora_configs:
            lora_path = lora_config.get("path")
            if not lora_path:
                logger.warning(f"{logger_prefix}Skipping LoRA config with no path")
                continue

            scale = lora_config.get("scale", 1.0)
            adapter_name = lora_config.get("adapter_name")

            try:
                returned_adapter_name = CudaGraphRecaptureLoRAManager.load_adapter(
                    model=model,
                    lora_path=lora_path,
                    strength=scale,
                    adapter_name=adapter_name,
                    capture_shape=None  # Skip capture for now
                )

                logger.info(f"{logger_prefix}Loaded LoRA '{Path(lora_path).name}' as '{returned_adapter_name}' (scale={scale})")

                loaded_adapters.append({
                    "adapter_name": returned_adapter_name,
                    "path": lora_path,
                    "scale": scale,
                })

            except FileNotFoundError as e:
                logger.error(f"{logger_prefix}LoRA file not found: {lora_path}")
                raise RuntimeError(
                    f"{logger_prefix}LoRA loading failed. File not found: {lora_path}"
                ) from e
            except Exception as e:
                logger.error(f"{logger_prefix}Failed to load LoRA: {e}", exc_info=True)
                raise RuntimeError(
                    f"{logger_prefix}LoRA loading failed: {e}"
                ) from e

        # Capture graph once after all LoRAs are loaded
        if capture_shape is not None and loaded_adapters:
            state = CudaGraphRecaptureLoRAManager._get_model_state(model)
            if state["peft_model"] is not None and torch.cuda.is_available():
                device = next(state["peft_model"].parameters()).device
                CudaGraphRecaptureLoRAManager._capture_cuda_graph(
                    state["peft_model"], capture_shape, device
                )
            elif not torch.cuda.is_available():
                logger.warning(f"{logger_prefix}CUDA not available, skipping graph capture")

        return loaded_adapters

    @staticmethod
    def update_adapter_scales(
        model: nn.Module,
        loaded_adapters: List[Dict[str, Any]],
        scale_updates: List[Dict[str, Any]],
        logger_prefix: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Update LoRA adapter scales at runtime and re-capture CUDA Graph.

        This operation takes 1-5s: instant PEFT scale update + graph re-capture.

        Args:
            model: PyTorch model with loaded LoRA adapters
            loaded_adapters: List of currently loaded adapter info dicts
            scale_updates: List of dicts with 'adapter_name' (or 'path') and 'scale' keys
            logger_prefix: Prefix for log messages

        Returns:
            Updated loaded_adapters list

        Example:
            >>> self.loaded_lora_adapters = CudaGraphRecaptureLoRAManager.update_adapter_scales(
            ...     model=self.stream.generator.model,
            ...     loaded_adapters=self.loaded_lora_adapters,
            ...     scale_updates=[{"adapter_name": "my_style", "scale": 0.5}]
            ... )
        """
        if not scale_updates:
            return loaded_adapters

        state = CudaGraphRecaptureLoRAManager._get_model_state(model)
        peft_model = state["peft_model"]

        if peft_model is None:
            logger.warning(f"{logger_prefix}No PEFT model found, cannot update scales")
            return loaded_adapters

        # Build map from adapter_name and path to scale
        scale_map = {}
        for update in scale_updates:
            adapter_name = update.get("adapter_name")
            path = update.get("path")
            scale = update.get("scale")

            if scale is None:
                continue

            if adapter_name:
                scale_map[("adapter_name", adapter_name)] = scale
            if path:
                scale_map[("path", path)] = scale

        if not scale_map:
            return loaded_adapters

        # Update scales in PEFT model
        updates_applied = 0
        for adapter_info in loaded_adapters:
            adapter_name = adapter_info.get("adapter_name")
            path = adapter_info.get("path")

            # Check if we have a scale update for this adapter
            new_scale = scale_map.get(("adapter_name", adapter_name))
            if new_scale is None:
                new_scale = scale_map.get(("path", path))

            if new_scale is None:
                continue

            old_scale = adapter_info.get("scale", 1.0)
            if abs(old_scale - new_scale) < 1e-6:
                continue

            # Update scale in all LoraLayer modules
            from peft.tuners.lora import LoraLayer

            for name, module in peft_model.named_modules():
                if isinstance(module, LoraLayer):
                    if adapter_name in module.scaling:
                        module.scaling[adapter_name] = new_scale

            # Update in loaded_adapters list
            adapter_info["scale"] = new_scale
            updates_applied += 1

            logger.info(
                f"{logger_prefix}Updated LoRA '{adapter_name}' scale: {old_scale:.3f} -> {new_scale:.3f}"
            )

        # Re-capture CUDA Graph if scales were updated and graph exists
        if updates_applied > 0 and state["captured"]:
            logger.info(f"{logger_prefix}Re-capturing CUDA Graph after scale updates...")
            recapture_start = time.time()

            # Retrieve original_call from the state (we need to store it during initial setup)
            # For now, use the peft_model's base __call__ (unwrapped version)
            original_call = state.get("original_call")
            if original_call is None:
                logger.warning(f"{logger_prefix}No original_call stored, using current peft_model.__call__")
                # This might not work if we've wrapped it, but it's a fallback
                original_call = super(type(peft_model), peft_model).__call__

            # Use the stored static args and kwargs as template for recapture
            # Let it fail if re-capture fails
            CudaGraphRecaptureLoRAManager._capture_cuda_graph(
                peft_model, original_call, state["static_args"], state["static_kwargs"]
            )
            recapture_time = time.time() - recapture_start
            logger.info(f"{logger_prefix}Graph re-captured in {recapture_time:.3f}s")

        if updates_applied > 0:
            logger.debug(f"{logger_prefix}Applied {updates_applied} scale updates")

        return loaded_adapters
