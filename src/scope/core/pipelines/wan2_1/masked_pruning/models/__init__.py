from .attention_blocks import create_pruned_block_class, create_pruned_self_attn_class
from .causal_pruned_model import CausalPrunedWanModel

__all__ = [
    "CausalPrunedWanModel",
    "create_pruned_block_class",
    "create_pruned_self_attn_class",
]
