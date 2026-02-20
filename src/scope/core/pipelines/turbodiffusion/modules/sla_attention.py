# Ported from https://github.com/thu-ml/TurboDiffusion
# SLA (Sparse-Linear Attention) replacement for Wan2.1 self-attention layers.
# Requires the SLA library from https://github.com/thu-ml/SLA:
#   pip install git+https://github.com/thu-ml/SLA.git
# SageSLA additionally requires SpargeAttn:
#   pip install git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation

import logging

import torch.nn as nn

logger = logging.getLogger(__name__)

SLA_AVAILABLE = False
SAGESLA_AVAILABLE = False

try:
    from sparse_linear_attention import SparseLinearAttention as SLA

    SLA_AVAILABLE = True
except ImportError:
    pass

try:
    from SageSLA import SageSparseLinearAttention as SageSLA

    SAGESLA_AVAILABLE = True
except ImportError:
    pass


def replace_attention_with_sla(
    model: nn.Module,
    attention_type: str = "sagesla",
    topk: float = 0.12,
) -> nn.Module:
    """Replace self-attention in Wan model blocks with SLA/SageSLA variants.

    Walks through all WanSelfAttention modules in the model and replaces
    their attention operation with a sparse variant for faster inference.

    Args:
        model: The WanModel instance to modify in-place.
        attention_type: One of "sla", "sagesla", or "original".
            If "original", no replacement is done.
        topk: Top-k ratio for sparse attention (fraction of tokens to attend to).

    Returns:
        The modified model (same object, modified in-place).
    """
    if attention_type == "original":
        return model

    if attention_type == "sagesla" and not SAGESLA_AVAILABLE:
        if SLA_AVAILABLE:
            logger.warning("SageSLA not available, falling back to SLA attention")
            attention_type = "sla"
        else:
            logger.warning(
                "Neither SageSLA nor SLA available (install spas/SLA package). "
                "Using original attention."
            )
            return model

    if attention_type == "sla" and not SLA_AVAILABLE:
        logger.warning(
            "SLA not available (install spas/SLA package). Using original attention."
        )
        return model

    replaced_count = 0
    for module in model.modules():
        # Match WanSelfAttention by checking for the expected attributes
        if (
            hasattr(module, "q")
            and hasattr(module, "k")
            and hasattr(module, "v")
            and hasattr(module, "o")
            and hasattr(module, "num_heads")
            and hasattr(module, "head_dim")
            and type(module).__name__ == "WanSelfAttention"
        ):
            head_dim = module.head_dim
            if attention_type == "sla":
                module.local_attn = SLA(head_dim=head_dim, topk=topk, BLKQ=128, BLKK=64)
            elif attention_type == "sagesla":
                module.local_attn = SageSLA(head_dim=head_dim, topk=topk)
            replaced_count += 1

    logger.info(
        f"Replaced {replaced_count} self-attention layers with {attention_type} "
        f"(topk={topk})"
    )
    return model
