"""LoRA strategy implementations.

This package contains the individual strategy implementations for different
LoRA merge modes.
"""

from pipelines.wan2_1.lora.strategies.peft_lora import PeftLoRAManager
from pipelines.wan2_1.lora.strategies.permanent_merge_lora import (
    PermanentMergeLoRAManager,
)

__all__ = [
    "PermanentMergeLoRAManager",
    "PeftLoRAManager",
]
