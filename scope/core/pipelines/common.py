"""Core common definitions for Scope pipelines."""

from enum import Enum


class Quantization(str, Enum):
    """Quantization method enumeration."""

    FP8_E4M3FN = "fp8_e4m3fn"
