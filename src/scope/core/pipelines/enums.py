"""Enum definitions for pipeline configuration.

This module contains enums used across pipeline configurations.
It is intentionally kept free of torch imports to allow import
by server modules without triggering torch DLL loading on Windows.
"""

from enum import Enum


class Quantization(str, Enum):
    """Quantization method enumeration."""

    FP8_E4M3FN = "fp8_e4m3fn"


class VaeType(str, Enum):
    """VAE type enumeration."""

    WAN = "wan"
    LIGHTVAE = "lightvae"
    TAE = "tae"
    LIGHTTAE = "lighttae"
