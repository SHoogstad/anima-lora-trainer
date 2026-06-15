"""Anima LoRA trainer: a kohya_ss-style LoRA trainer for the Anima DiT model
with native Intel XPU and CUDA support (no IPEX)."""

from .config import TrainConfig
from .device import Device, environment_report, list_devices

__version__ = "0.1.0"
__all__ = ["TrainConfig", "Device", "environment_report", "list_devices", "__version__"]
