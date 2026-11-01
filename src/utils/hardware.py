"""
Hardware and Compute Environment Utilities for Device Resolution and Profiling.
"""

import os
import platform
import psutil
from typing import Tuple, Optional
import torch


def resolve_device(device: Optional[str] = None) -> Tuple[torch.device, str, str]:
    """
    Resolves compute device and hardware target description.
    Returns: (torch.device, device_str, hw_target_description)
    """
    if device is not None:
        dev_str = device
    elif torch.cuda.is_available():
        dev_str = "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev_str = "mps"
    else:
        dev_str = "cpu"

    dev = torch.device(dev_str)

    if "cuda" in dev_str:
        hw_target = f"NVIDIA CUDA ({torch.cuda.get_device_name(0)})"
    elif "mps" in dev_str:
        hw_target = f"Apple Silicon MPS ({platform.machine()})"
    else:
        hw_target = f"Host CPU ({platform.processor() or platform.machine()})"

    return dev, dev_str, hw_target


def get_process_memory_gb() -> float:
    """Returns current process RAM utilization in gigabytes."""
    proc = psutil.Process(os.getpid())
    return float(proc.memory_info().rss / (1024**3))
