"""
General utilities package for Indonesian Data Augmentation.
Contains path handling, metrics computation, hardware inspection, LaTeX compilation, and configuration loading.
"""

from src.utils.path_utils import resolve_path, ensure_dir, get_project_root
from src.utils.metrics import (
    compute_distinct_n,
    compute_entropy,
    compute_mcc,
    compute_calibration_metrics,
    extract_f1_anchors,
)
from src.utils.hardware import resolve_device, get_process_memory_gb, release_model_vram, deep_cuda_purge
from src.utils.latex import compile_latex_manuscripts
from src.utils.config import (
    load_config,
    load_dataset_config,
    load_models_config,
    load_preprocessor_config,
    load_benchmark_config,
)

__all__ = [
    "resolve_path",
    "ensure_dir",
    "get_project_root",
    "compute_distinct_n",
    "compute_entropy",
    "compute_mcc",
    "compute_calibration_metrics",
    "extract_f1_anchors",
    "resolve_device",
    "get_process_memory_gb",
    "release_model_vram",
    "deep_cuda_purge",
    "compile_latex_manuscripts",
    "load_config",
    "load_dataset_config",
    "load_models_config",
    "load_preprocessor_config",
    "load_benchmark_config",
]
