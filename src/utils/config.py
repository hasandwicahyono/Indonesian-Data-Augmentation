"""
Configuration Loader Utility for Indonesian Data Augmentation.
Loads and caches YAML configuration files from the configs/ directory.
Supports dataset hyperparameter lookups, model registries, preprocessor mappings, and benchmarks.
"""

import os
from typing import Dict, Any, Optional
import yaml

from src.utils.path_utils import resolve_path, get_project_root

_CONFIG_CACHE: Dict[str, Dict[str, Any]] = {}


def load_config(config_name_or_path: str, force_reload: bool = False) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.
    Args:
        config_name_or_path: Either a simple name ('datasets', 'models', 'preprocessor', 'benchmarks')
                             or a full/relative file path to a YAML file.
        force_reload: If True, bypasses the in-memory cache and reloads from disk.
    Returns:
        Parsed configuration dictionary.
    """
    # Normalize name to file path if simple key provided
    if not config_name_or_path.endswith((".yaml", ".yml")):
        config_path = os.path.join(get_project_root(), "configs", f"{config_name_or_path}.yaml")
    else:
        config_path = resolve_path(config_name_or_path)

    abs_path = os.path.abspath(config_path)

    if not force_reload and abs_path in _CONFIG_CACHE:
        return _CONFIG_CACHE[abs_path]

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Configuration file not found: {abs_path}")

    with open(abs_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    _CONFIG_CACHE[abs_path] = data
    return data


def load_dataset_config(dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns the dataset hyperparameters dictionary from configs/datasets.yaml.
    If dataset_name is specified, returns only that dataset's section.
    """
    cfg = load_config("datasets")
    all_datasets = cfg.get("datasets", {})
    if dataset_name is not None:
        if dataset_name not in all_datasets:
            raise KeyError(f"Dataset '{dataset_name}' not configured in configs/datasets.yaml. Available: {list(all_datasets.keys())}")
        return all_datasets[dataset_name]
    return all_datasets


def load_models_config(model_alias: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns the generative models registry dictionary from configs/models.yaml.
    If model_alias is specified, returns only that model's metadata.
    """
    cfg = load_config("models")
    all_models = cfg.get("models", {})
    if model_alias is not None:
        norm_key = model_alias.strip().lower()
        if norm_key in all_models:
            return all_models[norm_key]
        for k, meta in all_models.items():
            if meta.get("model_id", "").lower() == norm_key:
                return meta
        raise KeyError(f"Model alias '{model_alias}' not found in configs/models.yaml. Available: {list(all_models.keys())}")
    return all_models


def load_preprocessor_config() -> Dict[str, Dict[str, str]]:
    """Returns the preprocessor dictionaries from configs/preprocessor.yaml."""
    return load_config("preprocessor")


def load_benchmark_config() -> Dict[str, Any]:
    """Returns the benchmark settings and prompts from configs/benchmarks.yaml."""
    return load_config("benchmarks")

