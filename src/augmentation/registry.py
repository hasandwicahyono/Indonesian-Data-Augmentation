"""
Generative Backbone Models Registry for Indonesian Text Augmentation.
Loads model configurations dynamically from configs/models.yaml.
"""

from typing import Dict, Any, List
from src.utils.config import load_models_config


def get_models_registry() -> Dict[str, Dict[str, Any]]:
    """Loads and returns the generative models registry dictionary."""
    return load_models_config()


# Backward-compatible global registry dictionary loaded dynamically from config
GENERATIVE_MODELS_REGISTRY: Dict[str, Dict[str, Any]] = get_models_registry()


def list_candidate_models() -> List[Dict[str, Any]]:
    """Returns metadata of all candidate generative models in the registry."""
    return list(load_models_config().values())


def display_model_comparisons() -> None:
    """Prints a structured comparison table of First-Gen vs. Contemporary models."""
    registry = load_models_config()
    print("=" * 105)
    print(f"{'Alias':<14} | {'Release':<7} | {'Generation':<26} | {'Params':<7} | {'Architecture':<22} | {'HuggingFace ID'}")
    print("-" * 105)
    for alias, info in registry.items():
        print(f"{alias:<14} | {info.get('release_year', 'N/A'):<7} | {info.get('generation', 'N/A'):<26} | {info.get('parameters', 'N/A'):<7} | {info.get('architecture', 'N/A'):<22} | {info.get('model_id', '')}")
    print("=" * 105)
