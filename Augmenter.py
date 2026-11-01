"""
Augmenter: backward-compatible entry point delegating to src.augmentation.augmenter.
"""

from src.augmentation.registry import (
    GENERATIVE_MODELS_REGISTRY,
    list_candidate_models,
    display_model_comparisons,
)
from src.augmentation.augmenter import IndonesianAugmenter
from src.augmentation.generative_models import benchmark_candidate_generative_models

__all__ = [
    "IndonesianAugmenter",
    "GENERATIVE_MODELS_REGISTRY",
    "list_candidate_models",
    "display_model_comparisons",
    "benchmark_candidate_generative_models",
]

if __name__ == "__main__":
    IndonesianAugmenter.display_model_comparisons()
