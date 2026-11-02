"""
Augmentation module for Indonesian Data Augmentation.
Provides similarity metrics, baseline techniques (EDA, BackTranslation),
generative model registries, and causal LM augmentation pipelines.
"""

from src.augmentation.similarity import (
    euclidean_distance,
    cos_similarity,
    jaccard_similarity,
    calculate_bleu_scores,
    distance_to_similarity,
)
from src.augmentation.baselines import (
    IndonesianEDA,
    IndonesianBackTranslator,
)
from src.augmentation.registry import (
    GENERATIVE_MODELS_REGISTRY,
    get_models_registry,
    list_candidate_models,
    display_model_comparisons,
)
from src.augmentation.generative_models import (
    benchmark_candidate_generative_models,
    compute_distinct_ngrams,
)
from src.augmentation.filtering import (
    is_hallucinatory_or_corrupt,
    clean_and_truncate_continuation,
    compute_similarity_thresholds,
    compute_dynamic_thresholds,
    check_threshold,
    build_augmented_datasets,
    filter_augmented_data,
    purge_hallucinations_from_existing_csv,
    export_augmented_datasets,
)
from src.augmentation.augmenter import (
    IndonesianAugmenter,
)

__all__ = [
    "euclidean_distance",
    "cos_similarity",
    "jaccard_similarity",
    "calculate_bleu_scores",
    "distance_to_similarity",
    "IndonesianEDA",
    "IndonesianBackTranslator",
    "GENERATIVE_MODELS_REGISTRY",
    "get_models_registry",
    "list_candidate_models",
    "display_model_comparisons",
    "benchmark_candidate_generative_models",
    "compute_distinct_ngrams",
    "is_hallucinatory_or_corrupt",
    "clean_and_truncate_continuation",
    "compute_similarity_thresholds",
    "compute_dynamic_thresholds",
    "check_threshold",
    "build_augmented_datasets",
    "filter_augmented_data",
    "purge_hallucinations_from_existing_csv",
    "export_augmented_datasets",
    "IndonesianAugmenter",
]
