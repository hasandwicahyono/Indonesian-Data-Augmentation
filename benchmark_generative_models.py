"""
benchmark_generative_models: backward-compatible entry point delegating to src.augmentation.generative_models.
"""

from src.augmentation.generative_models import (
    benchmark_candidate_generative_models,
    compute_distinct_ngrams,
    BENCHMARK_PROMPTS,
)

__all__ = [
    "benchmark_candidate_generative_models",
    "compute_distinct_ngrams",
    "BENCHMARK_PROMPTS",
]

if __name__ == "__main__":
    benchmark_candidate_generative_models()
