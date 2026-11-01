"""
run_extended_benchmarks: backward-compatible entry point delegating to src.benchmarks.extended_suite.
"""

from src.benchmarks.extended_suite import (
    compute_distinct_n,
    compute_entropy,
    evaluate_lexical_diversity,
    plot_similarity_distributions,
    benchmark_computational_budget,
    run_comparative_baselines,
    run_stratified_5fold_cv_all_datasets,
    plot_training_dynamics,
    main,
    _resolve_path,
)

__all__ = [
    "compute_distinct_n",
    "compute_entropy",
    "evaluate_lexical_diversity",
    "plot_similarity_distributions",
    "benchmark_computational_budget",
    "run_comparative_baselines",
    "run_stratified_5fold_cv_all_datasets",
    "plot_training_dynamics",
    "main",
    "_resolve_path",
]

if __name__ == "__main__":
    main()
