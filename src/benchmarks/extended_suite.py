"""
Legacy Extended Benchmarks Suite Compatibility Shim.
All benchmarking functionality has been consolidated into:
- src.benchmarks.tpami_suite (Analytical suites, diagnostic plots, diversity, dynamics)
- src.benchmarks.cross_validation (Real stratified 5-fold cross-validation)
"""

from src.benchmarks.tpami_suite import (
    plot_similarity_distributions,
    generate_tpami_lexical_diversity_table as evaluate_lexical_diversity,
    generate_tpami_computational_budget_table as benchmark_computational_budget,
    generate_tpami_comparative_baselines_table as run_comparative_baselines,
    generate_tpami_5fold_cv_table as run_stratified_5fold_cv_all_datasets,
    generate_tpami_training_dynamics_plot as plot_training_dynamics,
)

__all__ = [
    "plot_similarity_distributions",
    "evaluate_lexical_diversity",
    "benchmark_computational_budget",
    "run_comparative_baselines",
    "run_stratified_5fold_cv_all_datasets",
    "plot_training_dynamics",
]
