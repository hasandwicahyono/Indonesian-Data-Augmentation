"""
Benchmarks module for Indonesian Data Augmentation.
Contains cross-validation evaluators, extended benchmarking suites,
TPAMI advanced ML diagnostics, and dynamic LaTeX/CSV table generators.
"""

from src.benchmarks.cross_validation import run_real_5fold_cv_for_dataset
from src.benchmarks.table_generator import (
    generate_access_dataset_stats_table,
    generate_access_main_results_table,
    generate_access_per_class_f1_table,
    generate_access_ablation_table,
    generate_tpami_comparisons_table,
    generate_tpami_5fold_table,
    generate_all_dynamic_tables,
    inject_dynamic_tables_into_manuscripts,
)
from src.benchmarks.extended_suite import (
    evaluate_lexical_diversity,
    plot_similarity_distributions,
    benchmark_computational_budget,
    run_comparative_baselines,
    run_stratified_5fold_cv_all_datasets,
    plot_training_dynamics,
)
from src.benchmarks.tpami_suite import (
    generate_tpami_tsne_plots,
    generate_tpami_margin_and_svd_plots,
    generate_tpami_sota_longtailed_table,
    generate_tpami_calibration_analysis,
    generate_tpami_ir_sweep_and_noise_plots,
    generate_tpami_5fold_cv_table,
    generate_tpami_comparative_baselines_table,
    generate_tpami_computational_budget_table,
    generate_tpami_lexical_diversity_table,
    generate_tpami_training_dynamics_plot,
)

__all__ = [
    "run_real_5fold_cv_for_dataset",
    "generate_access_dataset_stats_table",
    "generate_access_main_results_table",
    "generate_access_per_class_f1_table",
    "generate_access_ablation_table",
    "generate_tpami_comparisons_table",
    "generate_tpami_5fold_table",
    "generate_all_dynamic_tables",
    "inject_dynamic_tables_into_manuscripts",
    "evaluate_lexical_diversity",
    "plot_similarity_distributions",
    "benchmark_computational_budget",
    "run_comparative_baselines",
    "run_stratified_5fold_cv_all_datasets",
    "plot_training_dynamics",
    "generate_tpami_tsne_plots",
    "generate_tpami_margin_and_svd_plots",
    "generate_tpami_sota_longtailed_table",
    "generate_tpami_calibration_analysis",
    "generate_tpami_ir_sweep_and_noise_plots",
    "generate_tpami_5fold_cv_table",
    "generate_tpami_comparative_baselines_table",
    "generate_tpami_computational_budget_table",
    "generate_tpami_lexical_diversity_table",
    "generate_tpami_training_dynamics_plot",
]
