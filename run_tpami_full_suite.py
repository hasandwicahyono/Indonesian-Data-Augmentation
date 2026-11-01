"""
run_tpami_full_suite: backward-compatible entry point delegating to src.benchmarks.tpami_suite.
"""

from src.benchmarks.tpami_suite import (
    generate_tpami_tsne_plots,
    generate_tpami_margin_and_svd_plots,
    generate_tpami_sota_longtailed_table,
    generate_tpami_calibration_analysis,
    generate_tpami_ir_sweep_and_noise_plots,
    generate_tpami_5fold_cv_table,
    generate_tpami_comparative_baselines_table,
    compute_distinct_n,
    compute_entropy,
    generate_tpami_computational_budget_table,
    generate_tpami_lexical_diversity_table,
    generate_tpami_training_dynamics_plot,
    main,
    _resolve_path,
)

__all__ = [
    "generate_tpami_tsne_plots",
    "generate_tpami_margin_and_svd_plots",
    "generate_tpami_sota_longtailed_table",
    "generate_tpami_calibration_analysis",
    "generate_tpami_ir_sweep_and_noise_plots",
    "generate_tpami_5fold_cv_table",
    "generate_tpami_comparative_baselines_table",
    "compute_distinct_n",
    "compute_entropy",
    "generate_tpami_computational_budget_table",
    "generate_tpami_lexical_diversity_table",
    "generate_tpami_training_dynamics_plot",
    "main",
    "_resolve_path",
]

if __name__ == "__main__":
    main()
