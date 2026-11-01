"""
generate_dynamic_paper_tables: backward-compatible entry point delegating to src.benchmarks.table_generator.
"""

from src.benchmarks.table_generator import (
    generate_access_dataset_stats_table,
    generate_access_main_results_table,
    generate_access_per_class_f1_table,
    generate_access_ablation_table,
    generate_tpami_comparisons_table,
    generate_tpami_5fold_table,
    generate_all_dynamic_tables,
    inject_dynamic_tables_into_manuscripts,
    compile_latex_manuscripts,
    _resolve_path,
)

__all__ = [
    "generate_access_dataset_stats_table",
    "generate_access_main_results_table",
    "generate_access_per_class_f1_table",
    "generate_access_ablation_table",
    "generate_tpami_comparisons_table",
    "generate_tpami_5fold_table",
    "generate_all_dynamic_tables",
    "inject_dynamic_tables_into_manuscripts",
    "compile_latex_manuscripts",
    "_resolve_path",
]

if __name__ == "__main__":
    generate_all_dynamic_tables()
