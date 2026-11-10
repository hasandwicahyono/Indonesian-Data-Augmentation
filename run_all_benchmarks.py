"""
Unified Benchmark Investigation and Visualization Pipeline.
Runs and visualizes the complete augmentation and classification pipeline across all 4 Indonesian datasets:
1. SmSA (doc-sentiment-prosa)
2. HoASA (absa-airy)
3. EmoT (emotion-twitter)
4. NusaX-Indonesian (nusax-sentiment-indonesian)

Features:
- Executes IndoBERT training and evaluation across 5 variants:
  * Baseline (Not-Augmented)
  * ECU-Augmented (Euclidean similarity)
  * COS-Augmented (Cosine similarity)
  * JAC-Augmented (Jaccard similarity)
  * BLEU-Augmented (BLEU score)
- Automatically saves prediction arrays (y_true, y_pred) to NPZ for auditability.
- Generates publication-ready visualizations:
  * 4-metric comparison bar charts (Accuracy, Precision, Recall, Macro-F1) for each dataset.
  * Side-by-side Confusion Matrix heatmaps (Baseline vs. Best-Augmented) for each dataset.
  * Multi-panel cross-dataset benchmark summary dashboard.
- Supports '--plot_only' flag to render or update all visual plots instantly from existing results.
"""

import os
import sys
import warnings

# Silence benign PyTorch DataParallel multi-GPU scalar gathering warning
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.parallel.scatter_gather")
warnings.filterwarnings("ignore", message=".*Was asked to gather along dimension 0.*")

# Ensure KMP duplicate lib is allowed on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Configure CUDA caching allocator to minimize fragmentation and prevent OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Ensure local modules are found
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import argparse
import copy
import gc
import json
import shutil
import torch
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional

from src.data import IndonesianDatasetsManager, CustomDataset
from src.augmentation import IndonesianAugmenter
from src.augmentation.filtering import build_augmented_datasets, export_augmented_datasets
from src.modeling import IndoBERTClassifierEvaluator, AutoImbalanceCalibrator
from src.utils.config import get_investigated_datasets, get_investigated_settings
from src.benchmarks import (
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
    run_real_5fold_cv_for_dataset,
    plot_similarity_distributions,
    generate_all_dynamic_tables,
)
from src.utils.config import load_dataset_config


# =====================================================================
# Dataset Specific Optimal Parameter Configurations (Loaded from configs/datasets.yaml)
# =====================================================================
DATASET_CONFIGS = load_dataset_config()


# =====================================================================
# Dataset and File Helpers
# =====================================================================
def load_dataset_data(manager: IndonesianDatasetsManager, dataset_name: str) -> CustomDataset:
    """Loads dataset partitions using IndonesianDatasetsManager."""
    if dataset_name == "SmSA":
        return manager.load_smsa()
    elif dataset_name == "MovieOpini":
        return manager.load_movie_opini()
    elif dataset_name == "HoASA":
        return manager.load_hoasa(aspect_target="general")
    elif dataset_name == "EmoT":
        return manager.load_emot()
    elif dataset_name in ["NusaX", "NusaX-Indonesian"]:
        # Standard 3-class curated benchmark (500 train, 100 val, 400 test)
        return manager.load_nusax_indonesian(binary_only=False)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


def load_augmented_datasets(
    dataset_name: str, augmented_dir: str = "data/augmented/all_text"
) -> Dict[str, pd.DataFrame]:
    """Loads all 4 filtered augmented datasets (ECU, COS, JAC, BLEU)."""
    metrics = ["ECU", "COS", "JAC", "BLEU"]
    aug_dfs = {}
    resolved_dir = os.path.abspath(augmented_dir)
    if not os.path.exists(resolved_dir):
        # Fallback search from project root or current dir
        cand = os.path.join(CURRENT_DIR, augmented_dir)
        if os.path.exists(cand):
            resolved_dir = cand

    for m in metrics:
        fname = f"{dataset_name}-Augmented-{m}-ALL-Text-Final.csv"
        fpath = os.path.join(resolved_dir, fname)
        if os.path.exists(fpath):
            aug_dfs[m] = pd.read_csv(fpath)
            print(f"  Loaded {m}-Augmented data from {fpath} ({len(aug_dfs[m])} rows)")
        else:
            print(f"  [Notice] Augmented file not found: {fpath}")

    return aug_dfs


def find_existing_results_file(dataset_name: str, results_dir: str) -> Optional[str]:
    """Searches for existing evaluation Excel file for a dataset."""
    candidates = [
        os.path.join(results_dir, f"{dataset_name}_evaluation_results.xlsx"),
        os.path.join(results_dir, f"{dataset_name}_evaluation_results_1.xlsx"),
        os.path.join(CURRENT_DIR, "data", "results", f"{dataset_name}_evaluation_results.xlsx"),
        os.path.join(CURRENT_DIR, "data", "results", f"{dataset_name}_evaluation_results_1.xlsx"),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return None


def clean_variant_name(raw_name: str) -> str:
    """Standardizes variant names into readable labels."""
    raw = str(raw_name).strip()
    if "Raw" in raw or "ERM" in raw:
        return "Raw-ERM"
    elif "Baseline" in raw or "NotAugmented" in raw:
        return "Baseline"
    elif "Ensemble" in raw:
        return "Ensemble"
    elif "ECU" in raw:
        return "ECU"
    elif "COS" in raw:
        return "COS"
    elif "JAC" in raw:
        return "JAC"
    elif "BLEU" in raw:
        return "BLEU"
    return raw


# =====================================================================
# Dataset Statistics Computation and Persistence
# =====================================================================
def compute_and_save_dataset_statistics(
    manager: Optional[IndonesianDatasetsManager] = None,
    results_dir: str = "data/results",
    extended_results_dir: str = "data/results/extended",
) -> pd.DataFrame:
    """Computes exact dataset partitions, class distributions, and IR from raw data and saves to file."""
    if manager is None:
        manager = IndonesianDatasetsManager()

    investigated = get_investigated_datasets()
    datasets = []
    for dname in investigated:
        setting = get_investigated_settings(dname)
        method_name = setting.get("loader_method", f"load_{dname.lower().replace('-', '_')}")
        domain = setting.get("domain", "Sentiment Analysis")
        loader_fn = getattr(manager, method_name) if hasattr(manager, method_name) else manager.load_nusax_indonesian
        datasets.append((dname, domain, loader_fn))

    records = []
    for name, domain, load_fn in datasets:
        ds = load_fn()
        num_classes = len(ds.label_list)
        train_len = len(ds.train)
        val_len = len(ds.val)
        test_len = len(ds.test)

        counts = ds.train["label"].value_counts()
        max_c = counts.max()
        min_c = counts.min()
        ir = round(max_c / max(1, min_c), 2)
        ir_desc = "Balanced" if ir <= 1.2 else ("Mild" if ir <= 3.0 else ("Moderate" if ir <= 10.0 else "Extreme"))

        dist_str_csv = ", ".join([f"{lbl}: {c} ({(c/train_len)*100:.1f}%)" for lbl, c in counts.items()])

        records.append({
            "Dataset": name,
            "Domain / Task": domain,
            "Classes (K)": num_classes,
            "Train Size": train_len,
            "Val Size": val_len,
            "Test Size": test_len,
            "Label Distribution (Train)": dist_str_csv,
            "Imbalance Ratio (IR)": f"{ir}x ({ir_desc})",
        })

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(extended_results_dir, exist_ok=True)

    json_path = os.path.join(results_dir, "dataset_statistics.json")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    ext_json_path = os.path.join(extended_results_dir, "dataset_statistics.json")
    with open(ext_json_path, "w") as f:
        json.dump(records, f, indent=2)

    df = pd.DataFrame(records)
    csv_path = os.path.join(extended_results_dir, "access_table_dataset_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"  [Dataset Stats] Saved {len(records)} dataset profiles to:\n    - {json_path}\n    - {csv_path}")
    return df


# =====================================================================
# Visualization 1: 4-Metric Performance Comparison per Dataset
# =====================================================================
def plot_dataset_metric_comparison(
    results_df: pd.DataFrame,
    dataset_name: str,
    output_dir: str = "data/results/figures",
) -> str:
    """
    Creates a publication-ready grouped bar chart for a dataset
    comparing Baseline, ECU, COS, JAC, and BLEU across:
    - Accuracy
    - Macro-Precision
    - Macro-Recall
    - Macro-F1
    """
    os.makedirs(output_dir, exist_ok=True)
    df = results_df.copy()
    df["variant"] = df["dataset_name"].apply(clean_variant_name)

    # Standardize column names
    col_mapping = {
        "accuracy": "Accuracy",
        "macro_precision": "Macro-Precision",
        "macro_recall": "Macro-Recall",
        "macro_f1": "Macro-F1",
    }
    for old_col, new_col in col_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]

    metrics = ["Accuracy", "Macro-Precision", "Macro-Recall", "Macro-F1"]
    available_metrics = [m for m in metrics if m in df.columns]

    # Ensure standard variant order
    variant_order = ["Raw-ERM", "Baseline", "ECU", "COS", "JAC", "BLEU", "Ensemble"]
    ordered_variants = [v for v in variant_order if v in df["variant"].values]
    if not ordered_variants:
        ordered_variants = df["variant"].tolist()

    df_plot = df.set_index("variant").reindex(ordered_variants).reset_index()

    # Plot settings
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

    x = np.arange(len(ordered_variants))
    num_metrics = len(available_metrics)
    width = 0.8 / num_metrics

    # Modern color palette
    palette = ["#2b5c8f", "#3a86c8", "#5aa469", "#d95f02"]

    for idx, (metric, color) in enumerate(zip(available_metrics, palette)):
        offset = (idx - (num_metrics - 1) / 2) * width
        values = df_plot[metric].values
        bars = ax.bar(x + offset, values, width, label=metric, color=color, edgecolor="black", linewidth=0.6, alpha=0.92)

        # Annotate numeric values above bars
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.annotate(
                    f"{val:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="semibold",
                    rotation=0,
                )

    # Calculate and annotate best improvement on Macro-F1
    if "Macro-F1" in df_plot.columns and len(df_plot) > 1:
        base_rows = df_plot[df_plot["variant"].isin(["Baseline", "Raw-ERM"])]
        aug_rows = df_plot[df_plot["variant"].isin(["ECU", "COS", "JAC", "BLEU", "Ensemble"])]
        if not base_rows.empty and not aug_rows.empty:
            base_f1_val = base_rows.iloc[0]["Macro-F1"]
            if base_f1_val > 0:
                best_aug_idx = aug_rows["Macro-F1"].idxmax()
                best_var = df_plot.loc[best_aug_idx, "variant"]
                best_f1 = df_plot.loc[best_aug_idx, "Macro-F1"]
                gain = best_f1 - base_f1_val
                pct_gain = (gain / base_f1_val) * 100.0
                sign = "+" if gain >= 0 else ""
                ax.text(
                    0.02,
                    0.96,
                    f"Best Variant ({best_var}): Macro-F1 {best_f1:.4f} ({sign}{gain:.4f} | {sign}{pct_gain:.2f}%)",
                    transform=ax.transAxes,
                    fontsize=10.5,
                    fontweight="bold",
                    color="#1a5276",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#eaf2f8", edgecolor="#aed6f1", alpha=0.95),
                )

    ax.set_title(f"{dataset_name} — Comprehensive Classification Performance Comparison", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Data Augmentation Technique", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Score (0.0 – 1.0)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_variants, fontsize=10.5, fontweight="semibold")
    ax.set_ylim(0.0, 1.08)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=9.5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    png_path = os.path.join(output_dir, f"{dataset_name}_metrics_comparison.png")
    pdf_path = os.path.join(output_dir, f"{dataset_name}_metrics_comparison.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close(fig)
    print(f"  [Figure Saved] {png_path}")
    return png_path


# =====================================================================
# Visualization 2: Confusion Matrix Heatmaps per Dataset
# =====================================================================
def plot_confusion_matrices(
    y_true_base: np.ndarray,
    y_pred_base: np.ndarray,
    y_true_aug: np.ndarray,
    y_pred_aug: np.ndarray,
    label_list: List[str],
    dataset_name: str,
    aug_variant_name: str = "Best Augmented",
    output_dir: str = "data/results/figures",
) -> str:
    """
    Plots side-by-side Confusion Matrices comparing Baseline vs. Best Augmented.
    Displays raw counts and row-normalized percentages in each cell.
    """
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

    os.makedirs(output_dir, exist_ok=True)
    num_labels = len(label_list)
    target_indices = list(range(num_labels))

    # Compute confusion matrices
    cm_base = confusion_matrix(y_true_base, y_pred_base, labels=target_indices)
    cm_aug = confusion_matrix(y_true_aug, y_pred_aug, labels=target_indices)

    # Compute row-normalized percentages (safe div by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_base_norm = np.nan_to_num(cm_base.astype(float) / cm_base.sum(axis=1)[:, np.newaxis])
        cm_aug_norm = np.nan_to_num(cm_aug.astype(float) / cm_aug.sum(axis=1)[:, np.newaxis])

    # Compute summary metrics
    acc_base = accuracy_score(y_true_base, y_pred_base)
    f1_base = f1_score(y_true_base, y_pred_base, labels=target_indices, average="macro", zero_division=0)
    acc_aug = accuracy_score(y_true_aug, y_pred_aug)
    f1_aug = f1_score(y_true_aug, y_pred_aug, labels=target_indices, average="macro", zero_division=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Cell annotation formatting: "count \n (percentage)"
    def make_annot(cm, cm_norm):
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"
        return annot

    annot_base = make_annot(cm_base, cm_base_norm)
    annot_aug = make_annot(cm_aug, cm_aug_norm)

    # Baseline Heatmap
    sns.heatmap(
        cm_base_norm,
        annot=annot_base,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=label_list,
        yticklabels=label_list,
        ax=axes[0],
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="#bdc3c7",
    )
    axes[0].set_title(f"Baseline (Not-Augmented)\nAccuracy: {acc_base:.3f} | Macro-F1: {f1_base:.3f}", fontsize=11.5, fontweight="bold", pad=10)
    axes[0].set_xlabel("Predicted Class", fontsize=10.5, fontweight="semibold")
    axes[0].set_ylabel("True Ground-Truth Class", fontsize=10.5, fontweight="semibold")

    # Augmented Heatmap
    sns.heatmap(
        cm_aug_norm,
        annot=annot_aug,
        fmt="",
        cmap="Greens",
        cbar=True,
        xticklabels=label_list,
        yticklabels=label_list,
        ax=axes[1],
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="#bdc3c7",
    )
    axes[1].set_title(f"{aug_variant_name}\nAccuracy: {acc_aug:.3f} | Macro-F1: {f1_aug:.3f}", fontsize=11.5, fontweight="bold", pad=10)
    axes[1].set_xlabel("Predicted Class", fontsize=10.5, fontweight="semibold")
    axes[1].set_ylabel("True Ground-Truth Class", fontsize=10.5, fontweight="semibold")

    plt.suptitle(f"{dataset_name} — Test Set Confusion Matrix Comparison", fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()

    png_path = os.path.join(output_dir, f"{dataset_name}_confusion_matrices.png")
    pdf_path = os.path.join(output_dir, f"{dataset_name}_confusion_matrices.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Figure Saved] {png_path}")
    return png_path


# =====================================================================
# Visualization 3: Cross-Dataset Multi-Panel Summary Dashboard
# =====================================================================
def plot_cross_dataset_summary(
    all_results: Dict[str, pd.DataFrame],
    output_dir: str = "data/results/figures",
) -> str:
    """
    Builds a 4-panel comprehensive dashboard displaying performance
    comparisons across all 4 datasets (SmSA, HoASA, EmoT, NusaX-Indonesian).
    """
    os.makedirs(output_dir, exist_ok=True)
    datasets = [d for d in get_investigated_datasets() if d in all_results]
    if not datasets:
        return ""

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)
    axes = axes.flatten()

    metrics = ["Accuracy", "Macro-Precision", "Macro-Recall", "Macro-F1"]
    col_mapping = {
        "accuracy": "Accuracy",
        "macro_precision": "Macro-Precision",
        "macro_recall": "Macro-Recall",
        "macro_f1": "Macro-F1",
    }
    palette = ["#2b5c8f", "#3a86c8", "#5aa469", "#d95f02"]

    for i, ds_name in enumerate(datasets):
        ax = axes[i]
        df = all_results[ds_name].copy()
        df["variant"] = df["dataset_name"].apply(clean_variant_name)

        for old_col, new_col in col_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]

        variant_order = ["Raw-ERM", "Baseline", "ECU", "COS", "JAC", "BLEU", "Ensemble"]
        ordered_vars = [v for v in variant_order if v in df["variant"].values]
        if not ordered_vars:
            ordered_vars = df["variant"].tolist()

        df_plot = df.set_index("variant").reindex(ordered_vars).reset_index()
        x = np.arange(len(ordered_vars))
        width = 0.8 / len(metrics)

        for idx, (m, col) in enumerate(zip(metrics, palette)):
            if m in df_plot.columns:
                offset = (idx - (len(metrics) - 1) / 2) * width
                vals = df_plot[m].values
                ax.bar(x + offset, vals, width, label=m if i == 0 else "", color=col, edgecolor="black", linewidth=0.5, alpha=0.92)

        # Baseline vs Best F1 text box
        if "Macro-F1" in df_plot.columns:
            base_rows = df_plot[df_plot["variant"].isin(["Baseline", "Raw-ERM"])]
            aug_rows = df_plot[df_plot["variant"].isin(["ECU", "COS", "JAC", "BLEU", "Ensemble"])]
            if not base_rows.empty and not aug_rows.empty:
                base_f1_val = base_rows.iloc[0]["Macro-F1"]
                best_idx = aug_rows["Macro-F1"].idxmax()
                best_v = df_plot.loc[best_idx, "variant"]
                best_f = df_plot.loc[best_idx, "Macro-F1"]
                if base_f1_val > 0:
                    gain_pct = ((best_f - base_f1_val) / base_f1_val) * 100.0
                    sign = "+" if gain_pct >= 0 else ""
                    ax.text(
                        0.03,
                        0.94,
                        f"Best ({best_v}): F1 {best_f:.3f} ({sign}{gain_pct:.1f}%)",
                        transform=ax.transAxes,
                        fontsize=9,
                        fontweight="bold",
                        color="#154360",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ebf5fb", edgecolor="#aed6f1", alpha=0.9),
                    )

        ax.set_title(f"({chr(65+i)}) {ds_name} Dataset", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(ordered_vars, fontsize=9.5, fontweight="semibold")
        ax.set_ylim(0.0, 1.08)
        ax.set_ylabel("Score", fontsize=10, fontweight="semibold")
        ax.grid(axis="y", linestyle="--", alpha=0.45)

    for j in range(len(datasets), len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=len(labels), frameon=True, prop={"weight": "semibold", "size": 11})
    else:
        fig.legend(metrics, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=len(metrics), frameon=True, prop={"weight": "semibold", "size": 11})

    plt.suptitle("Indonesian NLP Augmentation Benchmark — Cross-Dataset Performance Dashboard", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = os.path.join(output_dir, "all_datasets_benchmark_comparison.png")
    out_pdf = os.path.join(output_dir, "all_datasets_benchmark_comparison.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Dashboard Saved] {out_png}")
    return out_png


# =====================================================================
# TPAMI and IEEE Access Comprehensive Diagnostic Suite
# =====================================================================
def run_all_tpami_access_suite(
    target_datasets: Optional[List[str]] = None,
    figures_dir: str = "data/figures",
    results_dir: str = "data/results/extended",
    run_cv: bool = False,
    cv_epochs: int = 2,
    device: Optional[str] = None,
    paper_dir: str = "paper",
) -> Dict[str, Any]:
    """
    Executes the comprehensive IEEE TPAMI & IEEE Access evaluation suite:
    1. Real Stratified 5-Fold Cross Validation with Paired t-Test (No hard-coded values).
    2. Comparative Baselines (EDA & Back-Translation vs. Proposed).
    3. Computational Budget & Runtime Efficiency Profiling Table.
    4. Lexical Diversity (Distinct-1, Distinct-2, Entropy).
    5. Latent Representation Space Analysis (2D t-SNE Projections).
    6. Empirical Margin Distribution CDFs & Singular Value Spectrum (Neural Collapse audit).
    7. Expected Calibration Error (ECE) & Reliability Diagrams.
    8. Controlled Imbalance Ratio (IR: 1x to 200x) Sweep & Label Noise Robustness.
    9. SOTA Long-Tailed Learning Paradigms Benchmarking.
    10. Similarity Score Distribution Histograms (ECU, COS, JAC, BLEU with cutoffs).
    11. Training Dynamics & Convergence Trajectories.
    12. Synchronizes all generated figures to the manuscript directory (paper/).
    """
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 85)
    print("EXECUTING EXTENDED IEEE TPAMI & ACCESS DIAGNOSTIC BENCHMARK SUITE")
    print("=" * 85)

    suite_results = {}

    # 1. Real 5-Fold Cross Validation
    if run_cv and target_datasets:
        print("\n>>> [TPAMI / Access Test 1] Running Real 5-Fold Cross Validation Experiments...")
        mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        chosen_dev = device if device else ("mps" if mps_avail else "cpu")
        for ds in target_datasets:
            cv_res = run_real_5fold_cv_for_dataset(
                dataset_name=ds,
                epochs=cv_epochs,
                device=chosen_dev,
            )
            suite_results[f"{ds}_cv"] = cv_res

    # 2. 5-Fold CV Paired t-Test Table
    print("\n>>> [TPAMI / Access Test 2] Generating Stratified 5-Fold CV Paired t-Test Table...")
    cv_table_path = os.path.join(results_dir, "tpami_stratified_5fold_cv_paired_ttest.csv")
    cv_df = generate_tpami_5fold_cv_table(output_csv=cv_table_path)
    suite_results["5fold_cv_table"] = cv_df

    # 3. Comparative Baselines Table (EDA & Back-Translation)
    print("\n>>> [TPAMI / Access Test 3] Generating Comparative Baselines Table (EDA & MarianMT Back-Translation)...")
    eda_df = generate_tpami_comparative_baselines_table()
    suite_results["comparative_baselines"] = eda_df

    # 4. Computational Budget & Runtime Efficiency Table
    print("\n>>> [TPAMI / Access Test 4] Profiling Computational Budget and Runtime Efficiency...")
    comp_df = generate_tpami_computational_budget_table()
    suite_results["computational_budget"] = comp_df

    # 5. Lexical Diversity Table (Distinct-1, Distinct-2, Entropy)
    print("\n>>> [TPAMI / Access Test 5] Evaluating Lexical Diversity (Distinct-1 & Distinct-2)...")
    lex_df = generate_tpami_lexical_diversity_table()
    suite_results["lexical_diversity"] = lex_df

    # 6. Benchmark Against SOTA Long-Tailed Paradigms
    print("\n>>> [TPAMI / Access Test 6] Benchmarking Against SOTA Long-Tailed Paradigms...")
    sota_df = generate_tpami_sota_longtailed_table()
    suite_results["sota_longtailed"] = sota_df

    # 7. Probability Calibration & Reliability Diagrams (ECE / MCE)
    print("\n>>> [TPAMI / Access Test 7] Computing Expected Calibration Error (ECE) and Reliability Diagrams...")
    cal_png = os.path.join(figures_dir, "tpami_reliability_diagrams_ece.png")
    cal_pdf = os.path.join(figures_dir, "tpami_reliability_diagrams_ece.pdf")
    cal_df = generate_tpami_calibration_analysis(output_png=cal_png, output_pdf=cal_pdf)
    suite_results["calibration"] = cal_df

    # 8. 2D t-SNE Latent Feature Geometry
    print("\n>>> [TPAMI / Access Test 8] Generating 2D t-SNE Latent Geometry Projections...")
    tsne_png = os.path.join(figures_dir, "tpami_tsne_latent_geometry.png")
    tsne_pdf = os.path.join(figures_dir, "tpami_tsne_latent_geometry.pdf")
    generate_tpami_tsne_plots(output_png=tsne_png, output_pdf=tsne_pdf)

    # 9. Margin CDF & Singular Value Spectrum (Neural Collapse)
    print("\n>>> [TPAMI / Access Test 9] Generating Margin CDF and Singular Value Spectrum...")
    margin_png = os.path.join(figures_dir, "tpami_margin_cdf_and_singular_spectrum.png")
    margin_pdf = os.path.join(figures_dir, "tpami_margin_cdf_and_singular_spectrum.pdf")
    generate_tpami_margin_and_svd_plots(output_png=margin_png, output_pdf=margin_pdf)

    # 10. Imbalance Ratio (IR) Sweep & Noise Robustness
    print("\n>>> [TPAMI / Access Test 10] Generating Imbalance Ratio Sweep and Noise Robustness Plots...")
    ir_png = os.path.join(figures_dir, "tpami_imbalance_sweep_and_noise_stress.png")
    ir_pdf = os.path.join(figures_dir, "tpami_imbalance_sweep_and_noise_stress.pdf")
    generate_tpami_ir_sweep_and_noise_plots(output_png=ir_png, output_pdf=ir_pdf)

    # 11. Similarity Score Distribution Histograms
    print("\n>>> [TPAMI / Access Test 11] Plotting Similarity Distribution Histograms with Threshold Cutoffs...")
    sim_png = os.path.join(figures_dir, "similarity_distribution_histograms.png")
    sim_pdf = os.path.join(figures_dir, "similarity_distribution_histograms.pdf")
    plot_similarity_distributions(dataset_name="HoASA", output_png=sim_png, output_pdf=sim_pdf)

    # 12. Training Dynamics & Convergence Curves
    print("\n>>> [TPAMI / Access Test 12] Generating Training Dynamics & Convergence Curves...")
    dyn_png = os.path.join(figures_dir, "training_dynamics_convergence_curves.png")
    dyn_pdf = os.path.join(figures_dir, "training_dynamics_convergence_curves.pdf")
    generate_tpami_training_dynamics_plot(output_png=dyn_png, output_pdf=dyn_pdf)

    # 13. Generative Backbone Models Benchmark & Comparison (First-Gen vs Contemporary)
    print("\n>>> [TPAMI / Access Test 13] Benchmarking and Comparing Generative Backbone Models...")
    try:
        from src.augmentation.generative_models import benchmark_candidate_generative_models
        gen_df = benchmark_candidate_generative_models(
            output_csv=os.path.join(results_dir, "generative_models_comparison.csv"),
            output_tex=os.path.join(paper_dir, "table_generative_models_comparison.tex"),
            force=False,
        )
        suite_results["generative_models_comparison"] = gen_df
    except Exception as e:
        print(f"  [Notice] Generative models benchmarking deferred: {e}")

    # 14. Dynamic Manuscript Tables for IEEE Access and IEEE TPAMI
    print("\n>>> [TPAMI / Access Test 14] Generating All Dynamic Tables and Compiling Manuscripts...")
    from src.benchmarks.table_generator import generate_all_dynamic_tables
    dyn_tables = generate_all_dynamic_tables(
        results_dir="data/results",
        extended_results_dir=results_dir,
        paper_dir=paper_dir,
        compile_pdf=True,
    )
    suite_results.update(dyn_tables)

    # 15. Synchronize figures to paper/ directory for manuscript compilation
    if os.path.exists(paper_dir):
        print(f"\n>>> Synchronizing all generated figures to manuscript folder '{paper_dir}'...")
        fig_files = [
            "tpami_tsne_latent_geometry.png", "tpami_tsne_latent_geometry.pdf",
            "tpami_margin_cdf_and_singular_spectrum.png", "tpami_margin_cdf_and_singular_spectrum.pdf",
            "tpami_reliability_diagrams_ece.png", "tpami_reliability_diagrams_ece.pdf",
            "tpami_imbalance_sweep_and_noise_stress.png", "tpami_imbalance_sweep_and_noise_stress.pdf",
            "similarity_distribution_histograms.png", "similarity_distribution_histograms.pdf",
            "training_dynamics_convergence_curves.png", "training_dynamics_convergence_curves.pdf",
        ]
        for f in fig_files:
            src = os.path.join(figures_dir, f)
            if os.path.exists(src):
                dst = os.path.join(paper_dir, f)
                shutil.copy2(src, dst)
        print("  [Synced] All TPAMI & Access figures copied to paper/")

    print("\n" + "=" * 85)
    print("IEEE TPAMI & ACCESS EXTENDED SUITE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"All extended tables saved to: {os.path.abspath(results_dir)}")
    print(f"All extended figures saved to: {os.path.abspath(figures_dir)}")
    print("=" * 85)
    return suite_results


# =====================================================================
# Main Benchmark Pipeline Runner
# =====================================================================
def run_benchmark_pipeline(
    target_datasets: List[str],
    plot_only: bool = False,
    run_generation: bool = False,
    force_generation: bool = False,
    clean_gpu: bool = False,
    use_adafactor: bool = False,
    gen_model: str = "cahya/gpt2-small-indonesian-522M",
    generation_sample_limit: Optional[int] = None,
    results_dir: str = "data/results",
    figures_dir: str = "data/results/figures",
    augmented_dir: str = "data/augmented/all_text",
    device: Optional[str] = None,
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    seed: int = 42,
    run_ensemble: bool = True,
    auto_calibrate: bool = True,
    all_tpami_access: bool = True,
    run_tpami_suite: bool = False,
    run_cv: bool = False,
    cv_epochs: int = 2,
    run_comparative_baselines: bool = False,
    run_computational_budget: bool = False,
    run_lexical_diversity: bool = False,
    run_training_dynamics: bool = False,
    run_similarity_distributions: bool = False,
):
    """Executes the full evaluation pipeline and renders all visualizations."""
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    manager = IndonesianDatasetsManager()

    # -------------------------------------------------------------
    # Step 0: Compute and Persist Benchmark Dataset Statistics
    # -------------------------------------------------------------
    print("\n>>> [STEP 0] Computing and saving benchmark dataset statistics to persistent files...")
    compute_and_save_dataset_statistics(
        manager=manager,
        results_dir=results_dir,
        extended_results_dir="data/results/extended",
    )

    mode_desc = "PLOT-ONLY (Visualizations from existing results)" if plot_only else (
        "FULL END-TO-END PIPELINE (Phase 1 Generation + Phase 2 Filtering + Phase 3 Training + Phase 4 Visualizations)"
        if run_generation else "BENCHMARK EVALUATION (Phase 3 Training + Phase 4 Visualizations)"
    )

    print("=" * 80)
    print("INDONESIAN DATA AUGMENTATION BENCHMARK INVESTIGATION PIPELINE")
    print(f"Target Datasets: {target_datasets}")
    print(f"Mode: {mode_desc}")
    print(f"Results Directory: {results_dir}")
    print(f"Figures Directory: {figures_dir}")
    print("=" * 80)

    from src.utils.hardware import inspect_gpu_processes, clean_stale_gpu_processes
    if clean_gpu:
        clean_stale_gpu_processes(auto_kill=True)
    inspect_gpu_processes()

    if use_adafactor:
        os.environ["USE_ADAFACTOR"] = "1"

    all_results: Dict[str, pd.DataFrame] = {}

    for ds_name in target_datasets:
        print(f"\n[{ds_name}] Processing Investigation...")
        cfg = copy.deepcopy(DATASET_CONFIGS[ds_name])
        ds = load_dataset_data(manager, ds_name)

        # Dynamic Auto-Calibration of Imbalance Parameters
        if auto_calibrate:
            batch_size_probe = batch_size_override if batch_size_override is not None else cfg["batch_size"]
            cal = AutoImbalanceCalibrator.calibrate(
                train_df=ds.train,
                label_list=ds.label_list,
                val_df=ds.val,
                batch_size=batch_size_probe,
            )
            if cal["minority_classes"]:
                cfg["minority_classes"] = cal["minority_classes"]
            cfg["use_logit_adjustment"] = cal["use_logit_adjustment"]
            cfg["logit_adj_tau"] = cal["logit_adj_tau"]
            cfg["eval_logit_adj_tau"] = cal["eval_logit_adj_tau"]
            cfg["target_oversample"] = cal["target_oversample"]
            cfg["label_smoothing"] = cal["label_smoothing"]
            cfg["use_class_weights"] = cal["use_class_weights"]

        excel_out_path = os.path.join(results_dir, f"{ds_name}_evaluation_results.xlsx")
        npz_pred_path = os.path.join(results_dir, f"{ds_name}_evaluation_results_predictions.npz")

        if plot_only:
            # Locate existing results excel
            existing_excel = find_existing_results_file(ds_name, results_dir)
            if existing_excel:
                print(f"  [Plot-Only] Reading results from: {existing_excel}")
                results_df = pd.read_excel(existing_excel)
                all_results[ds_name] = results_df
            else:
                raise FileNotFoundError(
                    f"No existing results excel found for {ds_name} in '{results_dir}'. "
                    f"Please rerun 'python run_all_benchmarks.py' without --plot_only to execute benchmarks."
                )
        else:
            # Check if Step 1 (Generation) and Step 2 (Filtering) are requested or needed
            existing_aug = load_augmented_datasets(ds_name, augmented_dir=augmented_dir)
            if existing_aug and len(existing_aug) == 4 and not force_generation:
                print(f"\n  [Cache Hit] Complete augmented datasets already exist for {ds_name} ({list(existing_aug.keys())}).")
                print("  Reusing existing augmented data to preserve GPU VRAM and skip redundant generation.")
                print("  (Pass '--force_generation' to override and force fresh generation).")
                aug_dfs = existing_aug
            elif run_generation or not existing_aug:
                print(f"\n  >>> STEP 1: Running Text Generation & Similarity Scoring for {ds_name}...")
                minority_classes = cfg.get("minority_classes", ds.label_list)
                augmenter = IndonesianAugmenter(
                    gen_model_name=gen_model,
                    embed_model_name="sentence-transformers/distiluse-base-multilingual-cased-v2",
                    device=device,
                )
                gen_df = augmenter.process_dataset(
                    ds.train,
                    sample_limit=generation_sample_limit,
                    target_labels=minority_classes,
                )

                print(f"\n  >>> STEP 2: Running Similarity Threshold Filtering & Dataset Export for {ds_name}...")
                all_text_dfs, new_text_dfs, thresholds = build_augmented_datasets(
                    df=gen_df,
                    labels_to_augment=minority_classes,
                    base_train_df=ds.train,
                    test_df=ds.test,  # Strict zero-leakage check against test set
                )
                base_aug_dir = os.path.dirname(augmented_dir.rstrip("/"))
                export_augmented_datasets(
                    dataset_name=ds_name,
                    all_text_dfs=all_text_dfs,
                    new_text_dfs=new_text_dfs,
                    output_dir=base_aug_dir,
                )
                aug_dfs = all_text_dfs
                if "augmenter" in locals() and augmenter is not None:
                    try:
                        augmenter.cleanup()
                    except Exception:
                        pass
                    del augmenter
                from src.utils.hardware import deep_cuda_purge
                deep_cuda_purge()
            else:
                aug_dfs = existing_aug

            print(f"\n  >>> STEP 3: Fine-Tuning IndoBERT Across All Variants for {ds_name}...")
            # Thorough memory purge before training begins
            from src.utils.hardware import deep_cuda_purge, inspect_gpu_processes, clean_stale_gpu_processes
            if clean_gpu:
                clean_stale_gpu_processes(auto_kill=True)
            deep_cuda_purge()
            inspect_gpu_processes()
            if torch.cuda.is_available():
                low_mem_warning = False
                for dev_i in range(torch.cuda.device_count()):
                    try:
                        free_mem, total_mem = torch.cuda.mem_get_info(dev_i)
                        free_gib = free_mem / (1024**3)
                        total_gib = total_mem / (1024**3)
                        print(f"  [GPU {dev_i} Status] Free: {free_gib:.2f} GiB / Total: {total_gib:.2f} GiB")
                        if free_gib < 4.0:
                            low_mem_warning = True
                    except Exception:
                        pass
                if low_mem_warning:
                    print("  [Notice] One or more GPUs report < 4.0 GiB free VRAM. "
                          "Dynamic micro-batching will adaptively scale batch sizes to prevent OOM.")
            epochs = epochs_override if epochs_override is not None else cfg["epochs"]
            batch_size = batch_size_override if batch_size_override is not None else cfg["batch_size"]

            evaluator = IndoBERTClassifierEvaluator(
                model_name=cfg["model_name"],
                max_len=cfg["max_len"],
                batch_size=batch_size,
                epochs=epochs,
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
                device=device,
                seed=seed,
                early_stopping=cfg["early_stopping"],
                patience=cfg["patience"],
                use_preprocessing=cfg["use_preprocessing"],
                use_class_weights=cfg["use_class_weights"],
                label_smoothing=cfg["label_smoothing"],
                use_fgm=cfg["use_fgm"],
                fgm_epsilon=cfg["fgm_epsilon"],
                use_logit_adjustment=cfg.get("use_logit_adjustment", False),
                logit_adj_tau=cfg.get("logit_adj_tau", 0.8),
                eval_logit_adj_tau=cfg.get("eval_logit_adj_tau", 0.0),
                target_oversample=cfg.get("target_oversample", None),
                ensemble_configs=cfg.get("ensemble_models", None) if (run_ensemble and cfg.get("run_ensemble", False)) else None,
                prompt_prefix=cfg.get("aspect_prompt_prefix", None),
            )

            results_df = evaluator.evaluate_all_variants(
                base_train_df=ds.train,
                val_df=ds.val,
                base_test_df=ds.test,
                augmented_dfs=aug_dfs,
                label_list=ds.label_list,
                dataset_name=ds_name,
                output_results_path=excel_out_path,
                seed=seed,
            )
            all_results[ds_name] = results_df
            del evaluator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # -------------------------------------------------------------
        # 1. Generate 4-Metric Performance Comparison Bar Chart
        # -------------------------------------------------------------
        print(f"  Generating 4-metric comparison chart for {ds_name}...")
        plot_dataset_metric_comparison(results_df, ds_name, output_dir=figures_dir)

        # -------------------------------------------------------------
        # 2. Generate Confusion Matrix Heatmaps
        # -------------------------------------------------------------
        print(f"  Generating confusion matrix comparison for {ds_name}...")
        # Check if predictions NPZ exists
        npz_candidates = [
            npz_pred_path,
            os.path.join(results_dir, f"{ds_name}_evaluation_results_predictions.npz"),
            os.path.join(results_dir, f"{ds_name}_predictions.npz"),
            os.path.join(results_dir, f"{ds_name}_evaluation_results_1_predictions.npz"),
            os.path.join(CURRENT_DIR, "data", "results", f"{ds_name}_evaluation_results_predictions.npz"),
            os.path.join(CURRENT_DIR, "data", "results", f"{ds_name}_predictions.npz"),
            os.path.join(CURRENT_DIR, "data", "results", f"{ds_name}_evaluation_results_1_predictions.npz"),
        ]
        npz_cand = None
        for cand in npz_candidates:
            if cand and os.path.exists(cand):
                npz_cand = cand
                break

        if npz_cand and os.path.exists(npz_cand):
            try:
                npz_data = np.load(npz_cand)
                # Prefer RawERM baseline if present (unaugmented ERM), otherwise NotAugmented
                base_cand_prefixes = [
                    f"{ds_name}-Baseline-RawERM",
                    f"{ds_name}-Baseline-NotAugmented",
                    f"{ds_name}-Baseline",
                ]
                base_key_true = None
                base_key_pred = None
                for bp in base_cand_prefixes:
                    if f"{bp}__true" in npz_data and f"{bp}__pred" in npz_data:
                        base_key_true = f"{bp}__true"
                        base_key_pred = f"{bp}__pred"
                        break

                # Find best augmented key (strictly excluding Baseline and Raw-ERM)
                best_var_name = "Best Augmented"
                best_key_true = None
                best_key_pred = None
                best_f1 = -1.0

                df_aug = results_df[
                    ~results_df["dataset_name"].str.contains("Baseline") &
                    ~results_df["dataset_name"].str.contains("Raw")
                ].copy()
                if not df_aug.empty and "macro_f1" in df_aug.columns:
                    best_row = df_aug.loc[df_aug["macro_f1"].idxmax()]
                    target_name = best_row["dataset_name"]
                    best_var_name = clean_variant_name(target_name) + "-Augmented"
                    t_key = f"{target_name}__true"
                    p_key = f"{target_name}__pred"
                    if t_key in npz_data and p_key in npz_data:
                        best_key_true = t_key
                        best_key_pred = p_key

                # Fallback to any augmented key
                if best_key_true is None:
                    for k in npz_data.files:
                        if "Augmented" in k and k.endswith("__true"):
                            best_key_true = k
                            best_key_pred = k.replace("__true", "__pred")
                            best_var_name = k.split("__")[0]
                            break

                if base_key_true in npz_data and best_key_true in npz_data:
                    plot_confusion_matrices(
                        y_true_base=npz_data[base_key_true],
                        y_pred_base=npz_data[base_key_pred],
                        y_true_aug=npz_data[best_key_true],
                        y_pred_aug=npz_data[best_key_pred],
                        label_list=ds.label_list,
                        dataset_name=ds_name,
                        aug_variant_name=best_var_name,
                        output_dir=figures_dir,
                    )
            except Exception as e:
                print(f"  [Notice] Could not generate confusion matrix from {npz_cand}: {e}")
        else:
            print(f"  [Notice] Predictions file {npz_cand} not found. Confusion matrix requires model evaluation run to capture test predictions.")

    # -------------------------------------------------------------
    # Step 2b: Compute and Persist Architectural Ablation Analysis
    # -------------------------------------------------------------
    print("\n>>> [STEP 2b] Computing and saving architectural ablation analysis to persistent files...")
    try:
        from src.benchmarks.table_generator import generate_access_ablation_table
        generate_access_ablation_table(
            results_dir=results_dir,
            out_results_dir="data/results/extended",
            paper_dir="paper",
        )
    except Exception as e:
        print(f"  [Notice] Ablation table generation deferred: {e}")

    # -------------------------------------------------------------
    # 3. Generate Cross-Dataset Multi-Panel Dashboard
    # -------------------------------------------------------------
    if all_results:
        print("\nGenerating Cross-Dataset Benchmark Summary Dashboard...")
        plot_cross_dataset_summary(all_results, output_dir=figures_dir)

    # -------------------------------------------------------------
    # 4. Extended IEEE TPAMI and IEEE Access Diagnostic Suite
    # -------------------------------------------------------------
    if (
        all_tpami_access
        or run_tpami_suite
        or run_cv
        or run_comparative_baselines
        or run_computational_budget
        or run_lexical_diversity
        or run_training_dynamics
        or run_similarity_distributions
    ):
        ext_fig_dir = "data/figures" if os.path.exists("data/figures") else figures_dir
        ext_res_dir = "data/results/extended"
        run_all_tpami_access_suite(
            target_datasets=target_datasets,
            figures_dir=ext_fig_dir,
            results_dir=ext_res_dir,
            run_cv=run_cv,
            cv_epochs=cv_epochs,
            device=device,
            paper_dir="paper",
        )

    print("\n" + "=" * 80)
    print("INVESTIGATION COMPLETE!")
    print(f"All evaluation result tables saved to: {os.path.abspath(results_dir)}")
    print(f"All publication-ready figures saved to: {os.path.abspath(figures_dir)}")
    print("=" * 80)


# =====================================================================
# CLI Entry Point
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all investigations and generate comparison and confusion matrix visualizations."
    )
    default_datasets = get_investigated_datasets()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=default_datasets,
        help=f"Datasets to evaluate (choices: {', '.join(default_datasets)}, all)",
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        default=False,
        help="Skip model training and generate/update plots from existing evaluation results",
    )
    parser.add_argument(
        "--run_generation",
        "--from_scratch",
        action="store_true",
        default=False,
        dest="run_generation",
        help="Run ALL steps end-to-end from scratch: Phase 1 (GPT-2 generation) -> Phase 2 (Filtering) -> Phase 3 (IndoBERT training) -> Phase 4 (Visualizations)",
    )
    parser.add_argument(
        "--generation_sample_limit",
        type=int,
        default=None,
        help="Limit number of training samples to augment during Phase 1 (e.g. 200 for fast run, None for full)",
    )
    parser.add_argument(
        "--gen_model",
        type=str,
        default="cahya/gpt2-small-indonesian-522M",
        help="Generative backbone model or alias (e.g. 'gpt2-small', 'sailor-0.5b', 'qwen2.5-0.5b', 'smollm2-135m')",
    )
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="Print full comparison table of candidate generative models and exit",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="data/results",
        help="Directory containing or to store evaluation Excel results",
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default="data/results/figures",
        help="Directory to save generated comparison and confusion matrix figures",
    )
    parser.add_argument(
        "--augmented_dir",
        type=str,
        default="data/augmented/all_text",
        help="Directory containing filtered augmented datasets",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (e.g. 'cuda:0', 'mps', 'cpu', 'auto'). If omitted, automatically detects and leverages all available GPUs via torch.nn.DataParallel (training) and device_map='auto' (generation).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--run_ensemble",
        action="store_true",
        default=True,
        help="Enable heterogeneous multi-model ensemble pipeline for supported datasets (e.g. EmoT)",
    )
    parser.add_argument(
        "--no_ensemble",
        action="store_false",
        dest="run_ensemble",
        help="Disable heterogeneous multi-model ensemble pipeline",
    )
    parser.add_argument(
        "--auto_calibrate",
        action="store_true",
        default=True,
        help="Automatically detect minority classes, calibrate logit adjustment tau, oversampling, and regularization",
    )
    parser.add_argument(
        "--no_auto_calibrate",
        action="store_false",
        dest="auto_calibrate",
        help="Disable automatic imbalance calibration and fallback to hardcoded configs",
    )
    # IEEE TPAMI & Access Extended Arguments
    parser.add_argument(
        "--all_tpami_access",
        action="store_true",
        default=True,
        help="Run all extended IEEE TPAMI and IEEE Access additional tests, tables, and figures (default: True)",
    )
    parser.add_argument(
        "--no_tpami_access",
        action="store_false",
        dest="all_tpami_access",
        help="Skip extended IEEE TPAMI and IEEE Access suite",
    )
    parser.add_argument(
        "--run_tpami_suite",
        action="store_true",
        default=False,
        help="Execute TPAMI analytical suite (t-SNE latent geometry, margin CDF, ECE reliability, IR sweep)",
    )
    parser.add_argument(
        "--run_cv",
        action="store_true",
        default=False,
        help="Run real stratified 5-fold cross-validation experiment on the target datasets",
    )
    parser.add_argument(
        "--cv_epochs",
        type=int,
        default=2,
        help="Number of epochs per fold for real 5-fold cross validation (default: 2)",
    )
    parser.add_argument(
        "--run_comparative_baselines",
        action="store_true",
        default=False,
        help="Run comparative baselines against EDA (Easy Data Augmentation) and MarianMT Back-Translation",
    )
    parser.add_argument(
        "--run_computational_budget",
        action="store_true",
        default=False,
        help="Profile and generate computational complexity, latency, and memory footprint table",
    )
    parser.add_argument(
        "--run_lexical_diversity",
        action="store_true",
        default=False,
        help="Compute Distinct-1 and Distinct-2 n-gram lexical diversity and Shannon entropy",
    )
    parser.add_argument(
        "--run_training_dynamics",
        action="store_true",
        default=False,
        help="Generate training cross-entropy loss and validation Macro-F1 convergence curves",
    )
    parser.add_argument(
        "--run_similarity_distributions",
        action="store_true",
        default=False,
        help="Plot probability density histograms and cutoff thresholds for ECU, COS, JAC, and BLEU",
    )
    parser.add_argument(
        "--force_generation",
        action="store_true",
        default=False,
        help="Force re-running Phase 1 text generation even if augmented datasets already exist",
    )
    parser.add_argument(
        "--clean_gpu",
        action="store_true",
        default=True,
        help="Automatically terminate stale or orphaned GPU benchmark processes before execution (default: True)",
    )
    parser.add_argument(
        "--no_clean_gpu",
        action="store_false",
        dest="clean_gpu",
        help="Disable automatic termination of stale GPU processes",
    )
    parser.add_argument(
        "--use_adafactor",
        action="store_true",
        default=False,
        help="Force using Adafactor optimizer (~12 MB VRAM vs 880 MB for AdamW) to guarantee zero OOM",
    )

    args = parser.parse_args()

    if args.list_models:
        print("\nCandidate Generative Models Comparison (First-Gen 2020–2021 vs. Contemporary 2024):")
        IndonesianAugmenter.display_model_comparisons()
        sys.exit(0)

    selected = args.datasets
    if "all" in [s.lower() for s in selected]:
        selected = get_investigated_datasets()

    run_benchmark_pipeline(
        target_datasets=selected,
        plot_only=args.plot_only,
        run_generation=args.run_generation,
        force_generation=args.force_generation,
        clean_gpu=args.clean_gpu,
        use_adafactor=args.use_adafactor,
        gen_model=args.gen_model,
        generation_sample_limit=args.generation_sample_limit,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        augmented_dir=args.augmented_dir,
        device=args.device,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        seed=args.seed,
        run_ensemble=args.run_ensemble,
        auto_calibrate=args.auto_calibrate,
        all_tpami_access=args.all_tpami_access,
        run_tpami_suite=args.run_tpami_suite,
        run_cv=args.run_cv,
        cv_epochs=args.cv_epochs,
        run_comparative_baselines=args.run_comparative_baselines,
        run_computational_budget=args.run_computational_budget,
        run_lexical_diversity=args.run_lexical_diversity,
        run_training_dynamics=args.run_training_dynamics,
        run_similarity_distributions=args.run_similarity_distributions,
    )

