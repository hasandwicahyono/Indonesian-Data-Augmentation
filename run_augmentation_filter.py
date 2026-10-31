"""
Phase 2: Similarity-Based Augmentation Filter.
Computes threshold averages (Euclidean, Cosine, Jaccard, BLEU) and produces the 4 augmented datasets:
- ECU-Augmented
- COS-Augmented
- JAC-Augmented
- BLE-Augmented
Generates both 'All-Text' and 'New-Text' variants.
"""

import os
import argparse
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any


def compute_similarity_thresholds(df: pd.DataFrame) -> Dict[str, float]:
    """Computes mean similarity thresholds across the dataset."""
    thresholds = {
        "ECU": float(df["ecu_similarity"].mean()),
        "COS": float(df["cos_similarity"].mean()),
        "JAC": float(df["jacc_similarity"].mean()),
        "BLEU": float(df["bleu_sim_1"].mean()),
    }
    return thresholds


def check_threshold(
    value: float,
    label: str,
    metric: str,
    thresholds: Dict[str, float],
    labels_to_augment: List[str],
) -> bool:
    """Checks whether the metric meets or exceeds the threshold for minority classes."""
    # Only augment designated minority classes
    target_labels = [l.strip().lower() for l in labels_to_augment]
    if str(label).strip().lower() not in target_labels:
        return False

    metric_key = metric.upper()
    if metric_key not in thresholds:
        return False

    try:
        val = float(value)
        return val >= thresholds[metric_key]
    except (ValueError, TypeError):
        return False


def build_augmented_datasets(
    df: pd.DataFrame,
    labels_to_augment: List[str],
    base_train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    text_column: str = "text",
    label_column: str = "label",
    all_text_column: str = "all_text",
    new_text_column: str = "new_text",
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, float]]:
    """
    Constructs augmented datasets for All-Text and New-Text variants across the 4 similarity metrics.
    If base_train_df is provided, the augmented datasets will be built by augmenting that exact base_train_df.
    If test_df is provided, any newly generated sample matching the test set is strictly filtered out to prevent data leakage.
    """
    thresholds = compute_similarity_thresholds(df)
    print(f"Computed Similarity Thresholds: {thresholds}")

    test_texts = set(str(t).strip() for t in test_df[text_column].dropna()) if test_df is not None else set()
    leak_filtered_count = 0

    metrics = ["ECU", "COS", "JAC", "BLEU"]
    aug_all_samples: Dict[str, List[Dict[str, Any]]] = {m: [] for m in metrics}
    aug_new_samples: Dict[str, List[Dict[str, Any]]] = {m: [] for m in metrics}

    # Growth counters
    growth_counts = {m: 0 for m in metrics}

    for _, row in df.iterrows():
        label = row[label_column]
        all_text = row[all_text_column]
        new_text = row[new_text_column]

        # Prevent leakage if test_df is provided
        if test_texts:
            if str(all_text).strip() in test_texts or str(new_text).strip() in test_texts:
                leak_filtered_count += 1
                continue

        metric_values = {
            "ECU": row["ecu_similarity"],
            "COS": row["cos_similarity"],
            "JAC": row["jacc_similarity"],
            "BLEU": row["bleu_sim_1"],
        }

        # Check conditions and extract augmented samples
        for m in metrics:
            if check_threshold(metric_values[m], label, m, thresholds, labels_to_augment):
                growth_counts[m] += 1
                aug_all_samples[m].append({"text": all_text, "label": label})
                aug_new_samples[m].append({"text": new_text, "label": label})

    if leak_filtered_count > 0:
        print(f"  [ANTI-LEAKAGE] Filtered out {leak_filtered_count} generated candidate(s) identical to test set.")

    # Determine base training set to attach augmentations to
    effective_base = base_train_df[[text_column, label_column]].copy() if base_train_df is not None else df[[text_column, label_column]].copy()
    effective_base.columns = ["text", "label"]

    result_all = {}
    result_new = {}
    for m in metrics:
        aug_all_df = pd.DataFrame(aug_all_samples[m])
        aug_new_df = pd.DataFrame(aug_new_samples[m])
        result_all[m] = pd.concat([effective_base, aug_all_df], ignore_index=True) if len(aug_all_df) > 0 else effective_base.copy()
        result_new[m] = pd.concat([effective_base, aug_new_df], ignore_index=True) if len(aug_new_df) > 0 else effective_base.copy()

    print("Augmentation Summary (Growth Counts):")
    base_count = len(effective_base)
    for m in metrics:
        growth = growth_counts[m]
        pct = (growth / base_count) * 100.0 if base_count > 0 else 0.0
        print(f"  - {m}: Added {growth} samples (+{pct:.2f}%) -> Total {len(result_all[m])}")

    return result_all, result_new, thresholds


def export_augmented_datasets(
    dataset_name: str,
    all_text_dfs: Dict[str, pd.DataFrame],
    new_text_dfs: Dict[str, pd.DataFrame],
    output_dir: str = "data/augmented",
):
    """Exports augmented datasets into structured CSV files."""
    all_dir = os.path.join(output_dir, "all_text")
    new_dir = os.path.join(output_dir, "new_text")
    os.makedirs(all_dir, exist_ok=True)
    os.makedirs(new_dir, exist_ok=True)

    for metric, df in all_text_dfs.items():
        fname = f"{dataset_name}-Augmented-{metric}-ALL-Text-Final.csv"
        path = os.path.join(all_dir, fname)
        df.to_csv(path, index=False)
        print(f"Saved All-Text: {path} ({len(df)} rows)")

    for metric, df in new_text_dfs.items():
        fname = f"{dataset_name}-Augmented-{metric}-New-Text-Final.csv"
        path = os.path.join(new_dir, fname)
        df.to_csv(path, index=False)
        print(f"Saved New-Text: {path} ({len(df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and export augmented Indonesian datasets.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to intermediate augmented CSV/Excel")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset identifier (SmSA, EmoT, HoASA, MovieOpini)")
    parser.add_argument("--minority_classes", nargs="+", required=True, help="List of class labels to augment")
    parser.add_argument("--output_dir", type=str, default="data/augmented", help="Base directory to save augmented sets")
    args = parser.parse_args()

    if args.input_file.endswith(".xlsx"):
        df = pd.read_excel(args.input_file)
    else:
        df = pd.read_csv(args.input_file)

    all_dfs, new_dfs, _ = build_augmented_datasets(df, args.minority_classes)
    export_augmented_datasets(args.dataset_name, all_dfs, new_dfs, args.output_dir)
