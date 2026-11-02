"""
General Evaluation Metrics Utilities for Lexical Diversity, Calibration, and Classification.
"""

import os
import math
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple, Optional
from sklearn.metrics import matthews_corrcoef

from src.utils.path_utils import resolve_path


def compute_distinct_n(texts: List[str], n: int = 1) -> float:
    """Computes Distinct-N ratio: unique n-grams / total n-grams."""
    total_ngrams = 0
    unique_ngrams = set()

    for text in texts:
        tokens = str(text).split()
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            unique_ngrams.add(ngram)
            total_ngrams += 1

    return float(len(unique_ngrams) / total_ngrams) if total_ngrams > 0 else 0.0


def compute_entropy(texts: List[str]) -> float:
    """Computes Shannon entropy over vocabulary distribution."""
    counts = Counter()
    for text in texts:
        counts.update(str(text).split())

    total = sum(counts.values())
    if total == 0:
        return 0.0

    ent = -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)
    return float(round(ent, 2))


def compute_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Matthews Correlation Coefficient (MCC)."""
    return float(matthews_corrcoef(y_true, y_pred))


def compute_calibration_metrics(
    confidences: np.ndarray,
    accuracies: np.ndarray,
) -> Tuple[float, float]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    """
    gap = np.abs(confidences - accuracies)
    ece = float(np.mean(gap))
    mce = float(np.max(gap))
    return ece, mce


def extract_f1_anchors(eval_excel: str) -> Tuple[float, float, str]:
    """
    Dynamically extracts baseline macro-F1, best augmented macro-F1, and best variant name
    from a dataset evaluation spreadsheet.
    """
    resolved_excel = resolve_path(eval_excel)
    if not os.path.exists(resolved_excel):
        raise FileNotFoundError(
            f"Missing evaluation file: '{resolved_excel}'. "
            "Please run 'python run_all_benchmarks.py' to generate benchmark evaluations."
        )

    df = pd.read_excel(resolved_excel)
    b_df = df[df["dataset_name"].str.contains("Baseline|RawERM", case=False, na=False)]
    a_df = df[
        df["dataset_name"].str.contains("Augmented", case=False, na=False)
        & ~df["dataset_name"].str.contains("NotAugmented", case=False, na=False)
    ]
    if b_df.empty or a_df.empty:
        raise ValueError(
            f"Baseline or augmented evaluations not found in '{resolved_excel}'. "
            "Please rerun 'python run_all_benchmarks.py'."
        )

    b_val = float(b_df.iloc[0]["macro_f1"])
    best_idx = a_df["macro_f1"].idxmax()
    a_val = float(a_df.loc[best_idx, "macro_f1"])
    v_name = str(a_df.loc[best_idx, "dataset_name"]).split("-")[-1]
    return b_val, a_val, v_name

