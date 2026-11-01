"""
Phase 2: Similarity-Based Augmentation Filter with Anti-Hallucination & Quality Verification.
Computes threshold averages (Euclidean, Cosine, Jaccard, BLEU) and produces augmented datasets:
- ECU-Augmented
- COS-Augmented
- JAC-Augmented
- BLEU-Augmented
Generates both 'All-Text' and 'New-Text' variants, with strict suffix-level hallucination filtering.
"""

import os
import re
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any

from src.utils.path_utils import resolve_path, ensure_dir

# =====================================================================
# Anti-Hallucination & Wikipedia Encyclopedic Boilerplate Detector
# =====================================================================
HALLUCINATION_REGEXES = [
    r"\badalah\s+(sebuah\s+|salah\s+satu\s+)?(desa|kelurahan|kecamatan|kabupaten|provinsi|munisipalitas|gampong|powiat|wilayah|daerah)\b",
    r"\bmerupakan\s+(sebuah\s+|salah\s+satu\s+)?(desa|kelurahan|kecamatan|kabupaten|provinsi|munisipalitas|gampong|powiat|wilayah|daerah)\b",
    r"\bterletak\s+di\s+(kecamatan|kabupaten|provinsi|pulau|negara|antara)\b",
    r"\badalah\s+(sebuah\s+|suatu\s+)?(sinetron|film|lagu|album|serial|kontes|acara|stasiun|televisi|spesies|pesawat)\b",
    r"\bmerupakan\s+(sebuah\s+|suatu\s+)?(sinetron|film|lagu|album|serial|kontes|acara|stasiun|televisi|spesies|pesawat)\b",
    r"\badalah\s+(seorang\s+|tokoh\s+|raja\s+|sultan\s+|pahlawan\s+|presiden\s+|menteri\s+|gubernur)\b",
    r"\bmerupakan\s+(seorang\s+|tokoh\s+|raja\s+|sultan\s+|pahlawan\s+|presiden\s+|menteri\s+|gubernur)\b",
    r"\b(mitologi|yunani|zeus|populasi\s+sebesar|kode\s+nato|wikipedia|pesawat\s+tempur|suku\s+[A-Za-z]+|powiat)\b",
    r"\b(bahasa\s+[A-Za-z]+\s*:|\([^\)]*bahasa\s+[A-Za-z]+)\b",
    r"\b(dr\.\s+h\.|dr\.\s+|prof\.\s+|kgpaa|mangkunegara|paku\s+alam)\b",
    r"https?://|www\.",
]
HALLUCINATION_PATTERN = re.compile("|".join(HALLUCINATION_REGEXES), re.IGNORECASE)


def is_hallucinatory_or_corrupt(text: str) -> bool:
    """Detects whether generated text contains Wikipedia boilerplate, URLs, or degenerate repetition."""
    s = str(text).strip()
    if not s:
        return True

    # 1. Wikipedia encyclopedic boilerplate match
    if HALLUCINATION_PATTERN.search(s):
        return True

    # 2. Extreme repetition check (degenerate loop: e.g. "kata kata kata...")
    words = s.lower().split()
    if len(words) >= 5:
        distinct_ratio = len(set(words)) / len(words)
        if distinct_ratio < 0.35:  # Over 65% repetitive tokens
            return True

    return False


def compute_dynamic_thresholds(
    df: pd.DataFrame,
    ecu_col: str = "ecu_similarity",
    cos_col: str = "cos_similarity",
    jacc_col: str = "jacc_similarity",
    bleu_col: str = "bleu_sim_1",
) -> Dict[str, float]:
    """Computes arithmetic mean thresholds for each similarity metric."""
    return {
        "ECU": float(df[ecu_col].mean()),
        "COS": float(df[cos_col].mean()),
        "JAC": float(df[jacc_col].mean()),
        "BLEU": float(df[bleu_col].mean()),
    }


def filter_augmented_data(
    intermediate_df: pd.DataFrame,
    thresholds: Optional[Dict[str, float]] = None,
    filter_hallucinations: bool = True,
    metric_cols: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Filters generated samples based on similarity thresholds.
    Returns nested dictionary:
    {
        'ECU': {'all_text': df, 'new_text': df},
        'COS': {'all_text': df, 'new_text': df},
        'JAC': {'all_text': df, 'new_text': df},
        'BLEU': {'all_text': df, 'new_text': df},
    }
    """
    if metric_cols is None:
        metric_cols = {
            "ECU": "ecu_similarity",
            "COS": "cos_similarity",
            "JAC": "jacc_similarity",
            "BLEU": "bleu_sim_1",
        }

    if thresholds is None:
        thresholds = compute_dynamic_thresholds(intermediate_df)

    results: Dict[str, Dict[str, pd.DataFrame]] = {}

    for metric, col in metric_cols.items():
        thresh = thresholds[metric]
        valid_mask = intermediate_df[col] >= thresh

        filtered = intermediate_df[valid_mask].copy()

        # Strict Anti-Hallucination & Quality Verification
        if filter_hallucinations:
            before_cnt = len(filtered)
            bad_mask = filtered["new_text"].apply(is_hallucinatory_or_corrupt)
            filtered = filtered[~bad_mask].copy()
            discarded = before_cnt - len(filtered)
            if discarded > 0:
                print(f"  [{metric}] Discarded {discarded} hallucinatory / degenerate samples ({discarded/before_cnt:.1%}).")

        # Prepare All-Text variant (original seed + generated continuation)
        all_text_df = pd.DataFrame({
            "text": filtered["all_text"],
            "label": filtered["label"],
        })

        # Prepare New-Text variant (pure generated continuation)
        new_text_df = pd.DataFrame({
            "text": filtered["new_text"],
            "label": filtered["label"],
        })

        results[metric] = {
            "all_text": all_text_df,
            "new_text": new_text_df,
        }

    return results


def export_augmented_datasets(
    filtered_dict: Dict[str, Dict[str, pd.DataFrame]],
    dataset_name: str,
    output_base_dir: str = "data/augmented",
) -> None:
    """Exports filtered augmented sets into standard project directories."""
    all_dir = resolve_path(os.path.join(output_base_dir, "all_text"))
    new_dir = resolve_path(os.path.join(output_base_dir, "new_text"))
    ensure_dir(all_dir)
    ensure_dir(new_dir)

    for metric, variants in filtered_dict.items():
        all_path = os.path.join(all_dir, f"{dataset_name}-Augmented-{metric}-ALL-Text-Final.csv")
        new_path = os.path.join(new_dir, f"{dataset_name}-Augmented-{metric}-NEW-Text-Final.csv")

        variants["all_text"].to_csv(all_path, index=False)
        variants["new_text"].to_csv(new_path, index=False)
        print(f"  Saved {metric} All-Text: {all_path} ({len(variants['all_text'])} samples)")
        print(f"  Saved {metric} New-Text: {new_path} ({len(variants['new_text'])} samples)")
