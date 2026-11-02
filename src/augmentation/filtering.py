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

# =====================================================================
# Affective Valence & Cross-Emotion Boundary Guardrail
# =====================================================================
POSITIVE_VALENCE_KEYWORDS = {
    "senang", "bahagia", "gembira", "bersyukur", "alhamdulillah", "mantap", "hebat",
    "puas", "sukses", "keren", "cinta", "sayang", "suka", "love", "demen", "tresno"
}
NEGATIVE_VALENCE_KEYWORDS = {
    "marah", "benci", "kesal", "kecewa", "sedih", "nangis", "tangis", "hancur", "takut",
    "ngeri", "cemas", "khawatir", "bangsat", "anjing", "bodoh", "rusak", "parah", "jorok"
}
AGGRESSIVE_ANGER_KEYWORDS = {
    "bangsat", "keparat", "anjing", "kampret", "bodoh", "tolol", "bajingan", "bacok",
    "gebuk", "hajar", "pecat", "bubarkan", "kutuk", "celaka"
}


def is_valence_consistent(target_label: str, text: str) -> bool:
    """
    Ensures generated text does not blatantly violate psychological valence boundaries
    (Russell's Circumplex Model) or bleed across opposing emotional poles.
    """
    s = str(text).lower()
    lbl = str(target_label).strip().lower()

    # Negative emotions / polarities: should not be predominantly celebratory
    if lbl in {"anger", "sadness", "fear", "neg", "negative"}:
        pos_hits = sum(1 for kw in POSITIVE_VALENCE_KEYWORDS if re.search(r"\b" + re.escape(kw) + r"\b", s))
        neg_hits = sum(1 for kw in NEGATIVE_VALENCE_KEYWORDS if re.search(r"\b" + re.escape(kw) + r"\b", s))
        if pos_hits >= 2 and neg_hits == 0:
            return False

    # Positive emotions / polarities: should not contain severe aggressive invective
    if lbl in {"happy", "love", "pos", "positive"}:
        hostile_hits = sum(1 for kw in AGGRESSIVE_ANGER_KEYWORDS if re.search(r"\b" + re.escape(kw) + r"\b", s))
        if hostile_hits >= 1:
            return False

    # Fine-grained sadness vs anger protection:
    # If target is sadness (grief/sorrow), reject extreme aggressive expletives belonging strictly to anger
    if lbl == "sadness":
        hostile_hits = sum(1 for kw in AGGRESSIVE_ANGER_KEYWORDS if re.search(r"\b" + re.escape(kw) + r"\b", s))
        if hostile_hits >= 2:
            return False

    return True


def is_degenerate_jaccard_slang(text: str) -> bool:
    """
    Detects repeated informal particle loops (e.g. 'wkwk wkwk', 'bgt bgt') that
    artificially elevate lexical Jaccard overlap while destroying semantic coherence.
    """
    s = str(text).lower()
    slang_loops = [
        r"(wkwk\s*){3,}",
        r"(bgt\s*){3,}",
        r"(anjir\s*){3,}",
        r"(haha\s*){3,}",
        r"(ckck\s*){3,}",
    ]
    for pattern in slang_loops:
        if re.search(pattern, s):
            return True
    return False


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
        if distinct_ratio < 0.40:
            return True
        trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
        if len(trigrams) >= 3 and len(set(trigrams)) / len(trigrams) < 0.35:
            return True

    return False


def clean_and_truncate_continuation(text: str) -> str:
    """Strips citation brackets and truncates open trailing fragments to the last sentence terminator."""
    s = str(text).strip()
    s = re.sub(r"\[\d+\]", "", s)
    s = s.replace("&apos;", "'").replace("&quot;", '"')
    s = re.sub(r"\s+", " ", s).strip()

    last_punct = max(s.rfind("."), s.rfind("!"), s.rfind("?"))
    if last_punct > 15:
        s = s[: last_punct + 1].strip()

    return s


def compute_similarity_thresholds(df: pd.DataFrame) -> Dict[str, float]:
    """Computes mean similarity thresholds across valid dataset rows."""
    thresholds = {
        "ECU": float(df["ecu_similarity"].mean()) if "ecu_similarity" in df else 0.5,
        "COS": float(df["cos_similarity"].mean()) if "cos_similarity" in df else 0.5,
        "JAC": float(df["jacc_similarity"].mean()) if "jacc_similarity" in df else 0.3,
        "BLEU": float(df["bleu_sim_1"].mean()) if "bleu_sim_1" in df else 0.2,
    }
    return thresholds


compute_dynamic_thresholds = compute_similarity_thresholds


def check_threshold(
    value: float,
    label: str,
    metric: str,
    thresholds: Dict[str, float],
    labels_to_augment: List[str],
    min_floor: Optional[float] = None,
) -> bool:
    """Checks whether metric meets or exceeds threshold for minority classes with an absolute safety floor."""
    target_labels = [l.strip().lower() for l in labels_to_augment]
    if str(label).strip().lower() not in target_labels:
        return False

    metric_key = metric.upper()
    if metric_key not in thresholds:
        return False

    try:
        val = float(value)
        cutoff = thresholds[metric_key]
        if min_floor is not None:
            cutoff = max(cutoff, min_floor)
        return val >= cutoff
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
    Applies strict Anti-Hallucination verification and suffix-level semantic grounding.
    """
    thresholds = compute_similarity_thresholds(df)
    print(f"Computed Similarity Thresholds: {thresholds}")

    test_texts = set(str(t).strip() for t in test_df[text_column].dropna()) if test_df is not None else set()
    leak_filtered_count = 0
    hallucination_filtered_count = 0
    valence_filtered_count = 0
    slang_loop_filtered_count = 0

    metrics = ["ECU", "COS", "JAC", "BLEU"]
    aug_all_samples: Dict[str, List[Dict[str, Any]]] = {m: [] for m in metrics}
    aug_new_samples: Dict[str, List[Dict[str, Any]]] = {m: [] for m in metrics}
    growth_counts = {m: 0 for m in metrics}

    safety_floors = {
        "ECU": 0.20,
        "COS": 0.35,
        "JAC": 0.10,
        "BLEU": 0.08,
    }

    for _, row in df.iterrows():
        label = row[label_column]
        all_text = clean_and_truncate_continuation(row.get(all_text_column, row.get(text_column, "")))
        new_text = clean_and_truncate_continuation(row.get(new_text_column, ""))

        if is_hallucinatory_or_corrupt(all_text) or is_hallucinatory_or_corrupt(new_text):
            hallucination_filtered_count += 1
            continue

        if not is_valence_consistent(label, all_text) or not is_valence_consistent(label, new_text):
            valence_filtered_count += 1
            continue

        if is_degenerate_jaccard_slang(all_text) or is_degenerate_jaccard_slang(new_text):
            slang_loop_filtered_count += 1
            continue

        if test_texts:
            if str(all_text).strip() in test_texts or str(new_text).strip() in test_texts:
                leak_filtered_count += 1
                continue

        metric_values = {
            "ECU": row.get("ecu_similarity", 0.0),
            "COS": row.get("cos_similarity", 0.0),
            "JAC": row.get("jacc_similarity", 0.0),
            "BLEU": row.get("bleu_sim_1", 0.0),
        }

        for m in metrics:
            floor = safety_floors.get(m, 0.1)
            if check_threshold(metric_values[m], label, m, thresholds, labels_to_augment, min_floor=floor):
                growth_counts[m] += 1
                aug_all_samples[m].append({"text": all_text, "label": label})
                aug_new_samples[m].append({"text": new_text if new_text else all_text, "label": label})

    if hallucination_filtered_count > 0:
        print(f"  [ANTI-HALLUCINATION GUARD] Purged {hallucination_filtered_count} off-topic/Wikipedia hallucinated candidates.")
    if valence_filtered_count > 0:
        print(f"  [VALENCE GUARD] Filtered out {valence_filtered_count} cross-valence drifted candidates.")
    if slang_loop_filtered_count > 0:
        print(f"  [SLANG PURGE] Filtered out {slang_loop_filtered_count} degenerate slang/particle loop candidates.")
    if leak_filtered_count > 0:
        print(f"  [ANTI-LEAKAGE] Filtered out {leak_filtered_count} generated candidate(s) identical to test set.")

    effective_base = base_train_df[[text_column, label_column]].copy() if base_train_df is not None else df[[text_column, label_column]].copy()
    effective_base.columns = ["text", "label"]

    result_all = {}
    result_new = {}
    for m in metrics:
        aug_all_df = pd.DataFrame(aug_all_samples[m])
        aug_new_df = pd.DataFrame(aug_new_samples[m])
        result_all[m] = pd.concat([effective_base, aug_all_df], ignore_index=True) if len(aug_all_df) > 0 else effective_base.copy()
        result_new[m] = pd.concat([effective_base, aug_new_df], ignore_index=True) if len(aug_new_df) > 0 else effective_base.copy()

    print("Augmentation Summary (Growth Counts after Quality Verification):")
    base_count = len(effective_base)
    for m in metrics:
        growth = growth_counts[m]
        pct = (growth / base_count) * 100.0 if base_count > 0 else 0.0
        print(f"  - {m}: Added {growth} clean samples (+{pct:.2f}%) -> Total {len(result_all[m])}")

    return result_all, result_new, thresholds


filter_augmented_data = build_augmented_datasets


def purge_hallucinations_from_existing_csv(csv_path: str, raw_base_texts: Optional[set] = None) -> int:
    """
    Surgically removes Wikipedia hallucinations and corrupt sequences from an existing augmented CSV file.
    Preserves authentic raw baseline training samples.
    """
    resolved_path = resolve_path(csv_path)
    if not os.path.exists(resolved_path):
        return 0

    df = pd.read_csv(resolved_path)
    initial_len = len(df)

    if raw_base_texts:
        is_raw = df["text"].astype(str).str.strip().isin(raw_base_texts)
        is_corrupt = df["text"].apply(is_hallucinatory_or_corrupt)
        drop_mask = (~is_raw) & is_corrupt
    else:
        drop_mask = df["text"].apply(is_hallucinatory_or_corrupt)

    purged_count = int(drop_mask.sum())
    if purged_count > 0:
        cleaned_df = df[~drop_mask].copy().reset_index(drop=True)
        cleaned_df.to_csv(resolved_path, index=False)
        print(f"  [SCRUBBED] {os.path.basename(resolved_path)}: Purged {purged_count} hallucinated samples ({initial_len} -> {len(cleaned_df)})")
    return purged_count


def export_augmented_datasets(
    dataset_name: str,
    all_text_dfs: Dict[str, pd.DataFrame],
    new_text_dfs: Dict[str, pd.DataFrame],
    output_dir: str = "data/augmented",
) -> None:
    """Exports augmented datasets into structured CSV files."""
    all_dir = resolve_path(os.path.join(output_dir, "all_text"))
    new_dir = resolve_path(os.path.join(output_dir, "new_text"))
    ensure_dir(all_dir)
    ensure_dir(new_dir)

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
