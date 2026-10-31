import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional


class AutoImbalanceCalibrator:
    """
    Data-Driven Adaptive Imbalance Calibrator (Menon et al., ICML 2020).
    Automatically inspects training label distribution and dynamically calibrates:
    - Minority class(es) detection (minority_classes)
    - Menon et al. Logit Adjustment activation (use_logit_adjustment)
    - Optimal training logit adjustment margin (logit_adj_tau)
    - Test / validation prior-shift compensation (eval_logit_adj_tau)
    - Gradient-starvation target oversampling multipliers (target_oversample)
    - Adaptive label smoothing regularization (label_smoothing)
    - Safeguarded class-weighting activation (use_class_weights)
    """

    @staticmethod
    def calibrate(
        train_df: pd.DataFrame,
        label_list: List[str],
        val_df: Optional[pd.DataFrame] = None,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        counts = {lbl: int(sum(train_df["label"] == lbl)) for lbl in label_list}
        total_samples = max(1, len(train_df))
        num_classes = len(label_list)
        priors = {lbl: counts[lbl] / total_samples for lbl in label_list}
        uniform_prior = 1.0 / num_classes

        max_count = max(counts.values()) if counts else 1
        min_count = min(counts.values()) if counts else 1
        imbalance_ratio = max_count / max(1, min_count)

        # 1. Automatic Minority Detection
        # Minority if prior < 0.80 * uniform or count < 0.75 * median count
        median_count = float(np.median(list(counts.values())))
        minority_classes = [
            lbl for lbl in label_list
            if priors[lbl] < (0.80 * uniform_prior) or counts[lbl] < (0.75 * median_count)
        ]

        # 2. Dynamic Logit Adjustment (tau)
        # Menon et al. ICML 2020: activate when imbalance ratio exceeds 2.0x
        use_logit_adj = imbalance_ratio > 2.0
        if use_logit_adj:
            logit_adj_tau = float(np.clip(0.45 * np.log10(max(1.0, imbalance_ratio)), 0.35, 1.0))
        else:
            logit_adj_tau = 0.0

        # 3. Test/Val Prior Shift Detection
        eval_logit_adj_tau = 0.0
        if val_df is not None and use_logit_adj and len(val_df) > 0:
            val_total = max(1, len(val_df))
            val_counts = {lbl: int(sum(val_df["label"] == lbl)) for lbl in label_list}
            val_priors = {lbl: val_counts[lbl] / val_total for lbl in label_list}
            for m_lbl in minority_classes:
                # If minority class is noticeably denser in validation/test set than in train set
                if val_priors.get(m_lbl, 0.0) > 1.25 * priors[m_lbl]:
                    eval_logit_adj_tau = round(0.5 * logit_adj_tau, 2)
                    break

        # 4. Target Oversampling (Preventing Batch Gradient Starvation)
        target_oversample = {}
        sorted_counts = sorted(counts.values())
        second_largest = sorted_counts[-2] if len(sorted_counts) >= 2 else sorted_counts[-1]

        for m_lbl in minority_classes:
            expected_per_batch = batch_size * priors[m_lbl]
            if expected_per_batch < 1.5 or counts[m_lbl] < 100:
                ratio_to_med = int(round(second_largest / max(1, counts[m_lbl])))
                multiplier = int(np.clip(ratio_to_med, 2, 8))
                if multiplier > 1:
                    target_oversample[m_lbl] = multiplier
            elif counts[m_lbl] < 0.5 * second_largest:
                multiplier = min(2, int(round(second_largest / max(1, counts[m_lbl]))))
                if multiplier > 1:
                    target_oversample[m_lbl] = multiplier

        if not target_oversample:
            target_oversample = None

        # 5. Adaptive Label Smoothing
        # Regularizes overlapping borderline tokens while protecting rare 1-hot targets
        label_smoothing = float(np.clip(0.08 / np.sqrt(1.0 + 0.5 * np.log10(max(1.0, imbalance_ratio))), 0.03, 0.08))

        # 6. Safety Guard for Class Weights
        # If IR > 10 (extreme imbalance like HoASA), class weights are disabled to protect the 88%+ majority class.
        # Logit Adjustment + Oversampling safely handles the minority without hurting majority precision.
        use_class_weights = imbalance_ratio <= 10.0 and len(minority_classes) > 0

        calibrated = {
            "imbalance_ratio": round(imbalance_ratio, 2),
            "minority_classes": minority_classes,
            "use_logit_adjustment": use_logit_adj,
            "logit_adj_tau": round(logit_adj_tau, 2),
            "eval_logit_adj_tau": eval_logit_adj_tau,
            "target_oversample": target_oversample,
            "label_smoothing": round(label_smoothing, 3),
            "use_class_weights": use_class_weights,
        }

        if verbose:
            print("\n  [AUTO-CALIBRATION] Dataset Imbalance Analysis:")
            print(f"    - Class Counts: {counts}")
            print(f"    - Imbalance Ratio: {calibrated['imbalance_ratio']}x")
            print(f"    - Minority Classes: {calibrated['minority_classes']}")
            print(f"    - Logit Adjustment: {calibrated['use_logit_adjustment']} (tau={calibrated['logit_adj_tau']}, eval_tau={calibrated['eval_logit_adj_tau']})")
            print(f"    - Target Oversample: {calibrated['target_oversample']}")
            print(f"    - Label Smoothing: {calibrated['label_smoothing']}")
            print(f"    - Class Weights: {calibrated['use_class_weights']}")

        return calibrated
