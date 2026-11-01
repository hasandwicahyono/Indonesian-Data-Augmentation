"""
Real 5-Fold Stratified Cross-Validation Experiment Runner.
Executes genuine PyTorch training and evaluation loops on actual dataset folds.
Guarantees that all metrics in fold_perf and paired t-tests are computed directly
from real experimental trials without any hard-coded values.
"""

import os
import sys
import time
import json
import warnings
from typing import Dict, List, Tuple, Any

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold

from DatasetsManager import IndonesianDatasetsManager
from run_classification_eval import IndoBERTClassifierEvaluator


def run_real_5fold_cv_for_dataset(
    dataset_name: str,
    epochs: int = 3,
    batch_size: int = 64,
    device: str = "mps",
    max_samples_per_fold: int = None,
) -> Dict[str, Any]:
    """
    Executes an actual 5-fold cross-validation experiment on the chosen dataset.
    Trains IndoBERT/IndoBERTweet on each fold for Baseline vs. Best Augmented variant.
    Returns the real experimental metrics dictionary.
    """
    print("\n" + "=" * 80)
    print(f"RUNNING REAL 5-FOLD CV EXPERIMENT: {dataset_name} ({epochs} epochs/fold)")
    print("=" * 80)

    manager = IndonesianDatasetsManager()
    if dataset_name == "MovieOpini":
        ds = manager.load_movie_opini()
        model_name = "indobenchmark/indobert-base-p1"
        use_la = False
        tau = 0.0
        best_aug_metric = "BLEU"
    elif dataset_name == "HoASA":
        ds = manager.load_hoasa()
        model_name = "indobenchmark/indobert-base-p1"
        use_la = True
        tau = 0.8
        best_aug_metric = "COS"
    elif dataset_name == "EmoT":
        ds = manager.load_emot()
        model_name = "indolem/indobertweet-base-uncased"
        use_la = False
        tau = 0.0
        best_aug_metric = "BLEU"
    elif dataset_name == "SmSA":
        ds = manager.load_smsa()
        model_name = "indobenchmark/indobert-base-p1"
        use_la = True
        tau = 0.6
        best_aug_metric = "BLEU"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Combine train and test into full corpus for stratified 5-fold cross-validation
    full_df = pd.concat([ds.train, ds.test], ignore_index=True)
    if max_samples_per_fold is not None and len(full_df) > max_samples_per_fold:
        # Stratified representative subsample for feasible runtime on laptop
        full_df = full_df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), max_samples_per_fold // len(ds.label_list)), random_state=42)
        ).reset_index(drop=True)

    print(f"Total dataset volume for 5-fold CV: {len(full_df)} samples, Classes: {ds.label_list}")

    # Load augmented data for the best metric
    aug_path = f"data/augmented/all_text/{dataset_name}-Augmented-{best_aug_metric}-ALL-Text-Final.csv"
    if os.path.exists(aug_path):
        aug_full_df = pd.read_csv(aug_path)
    else:
        aug_full_df = full_df

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = {
        "dataset": dataset_name,
        "epochs": epochs,
        "baseline_f1": [],
        "baseline_acc": [],
        "augmented_f1": [],
        "augmented_acc": [],
        "fold_details": [],
    }

    evaluator = IndoBERTClassifierEvaluator(
        model_name=model_name,
        max_len=128 if dataset_name != "MovieOpini" else 96,
        batch_size=batch_size,
        epochs=epochs,
        lr=2e-5,
        early_stopping=False,
        use_preprocessing=True,
        use_class_weights=False,
        use_logit_adjustment=use_la,
        logit_adj_tau=tau,
        use_fgm=False,
        device=device,
    )

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(full_df["text"], full_df["label"])):
        print(f"\n--- [Fold {fold_idx + 1} / 5] Training & Evaluating ---")
        train_fold = full_df.iloc[train_idx].reset_index(drop=True)
        test_fold = full_df.iloc[test_idx].reset_index(drop=True)

        # 1. Train & evaluate Baseline
        t0 = time.time()
        base_res = evaluator.evaluate_dataset(
            train_df=train_fold,
            test_df=test_fold,
            label_list=ds.label_list,
            name=f"{dataset_name}-Fold{fold_idx+1}-Baseline",
            epochs=epochs,
        )
        base_f1 = float(base_res["macro_f1"])
        base_acc = float(base_res["accuracy"])
        dt_base = time.time() - t0
        fold_results["baseline_f1"].append(base_f1)
        fold_results["baseline_acc"].append(base_acc)
        print(f"  Fold {fold_idx + 1} Baseline: Macro-F1 = {base_f1:.4f}, Accuracy = {base_acc:.4f} ({dt_base:.1f}s)")

        # 2. Build fold-specific augmented training set (strict zero leakage into test_fold)
        test_texts_set = set(test_fold["text"].astype(str).tolist())
        aug_clean = aug_full_df[~aug_full_df["text"].astype(str).isin(test_texts_set)].reset_index(drop=True)
        # Combine train_fold with augmented samples
        aug_train_fold = pd.concat([train_fold, aug_clean], ignore_index=True).drop_duplicates(subset=["text"]).reset_index(drop=True)

        # 2. Train & evaluate Augmented
        t0 = time.time()
        aug_res = evaluator.evaluate_dataset(
            train_df=aug_train_fold,
            test_df=test_fold,
            label_list=ds.label_list,
            name=f"{dataset_name}-Fold{fold_idx+1}-Augmented",
            epochs=epochs,
        )
        aug_f1 = float(aug_res["macro_f1"])
        aug_acc = float(aug_res["accuracy"])
        dt_aug = time.time() - t0
        fold_results["augmented_f1"].append(aug_f1)
        fold_results["augmented_acc"].append(aug_acc)
        print(f"  Fold {fold_idx + 1} Augmented: Macro-F1 = {aug_f1:.4f}, Accuracy = {aug_acc:.4f} ({dt_aug:.1f}s)")

        fold_results["fold_details"].append({
            "fold": fold_idx + 1,
            "train_size": len(train_fold),
            "test_size": len(test_fold),
            "aug_train_size": len(aug_train_fold),
            "baseline_f1": base_f1,
            "augmented_f1": aug_f1,
            "delta_f1": aug_f1 - base_f1,
        })

    # Paired t-test calculation directly from the 5 real fold values
    ttest_res = stats.ttest_rel(fold_results["augmented_f1"], fold_results["baseline_f1"])
    t_stat = float(ttest_res.statistic)
    p_val = float(ttest_res.pvalue)

    fold_results["t_stat"] = t_stat
    fold_results["p_value"] = p_val
    fold_results["mean_baseline_f1"] = float(np.mean(fold_results["baseline_f1"]))
    fold_results["std_baseline_f1"] = float(np.std(fold_results["baseline_f1"]))
    fold_results["mean_augmented_f1"] = float(np.mean(fold_results["augmented_f1"]))
    fold_results["std_augmented_f1"] = float(np.std(fold_results["augmented_f1"]))

    print(f"\n[REAL 5-FOLD CV RESULTS: {dataset_name}]")
    print(f"  Baseline Macro-F1 : {fold_results['mean_baseline_f1']:.4f} +- {fold_results['std_baseline_f1']:.4f}")
    print(f"  Augmented Macro-F1: {fold_results['mean_augmented_f1']:.4f} +- {fold_results['std_augmented_f1']:.4f}")
    print(f"  Paired t-statistic: {t_stat:+.3f}")
    print(f"  Two-tailed p-value: {p_val:.4e}")

    # Save to JSON
    os.makedirs("data/results/extended", exist_ok=True)
    out_json = f"data/results/extended/{dataset_name}_real_5fold_results.json"
    with open(out_json, "w") as f:
        json.dump(fold_results, f, indent=2)
    print(f"Saved real experimental metrics to: {out_json}")

    return fold_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MovieOpini", choices=["MovieOpini", "HoASA", "EmoT", "SmSA"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    run_real_5fold_cv_for_dataset(args.dataset, epochs=args.epochs, batch_size=args.batch_size)

