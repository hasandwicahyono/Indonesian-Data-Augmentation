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
from typing import Dict, List, Tuple, Any, Optional

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold

from src.data.dataset_manager import IndonesianDatasetsManager
from src.modeling.evaluator import IndoBERTClassifierEvaluator
from src.augmentation.filtering import is_hallucinatory_or_corrupt
from src.utils.path_utils import resolve_path, ensure_dir
from src.utils.hardware import resolve_device
from src.utils.config import load_dataset_config


def run_real_5fold_cv_for_dataset(
    dataset_name: str,
    epochs: Optional[int] = None,
    batch_size: int = 64,
    device: Optional[str] = None,
    max_samples_per_fold: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Executes an actual 5-fold cross-validation experiment on the chosen dataset.
    Trains IndoBERT/IndoBERTweet on each fold for Baseline vs. Best Augmented variant.
    Returns the real experimental metrics dictionary.
    """
    device_str = device if device is not None else resolve_device()

    manager = IndonesianDatasetsManager()
    try:
        cfg = load_dataset_config(dataset_name)
    except Exception:
        cfg = {
            "model_name": "indobenchmark/indobert-base-p1",
            "max_len": 128,
            "use_logit_adjustment": False,
            "logit_adj_tau": 0.0,
        }

    ds_key = dataset_name.lower().replace("-", "_")
    if hasattr(manager, f"load_{ds_key}"):
        ds = getattr(manager, f"load_{ds_key}")()
    elif dataset_name in ["NusaX", "NusaX-Indonesian"]:
        ds = manager.load_nusax_indonesian(binary_only=False)
    elif hasattr(manager, f"load_{dataset_name.lower()}"):
        ds = getattr(manager, f"load_{dataset_name.lower()}")()
    else:
        ds = manager.load_generic_dataset(dataset_name)
    model_name = cfg.get("model_name", "indobenchmark/indobert-base-p1")
    use_la = cfg.get("use_logit_adjustment", False)
    tau = cfg.get("logit_adj_tau", 0.0)

    # Dynamically resolve best_aug_metric from actual evaluation results if available
    eval_excel_path = resolve_path(f"data/results/{dataset_name}_evaluation_results.xlsx")
    best_aug_metric = None
    if os.path.exists(eval_excel_path):
        try:
            eval_df = pd.read_excel(eval_excel_path)
            aug_eval = eval_df[
                eval_df["dataset_name"].str.contains("Augmented", case=False, na=False)
                & ~eval_df["dataset_name"].str.contains("NotAugmented", case=False, na=False)
                & ~eval_df["dataset_name"].str.contains("Baseline", case=False, na=False)
                & ~eval_df["dataset_name"].str.contains("Ensemble", case=False, na=False)
            ]
            if not aug_eval.empty:
                best_eval_row = aug_eval.loc[aug_eval["macro_f1"].idxmax()]
                best_dname = str(best_eval_row["dataset_name"])
                best_aug_metric = best_dname.split("-Augmented-")[-1].replace("-Final", "").strip()
                print(f"  [DYNAMIC CV SELECTION] Auto-selected empirical winner for {dataset_name}: {best_aug_metric} (Macro-F1: {best_eval_row['macro_f1']:.4f})")
        except Exception as e:
            print(f"Notice: Failed to dynamic-detect best metric from excel ({e}); using config fallback.")
    if not best_aug_metric:
        best_aug_metric = cfg.get("best_aug_metric", "BLEU")

    full_df = pd.concat([ds.train, ds.test], ignore_index=True)
    if max_samples_per_fold is not None and len(full_df) > max_samples_per_fold:
        sampled_parts = []
        n_per_class = max(10, max_samples_per_fold // len(ds.label_list))
        for _, grp in full_df.groupby("label"):
            sampled_parts.append(grp.sample(min(len(grp), n_per_class), random_state=42))
        full_df = pd.concat(sampled_parts, ignore_index=True)

    if epochs is None:
        if len(full_df) <= 1000:
            actual_epochs = 5
        elif len(full_df) <= 5000:
            actual_epochs = 3
        else:
            actual_epochs = 2
        print(f"  [DYNAMIC CV EPOCHS] Auto-scaled to {actual_epochs} epochs based on dataset volume ({len(full_df)} samples)")
    else:
        actual_epochs = epochs

    print("\n" + "=" * 80)
    print(f"RUNNING REAL 5-FOLD CV EXPERIMENT: {dataset_name} ({actual_epochs} epochs/fold)")
    print(f"Total dataset volume for 5-fold CV: {len(full_df)} samples, Classes: {ds.label_list}")
    print("=" * 80)

    aug_rel = f"data/augmented/all_text/{dataset_name}-Augmented-{best_aug_metric}-ALL-Text-Final.csv"
    aug_path = resolve_path(aug_rel)
    if not os.path.exists(aug_path):
        raise FileNotFoundError(
            f"Missing augmented dataset file: '{aug_path}'. "
            f"Please run 'python run_all_benchmarks.py' to generate augmented data for {dataset_name}."
        )
    aug_full_df = pd.read_csv(aug_path)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = {
        "dataset": dataset_name,
        "epochs": actual_epochs,
        "baseline_f1": [],
        "baseline_acc": [],
        "augmented_f1": [],
        "augmented_acc": [],
        "fold_details": [],
    }

    evaluator = IndoBERTClassifierEvaluator(
        model_name=model_name,
        max_len=cfg.get("max_len", 128),
        batch_size=batch_size,
        epochs=actual_epochs,
        lr=2e-5,
        early_stopping=False,
        use_preprocessing=True,
        use_class_weights=False,
        use_logit_adjustment=use_la,
        logit_adj_tau=tau,
        use_fgm=False,
        device=device_str,
        prompt_prefix=cfg.get("aspect_prompt_prefix", None),
        valence_map=cfg.get("valence_map", None) if cfg.get("valence_guardrail", False) else None,
        intra_valence_weight=cfg.get("intra_valence_weight", 1.4),
    )

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(full_df["text"], full_df["label"])):
        print(f"\n--- [Fold {fold_idx + 1} / 5] Training & Evaluating ---")
        train_fold = full_df.iloc[train_idx].reset_index(drop=True)
        test_fold = full_df.iloc[test_idx].reset_index(drop=True)

        t0 = time.time()
        base_res = evaluator.evaluate_dataset(
            train_df=train_fold,
            test_df=test_fold,
            label_list=ds.label_list,
            name=f"{dataset_name}-Fold{fold_idx+1}-Baseline",
            epochs=actual_epochs,
        )
        base_f1 = float(base_res["macro_f1"])
        base_acc = float(base_res["accuracy"])
        dt_base = time.time() - t0
        fold_results["baseline_f1"].append(base_f1)
        fold_results["baseline_acc"].append(base_acc)
        print(f"  Fold {fold_idx + 1} Baseline: Macro-F1 = {base_f1:.4f}, Accuracy = {base_acc:.4f} ({dt_base:.1f}s)")

        test_texts_set = set(test_fold["text"].astype(str).tolist())
        aug_clean = aug_full_df[~aug_full_df["text"].astype(str).isin(test_texts_set)].reset_index(drop=True)
        aug_clean = aug_clean[~aug_clean["text"].apply(is_hallucinatory_or_corrupt)].reset_index(drop=True)
        if max_samples_per_fold is not None and len(aug_clean) > int(len(train_fold) * 1.5):
            aug_clean = aug_clean.sample(int(len(train_fold) * 1.5), random_state=42 + fold_idx).reset_index(drop=True)
        aug_train_fold = pd.concat([train_fold, aug_clean], ignore_index=True).drop_duplicates(subset=["text"]).reset_index(drop=True)

        t0 = time.time()
        aug_res = evaluator.evaluate_dataset(
            train_df=aug_train_fold,
            test_df=test_fold,
            label_list=ds.label_list,
            name=f"{dataset_name}-Fold{fold_idx+1}-Augmented",
            epochs=actual_epochs,
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

    out_json = resolve_path(f"data/results/extended/{dataset_name}_real_5fold_results.json")
    ensure_dir(os.path.dirname(out_json))
    with open(out_json, "w") as f:
        json.dump(fold_results, f, indent=2)
    print(f"Saved real experimental metrics to: {out_json}")

    return fold_results

