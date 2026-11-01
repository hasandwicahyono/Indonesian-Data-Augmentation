"""
Phase 3: Classification Performance Evaluation with IndoBERT and Foundation Ensembles.
Backward-compatible CLI runner delegating to src.modeling.evaluator.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from src.modeling.adversarial import FGM
from src.modeling.evaluator import IndoBERTClassifierEvaluator
from src.modeling.calibrator import AutoImbalanceCalibrator
from src.data.dataset_manager import IndonesianDatasetsManager

__all__ = [
    "FGM",
    "IndoBERTClassifierEvaluator",
    "AutoImbalanceCalibrator",
]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate IndoBERT models on Indonesian datasets with all enhancements.")
    parser.add_argument("--dataset", type=str, default="EmoT", choices=["EmoT", "HoASA", "MovieOpini", "SmSA"], help="Dataset to evaluate")
    parser.add_argument("--model_name", type=str, default="indobenchmark/indobert-base-p1", help="HuggingFace model path")
    parser.add_argument("--epochs", type=int, default=6, help="Maximum epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--early_stopping", action="store_true", default=True, help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument("--use_preprocessing", action="store_true", default=True, help="Apply Indonesian text preprocessor")
    parser.add_argument("--use_class_weights", action="store_true", default=False, help="Enable class weighting")
    parser.add_argument("--label_smoothing", type=float, default=0.08, help="Label smoothing value")
    parser.add_argument("--use_fgm", action="store_true", default=True, help="Enable FGM adversarial training")
    parser.add_argument("--device", type=str, default=None, help="Compute device (e.g. 'cuda:0', 'mps', 'cpu', 'auto'). If omitted, automatically detects and leverages all available GPUs via DataParallel.")
    parser.add_argument("--use_logit_adjustment", action="store_true", default=False, help="Enable Menon et al. ICML 2020 Logit Adjustment")
    parser.add_argument("--logit_adj_tau", type=float, default=0.8, help="Logit adjustment temperature tau")
    parser.add_argument("--eval_logit_adj_tau", type=float, default=0.0, help="Inference logit adjustment tau")
    parser.add_argument("--negative_oversample", type=int, default=None, help="Oversample factor for negative/minority class (e.g. 8 for HoASA)")
    parser.add_argument("--neutral_oversample", type=int, default=None, help="Oversample factor for neutral class (e.g. 2 for SmSA)")
    parser.add_argument("--auto_calibrate", action="store_true", default=True, help="Automatically detect minority classes, calibrate logit adjustment tau, oversampling, and regularization")
    parser.add_argument("--no_auto_calibrate", action="store_false", dest="auto_calibrate", help="Disable automatic imbalance calibration")
    parser.add_argument("--run_ensemble", action="store_true", default=False, help="Run heterogeneous ensemble pipeline")
    parser.add_argument("--augmented_dir", type=str, default="data/augmented/all_text", help="Directory containing augmented CSVs")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save evaluation Excel")
    args = parser.parse_args()

    manager = IndonesianDatasetsManager()
    target_oversample = None
    use_la = args.use_logit_adjustment
    tau_to_use = args.logit_adj_tau
    eval_tau_to_use = args.eval_logit_adj_tau
    ensemble_configs = None

    if args.dataset == "EmoT":
        ds = manager.load_emot()
        default_model = "indolem/indobertweet-base-uncased"
        if args.run_ensemble:
            ensemble_configs = [
                {"model_name": "indolem/indobertweet-base-uncased", "weight": 0.50},
                {"model_name": "indobenchmark/indobert-base-p1", "weight": 0.30},
                {"model_name": "cahya/bert-base-indonesian-522M", "weight": 0.20},
            ]
    elif args.dataset == "HoASA":
        ds = manager.load_hoasa()
        default_model = "indobenchmark/indobert-base-p1"
        use_la = True
        oversample_val = args.negative_oversample if args.negative_oversample is not None else 8
        target_oversample = {"neg": oversample_val}
    elif args.dataset == "MovieOpini":
        ds = manager.load_movie_opini()
        default_model = "indobenchmark/indobert-base-p1"
    elif args.dataset == "SmSA":
        ds = manager.load_smsa()
        default_model = "indobenchmark/indobert-base-p1"
        use_la = True
        oversample_val = args.neutral_oversample if args.neutral_oversample is not None else 2
        target_oversample = {"neutral": oversample_val}
        if tau_to_use == 0.8 and not args.use_logit_adjustment:
            tau_to_use = 0.6
        if eval_tau_to_use == 0.0:
            eval_tau_to_use = 0.3

    if args.auto_calibrate:
        cal = AutoImbalanceCalibrator.calibrate(
            train_df=ds.train,
            label_list=ds.label_list,
            val_df=ds.val,
            batch_size=args.batch_size,
        )
        if not args.use_logit_adjustment and cal["use_logit_adjustment"]:
            use_la = True
            tau_to_use = cal["logit_adj_tau"]
            eval_tau_to_use = cal["eval_logit_adj_tau"]
        if target_oversample is None and cal["target_oversample"]:
            target_oversample = cal["target_oversample"]
        if args.label_smoothing == 0.08:
            args.label_smoothing = cal["label_smoothing"]
        if not args.use_class_weights:
            args.use_class_weights = cal["use_class_weights"]

    model_to_use = args.model_name if args.model_name != "indobenchmark/indobert-base-p1" else default_model
    out_excel = args.output_path or f"data/results/{args.dataset}_evaluation_results.xlsx"

    aug_dfs = {}
    for metric in ["ECU", "COS", "JAC", "BLEU"]:
        path = os.path.join(args.augmented_dir, f"{args.dataset}-Augmented-{metric}-ALL-Text-Final.csv")
        if os.path.exists(path):
            aug_dfs[metric] = pd.read_csv(path)

    evaluator = IndoBERTClassifierEvaluator(
        model_name=model_to_use,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        early_stopping=args.early_stopping,
        patience=args.patience,
        use_preprocessing=args.use_preprocessing,
        use_class_weights=args.use_class_weights,
        label_smoothing=args.label_smoothing,
        use_fgm=args.use_fgm,
        use_logit_adjustment=use_la,
        logit_adj_tau=tau_to_use,
        eval_logit_adj_tau=eval_tau_to_use,
        target_oversample=target_oversample,
        ensemble_configs=ensemble_configs,
        device=args.device,
    )

    evaluator.evaluate_all_variants(
        base_train_df=ds.train,
        val_df=ds.val,
        base_test_df=ds.test,
        augmented_dfs=aug_dfs,
        label_list=ds.label_list,
        dataset_name=args.dataset,
        output_results_path=out_excel,
        seed=args.seed,
    )
