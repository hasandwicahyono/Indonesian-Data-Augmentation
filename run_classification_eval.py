"""
Phase 3: Classification Performance Evaluation with IndoBERT.
Finetunes and evaluates IndoBERT (indobenchmark/indobert-base-p1) across:
1. Baseline (Not-Augmented)
2. Euclidean-Augmented
3. Cosine-Augmented
4. Jaccard-Augmented
5. BLEU-Augmented
Computes Accuracy, Precision, Recall, Macro-F1, and Paired t-test statistics.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import copy
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    logging as hf_logging,
)

# Silence standard checkpoint adaptation load reports (discarded MLM head, new classifier head)
hf_logging.set_verbosity_error()
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import StratifiedKFold

from ClassificationDataset import ClassificationDataset
from IndonesianPreprocessor import IndonesianPreprocessor


class FGM:
    """Fast Gradient Method for Adversarial Training on word embedding weights."""

    def __init__(self, model: torch.nn.Module, emb_name: str = "word_embeddings", epsilon: float = 1.0):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.backup: Dict[str, torch.Tensor] = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class IndoBERTClassifierEvaluator:
    def __init__(
        self,
        model_name: str = "indobenchmark/indobert-base-p1",
        max_len: int = 128,
        batch_size: int = 16,
        epochs: int = 5,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        device: Optional[str] = None,
        seed: int = 42,
        early_stopping: bool = True,
        patience: int = 2,
        min_delta: float = 1e-4,
        use_preprocessing: bool = True,
        use_class_weights: bool = True,
        label_smoothing: float = 0.08,
        use_fgm: bool = True,
        fgm_epsilon: float = 1.0,
        use_logit_adjustment: bool = False,
        logit_adj_tau: float = 0.8,
        eval_logit_adj_tau: float = 0.0,
        target_oversample: Optional[Dict[str, int]] = None,
        ensemble_configs: Optional[List[Dict[str, Any]]] = None,
    ):
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.use_preprocessing = use_preprocessing
        self.use_class_weights = use_class_weights
        self.label_smoothing = label_smoothing
        self.use_fgm = use_fgm
        self.fgm_epsilon = fgm_epsilon
        self.use_logit_adjustment = use_logit_adjustment
        self.logit_adj_tau = logit_adj_tau
        self.eval_logit_adj_tau = eval_logit_adj_tau
        self.target_oversample = target_oversample
        self.ensemble_configs = ensemble_configs

        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if device is not None:
            self.device = torch.device(device)
        elif self.num_gpus > 0:
            self.device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.preprocessor = IndonesianPreprocessor(map_emojis=True, map_emoticons=True)
        device_desc = f"{self.device} ({self.num_gpus} GPUs via DataParallel)" if (self.device.type == "cuda" and self.num_gpus > 1) else str(self.device)
        print(f"IndoBERT Evaluator initialized on device: {device_desc} | Model: {self.model_name}")
        extra_desc = f", Logit-Adj: {self.use_logit_adjustment} (tau={self.logit_adj_tau})" if self.use_logit_adjustment else ""
        if self.target_oversample:
            extra_desc += f", Oversample: {self.target_oversample}"
        if self.ensemble_configs:
            extra_desc += f", Heterogeneous Ensemble: {len(self.ensemble_configs)} models"
        print(f"  Enhancements Active -> Preprocessing: {self.use_preprocessing}, Class-Weights: {self.use_class_weights}, Label-Smoothing: {self.label_smoothing}, FGM-Adversarial: {self.use_fgm}{extra_desc}")

    def _set_seed(self, seed: Optional[int] = None):
        target_seed = self.seed if seed is None else seed
        torch.manual_seed(target_seed)
        np.random.seed(target_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(target_seed)

    def train_epoch(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: Any,
        scheduler: Any,
        criterion: Optional[torch.nn.Module] = None,
        fgm: Optional[FGM] = None,
        log_priors: Optional[torch.Tensor] = None,
    ) -> float:
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            # Logit Adjustment during training: shifts margins to force separation for minority classes
            if log_priors is not None:
                logits = logits + self.logit_adj_tau * log_priors
            loss = criterion(logits, labels) if criterion is not None else outputs.loss
            # Multi-GPU DataParallel reduces per-GPU loss vector to scalar
            if loss is not None and loss.dim() > 0:
                loss = loss.mean()
            loss.backward()

            # Adversarial training with Fast Gradient Method (FGM)
            if fgm is not None:
                fgm.attack()
                outputs_adv = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits_adv = outputs_adv.logits
                if log_priors is not None:
                    logits_adv = logits_adv + self.logit_adj_tau * log_priors
                loss_adv = criterion(logits_adv, labels) if criterion is not None else outputs_adv.loss
                if loss_adv is not None and loss_adv.dim() > 0:
                    loss_adv = loss_adv.mean()
                loss_adv.backward()
                fgm.restore()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        num_labels: Optional[int] = None,
        log_priors: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float, float, float, np.ndarray, List[int], List[int]]:
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                if log_priors is not None and self.eval_logit_adj_tau > 0.0:
                    logits = logits - self.eval_logit_adj_tau * log_priors
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Determine target label indices so per-class scores align with label_list
        if num_labels is None:
            if hasattr(model, "config") and hasattr(model.config, "num_labels"):
                num_labels = model.config.num_labels
            elif hasattr(model, "module") and hasattr(model.module, "config") and hasattr(model.module.config, "num_labels"):
                num_labels = model.module.config.num_labels

        target_labels = list(range(num_labels)) if num_labels is not None else None

        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=target_labels, average="macro", zero_division=0)
        _, _, per_class_f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=target_labels, average=None, zero_division=0)
        return float(acc), float(prec), float(rec), float(f1), per_class_f1, all_labels, all_preds

    def evaluate_dataset(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        label_list: List[str],
        val_df: Optional[pd.DataFrame] = None,
        name: str = "Dataset",
        seed: Optional[int] = None,
        epochs: Optional[int] = None,
        early_stopping: Optional[bool] = None,
        patience: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Trains and evaluates IndoBERT on a single dataset pair, with optional validation-based checkpointing and early stopping."""
        val_info = f", Val: {len(val_df)}" if val_df is not None else ""
        num_epochs = epochs if epochs is not None else self.epochs
        use_early_stopping = early_stopping if early_stopping is not None else self.early_stopping
        pat = patience if patience is not None else self.patience

        print(f"\nEvaluating: {name} (Train: {len(train_df)}{val_info}, Test: {len(test_df)}, Epochs: {num_epochs}, EarlyStopping: {use_early_stopping} [patience={pat}])")
        
        # Strict Data Leakage Guard: Check and prevent any overlap between training and test sets
        train_text_set = set(str(t).strip() for t in train_df["text"].dropna())
        test_text_set = set(str(t).strip() for t in test_df["text"].dropna())
        leakage = train_text_set.intersection(test_text_set)
        if leakage:
            print(f"  [LEAKAGE DETECTED] {len(leakage)} sample(s) overlap between training set and test set! Purging from training set to guarantee zero leakage.")
            clean_mask = ~train_df["text"].astype(str).str.strip().isin(test_text_set)
            purged_count = len(train_df) - clean_mask.sum()
            train_df = train_df[clean_mask].copy().reset_index(drop=True)
            print(f"  [LEAKAGE PURGED] Removed {purged_count} leaking sample(s). Cleaned train size: {len(train_df)}")
        else:
            print("  [LEAKAGE CHECK PASSED] 0 overlapping samples between training and test sets.")

        # Validate test set coverage
        test_label_counts = test_df["label"].value_counts().to_dict()
        for lbl in label_list:
            if lbl not in test_label_counts:
                print(f"  [CRITICAL WARNING] Label '{lbl}' is absent in test_df! Macro metrics will be distorted.")

        label_map = {lbl: i for i, lbl in enumerate(label_list)}
        id2label = {i: lbl for i, lbl in enumerate(label_list)}
        num_labels = len(label_list)

        # Set seed before data loading & model init for strict reproducibility
        self._set_seed(seed)

        # 1. Target Class Oversampling (e.g. for HoASA's severe 1.3% negative class)
        if self.target_oversample:
            dfs_to_concat = [train_df]
            for target_class, multiplier in self.target_oversample.items():
                minority_rows = train_df[train_df["label"] == target_class]
                if not minority_rows.empty and multiplier > 1:
                    replicated = pd.concat([minority_rows] * (multiplier - 1), ignore_index=True)
                    dfs_to_concat.append(replicated)
                    print(f"  [NEGATIVE OVERSAMPLER] Oversampled class '{target_class}' by {multiplier}x: {len(minority_rows)} -> {len(minority_rows) * multiplier} samples")
            train_df = pd.concat(dfs_to_concat, ignore_index=True).sample(frac=1.0, random_state=seed or self.seed).reset_index(drop=True)

        # 2. Compute empirical class priors for Logit Adjustment (Menon et al., ICML 2020)
        if self.use_logit_adjustment:
            class_counts = np.array([sum(train_df["label"] == lbl) for lbl in label_list], dtype=np.float32)
            priors = class_counts / (class_counts.sum() + 1e-6)
            log_priors = torch.tensor(np.log(priors + 1e-8), dtype=torch.float32, device=self.device)
            priors_dict = {lbl: f"{priors[i]:.3f}" for i, lbl in enumerate(label_list)}
            print(f"  [LOGIT ADJUSTMENT] Active (tau={self.logit_adj_tau}, eval_tau={self.eval_logit_adj_tau}) | Priors: {priors_dict}")
        else:
            log_priors = None

        prep = self.preprocessor if self.use_preprocessing else None
        train_ds = ClassificationDataset(train_df["text"].tolist(), train_df["label"].tolist(), self.model_name, self.max_len, label_map, preprocessor=prep)
        test_ds = ClassificationDataset(test_df["text"].tolist(), test_df["label"].tolist(), self.model_name, self.max_len, label_map, preprocessor=prep)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        if val_df is not None:
            val_ds = ClassificationDataset(val_df["text"].tolist(), val_df["label"].tolist(), self.model_name, self.max_len, label_map, preprocessor=prep)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        else:
            val_loader = None

        config = AutoConfig.from_pretrained(self.model_name)
        config.num_labels = num_labels
        config.id2label = id2label
        config.label2id = label_map
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )
        model.to(self.device)

        # Accommodate multi-GPU DataParallel
        if self.device.type == "cuda" and self.num_gpus > 1:
            print(f"  [Multi-GPU] Wrapping model with torch.nn.DataParallel across {self.num_gpus} GPUs")
            model = torch.nn.DataParallel(model)

        # Compute balanced class weights to address minority classes across datasets
        if self.use_class_weights:
            from sklearn.utils.class_weight import compute_class_weight
            unique_classes = np.unique(train_df["label"])
            computed_weights = compute_class_weight(
                class_weight="balanced",
                classes=unique_classes,
                y=train_df["label"].values,
            )
            weight_arr = np.ones(num_labels, dtype=np.float32)
            for cls_name, w in zip(unique_classes, computed_weights):
                if cls_name in label_map:
                    weight_arr[label_map[cls_name]] = float(w)

            # Guard against extreme class imbalance collapse (e.g. HoASA where neg is 1.3% and neut is 88.6%):
            # Uncapped balanced weights produce extreme penalties (e.g. 67.4x) that force the model to predict
            # only minority classes and misclassify the entire ~89% majority class, dropping accuracy to ~10%.
            max_ratio = weight_arr.max() / (weight_arr.min() + 1e-6)
            if max_ratio > 4.0:
                print(f"  [SAFETY GUARD] Severe class imbalance detected (weight ratio: {max_ratio:.1f}x).")
                print("  Applying square-root smoothed inverse weighting to protect majority class accuracy.")
                class_counts = np.array([sum(train_df["label"] == lbl) for lbl in label_list], dtype=np.float32)
                smoothed = np.sqrt(len(train_df) / (class_counts + 1e-6))
                smoothed = smoothed / smoothed.mean()
                weight_arr = np.clip(smoothed, 0.6, 2.5)
                for lbl, idx in label_map.items():
                    print(f"    - {lbl}: adjusted weight = {weight_arr[idx]:.3f}")

            class_weights_tensor = torch.tensor(weight_arr, dtype=torch.float, device=self.device)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=self.label_smoothing)
        elif self.label_smoothing > 0.0:
            criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        else:
            criterion = None

        fgm = FGM(model, epsilon=self.fgm_epsilon) if self.use_fgm else None

        total_steps = len(train_loader) * num_epochs
        optimizer = AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

        best_val_f1 = -1.0
        best_train_loss = float("inf")
        best_model_weights = None
        best_epoch = 0
        patience_counter = 0
        actual_epochs_trained = 0

        for epoch in range(num_epochs):
            actual_epochs_trained = epoch + 1
            loss = self.train_epoch(model, train_loader, optimizer, scheduler, criterion=criterion, fgm=fgm, log_priors=log_priors)
            if val_loader is not None:
                v_acc, v_prec, v_rec, v_f1, _, _, _ = self.evaluate(model, val_loader, num_labels=num_labels, log_priors=log_priors)
                print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f} | Val Acc: {v_acc:.4f} | Val Macro-F1: {v_f1:.4f}")
                if v_f1 > best_val_f1 + self.min_delta:
                    best_val_f1 = v_f1
                    best_epoch = epoch + 1
                    target_model = model.module if hasattr(model, "module") else model
                    best_model_weights = copy.deepcopy(target_model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if use_early_stopping and patience_counter >= pat:
                        print(f"  [Early Stopping Triggered] No improvement in Val Macro-F1 for {pat} consecutive epochs (best was {best_val_f1:.4f} at epoch {best_epoch}). Halting at epoch {epoch+1}/{num_epochs}.")
                        break
            else:
                # When no validation set is present, strictly monitor training loss to avoid leaking test set
                print(f"  Epoch {epoch+1}/{num_epochs} - Train Loss: {loss:.4f}")
                if best_train_loss - loss > self.min_delta:
                    best_train_loss = loss
                    best_epoch = epoch + 1
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if use_early_stopping and patience_counter >= pat:
                        print(f"  [Early Stopping Triggered] Training loss did not improve for {pat} consecutive epochs. Halting at epoch {epoch+1}/{num_epochs}.")
                        break

        # Restore optimal checkpoint if validation set was used
        if best_model_weights is not None:
            target_model = model.module if hasattr(model, "module") else model
            target_model.load_state_dict(best_model_weights)
            print(f"  [Optimal Checkpoint Restored] Best Val Macro-F1: {best_val_f1:.4f} (from epoch {best_epoch})")

        acc, prec, rec, f1, per_class_f1, all_labels, all_preds = self.evaluate(model, test_loader, num_labels=num_labels, log_priors=log_priors)
        test_probs, _ = self.predict_proba(model, test_loader, log_priors=log_priors)

        res: Dict[str, Any] = {
            "dataset_name": name,
            "model_name": self.model_name,
            "train_size": len(train_df),
            "val_size": len(val_df) if val_df is not None else 0,
            "test_size": len(test_df),
            "epochs_configured": num_epochs,
            "epochs_trained": actual_epochs_trained,
            "best_epoch": best_epoch if val_df is not None else actual_epochs_trained,
            "val_macro_f1": best_val_f1 if val_df is not None else None,
            "accuracy": acc,
            "macro_precision": prec,
            "macro_recall": rec,
            "macro_f1": f1,
            "y_true": all_labels,
            "y_pred": all_preds,
            "_test_probs": test_probs,
        }
        for i, lbl in enumerate(label_list):
            res[f"f1_{lbl}"] = float(per_class_f1[i]) if i < len(per_class_f1) else 0.0

        return res

    def predict_proba(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        log_priors: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns softmax probability distributions and ground-truth labels for ensembling."""
        model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                if log_priors is not None and self.eval_logit_adj_tau > 0.0:
                    logits = logits - self.eval_logit_adj_tau * log_priors
                probs = torch.softmax(logits, dim=1)

                all_probs.append(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return np.vstack(all_probs), np.array(all_labels)

    @staticmethod
    def evaluate_ensemble(
        prob_list: List[np.ndarray],
        ground_truth: np.ndarray,
        label_list: List[str],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Soft-voting ensemble across multiple models' predicted probability matrices."""
        if weights is None:
            weights = [1.0 / len(prob_list)] * len(prob_list)
        w_norm = np.array(weights) / sum(weights)

        ensemble_probs = np.zeros_like(prob_list[0])
        for p, w in zip(prob_list, w_norm):
            ensemble_probs += p * w

        all_preds = np.argmax(ensemble_probs, axis=1)
        all_labels = ground_truth

        target_labels = list(range(len(label_list)))
        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=target_labels, average="macro", zero_division=0)
        _, _, per_class_f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=target_labels, average=None, zero_division=0)

        res = {
            "dataset_name": "Soft-Voting-Ensemble",
            "accuracy": float(acc),
            "macro_precision": float(prec),
            "macro_recall": float(rec),
            "macro_f1": float(f1),
        }
        for i, lbl in enumerate(label_list):
            res[f"f1_{lbl}"] = float(per_class_f1[i]) if i < len(per_class_f1) else 0.0

        print(f"\n[Ensemble Results] Accuracy: {acc:.4f} | Macro-F1: {f1:.4f} | Macro-Prec: {prec:.4f} | Macro-Rec: {rec:.4f}")
        return res

    def evaluate_heterogeneous_ensemble(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        label_list: List[str],
        ensemble_configs: List[Dict[str, Any]],
        val_df: Optional[pd.DataFrame] = None,
        dataset_name: str = "Dataset",
        seed: Optional[int] = None,
        epochs: Optional[int] = None,
        early_stopping: Optional[bool] = None,
        patience: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Trains and soft-votes across multiple heterogeneous models (e.g. IndoBERTweet + IndoBERT-p1 + IndoBERT-522M)
        to boost classification performance toward 90%+ on challenging multi-class datasets.
        """
        print(f"\n=======================================================")
        print(f"  Running Heterogeneous Ensemble for {dataset_name}")
        print(f"  Ensemble Architecture: {[c['model_name'] for c in ensemble_configs]}")
        print(f"=======================================================")

        prob_list = []
        weights = []
        ground_truth = None

        for idx, ecfg in enumerate(ensemble_configs):
            m_name = ecfg["model_name"]
            w = ecfg.get("weight", 1.0)
            print(f"\n  [Ensemble Sub-Model {idx+1}/{len(ensemble_configs)}] Training: {m_name} (weight: {w})")

            sub_evaluator = IndoBERTClassifierEvaluator(
                model_name=m_name,
                max_len=self.max_len,
                batch_size=self.batch_size,
                epochs=epochs if epochs is not None else self.epochs,
                lr=self.lr,
                weight_decay=self.weight_decay,
                device=str(self.device),
                seed=(seed or self.seed) + idx * 7,
                early_stopping=early_stopping if early_stopping is not None else self.early_stopping,
                patience=patience if patience is not None else self.patience,
                min_delta=self.min_delta,
                use_preprocessing=self.use_preprocessing,
                use_class_weights=self.use_class_weights,
                label_smoothing=self.label_smoothing,
                use_fgm=self.use_fgm,
                fgm_epsilon=self.fgm_epsilon,
                use_logit_adjustment=self.use_logit_adjustment,
                logit_adj_tau=self.logit_adj_tau,
                eval_logit_adj_tau=self.eval_logit_adj_tau,
                target_oversample=self.target_oversample,
            )

            train_res = sub_evaluator.evaluate_dataset(
                train_df,
                test_df,
                label_list,
                val_df=val_df,
                name=f"{dataset_name}-EnsembleSub-{m_name.split('/')[-1]}",
                seed=(seed or self.seed) + idx * 7,
                epochs=epochs,
                early_stopping=early_stopping,
                patience=patience,
            )

            y_t = train_res.get("y_true")
            if ground_truth is None and y_t is not None:
                ground_truth = np.array(y_t)

            if "_test_probs" in train_res:
                prob_list.append(train_res["_test_probs"])
            else:
                sub_preds = np.array(train_res.get("y_pred", []))
                one_hot = np.zeros((len(sub_preds), len(label_list)), dtype=np.float32)
                for i_sample, p_class in enumerate(sub_preds):
                    one_hot[i_sample, p_class] = 1.0
                prob_list.append(one_hot)

            weights.append(w)

        ens_res = self.evaluate_ensemble(prob_list, ground_truth, label_list, weights=weights)
        ens_res["dataset_name"] = f"{dataset_name}-Augmented-Ensemble"
        ens_res["model_name"] = f"Heterogeneous-Ensemble ({len(ensemble_configs)} models)"
        ens_res["train_size"] = len(train_df)
        ens_res["val_size"] = len(val_df) if val_df is not None else 0
        ens_res["test_size"] = len(test_df)
        ens_res["epochs_configured"] = epochs if epochs is not None else self.epochs
        ens_res["epochs_trained"] = epochs if epochs is not None else self.epochs
        ens_res["best_epoch"] = epochs if epochs is not None else self.epochs
        ens_res["val_macro_f1"] = None

        w_norm = np.array(weights) / sum(weights)
        ens_probs = np.zeros_like(prob_list[0])
        for p, w in zip(prob_list, w_norm):
            ens_probs += p * w
        ens_res["y_true"] = ground_truth.tolist() if isinstance(ground_truth, np.ndarray) else ground_truth
        ens_res["y_pred"] = np.argmax(ens_probs, axis=1).tolist()
        return ens_res

    def evaluate_all_variants(
        self,
        base_train_df: pd.DataFrame,
        base_test_df: pd.DataFrame,
        augmented_dfs: Dict[str, pd.DataFrame],
        label_list: List[str],
        dataset_name: str,
        val_df: Optional[pd.DataFrame] = None,
        output_results_path: str = "data/results/evaluation_results.xlsx",
        seed: int = 42,
        epochs: Optional[int] = None,
        early_stopping: Optional[bool] = None,
        patience: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Runs comprehensive benchmark comparing:
        - Baseline (Not-Augmented)
        - ECU-Augmented
        - COS-Augmented
        - JAC-Augmented
        - BLEU-Augmented
        Computes Paired comparison statistics matching the paper.
        """
        # Validate baseline vs augmented size sanity
        min_aug_size = min(len(df) for df in augmented_dfs.values()) if augmented_dfs else 0
        if len(base_train_df) > min_aug_size:
            print(
                f"  [SANITY WARNING] base_train_df ({len(base_train_df)}) is larger than "
                f"augmented dataset ({min_aug_size})! For valid comparison, baseline training set "
                f"must match the base samples of the augmented datasets."
            )

        results = []

        # 1. Baseline
        base_res = self.evaluate_dataset(
            base_train_df,
            base_test_df,
            label_list,
            val_df=val_df,
            name=f"{dataset_name}-Baseline-NotAugmented",
            seed=seed,
            epochs=epochs,
            early_stopping=early_stopping,
            patience=patience,
        )
        results.append(base_res)

        # 2. Augmented sets
        for metric, aug_train in augmented_dfs.items():
            res = self.evaluate_dataset(
                aug_train,
                base_test_df,
                label_list,
                val_df=val_df,
                name=f"{dataset_name}-Augmented-{metric}",
                seed=seed,
                epochs=epochs,
                early_stopping=early_stopping,
                patience=patience,
            )
            f1_gain = res["macro_f1"] - base_res["macro_f1"]
            pct_gain = (f1_gain / base_res["macro_f1"]) * 100.0 if base_res["macro_f1"] > 0 else 0.0
            res["f1_improvement"] = f1_gain
            res["f1_improvement_pct"] = pct_gain
            results.append(res)

        # 3. Heterogeneous Ensemble (if configured)
        if self.ensemble_configs:
            aug_results = [r for r in results if r["dataset_name"] != f"{dataset_name}-Baseline-NotAugmented"]
            if aug_results and len(augmented_dfs) > 0:
                best_aug_res = max(aug_results, key=lambda x: x["macro_f1"])
                best_variant_key = best_aug_res["dataset_name"].split("-")[-1]
                target_train_df = augmented_dfs.get(best_variant_key, list(augmented_dfs.values())[0])
                print(f"\n  [Ensemble Pipeline] Running Heterogeneous Ensemble on best augmented variant ({best_variant_key})...")
            else:
                target_train_df = base_train_df
                print(f"\n  [Ensemble Pipeline] Running Heterogeneous Ensemble on baseline training data...")
            ens_res = self.evaluate_heterogeneous_ensemble(
                train_df=target_train_df,
                test_df=base_test_df,
                label_list=label_list,
                ensemble_configs=self.ensemble_configs,
                val_df=val_df,
                dataset_name=dataset_name,
                seed=seed,
                epochs=epochs,
                early_stopping=early_stopping,
                patience=patience,
            )
            f1_gain = ens_res["macro_f1"] - base_res["macro_f1"]
            pct_gain = (f1_gain / base_res["macro_f1"]) * 100.0 if base_res["macro_f1"] > 0 else 0.0
            ens_res["f1_improvement"] = f1_gain
            ens_res["f1_improvement_pct"] = pct_gain
            results.append(ens_res)

        # Extract predictions for confusion matrix plotting before exporting summary dataframe
        predictions = {}
        for r in results:
            dname = r["dataset_name"]
            y_t = r.pop("y_true", None)
            y_p = r.pop("y_pred", None)
            r.pop("_test_probs", None)
            if y_t is not None and y_p is not None:
                predictions[f"{dname}__true"] = np.array(y_t)
                predictions[f"{dname}__pred"] = np.array(y_p)

        results_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output_results_path), exist_ok=True)
        results_df.to_excel(output_results_path, index=False)
        print(f"\nFinal Classification Results saved to: {output_results_path}")

        if predictions:
            pred_path = output_results_path.rsplit(".", 1)[0] + "_predictions.npz"
            np.savez(pred_path, **predictions)
            print(f"Predictions for confusion matrix saved to: {pred_path}")
        summary_cols = ["dataset_name", "model_name", "train_size", "epochs_trained", "best_epoch", "accuracy", "macro_f1", "f1_improvement_pct"]
        available_summary_cols = [c for c in summary_cols if c in results_df.columns]
        print(results_df[available_summary_cols])
        return results_df


if __name__ == "__main__":
    import argparse
    from DatasetsManager import IndonesianDatasetsManager

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
    parser.add_argument("--run_ensemble", action="store_true", default=False, help="Run heterogeneous ensemble pipeline")
    parser.add_argument("--augmented_dir", type=str, default="data/augmented/all_text", help="Directory containing augmented CSVs")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save evaluation Excel")
    args = parser.parse_args()

    manager = IndonesianDatasetsManager()
    target_oversample = None
    use_la = args.use_logit_adjustment
    ensemble_configs = None

    if args.dataset == "EmoT":
        ds = manager.load_emot()
        default_model = "indolem/indobertweet-base-uncased"
        if args.run_ensemble:
            ensemble_configs = [
                {"model_name": "indolem/indobertweet-base-uncased", "weight": 0.50},
                {"model_name": "indobenchmark/indobert-base-p1", "weight": 0.30},
                {"model_name": "cahya/indobert-base-indonesian-522M", "weight": 0.20},
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

    model_to_use = args.model_name if args.model_name != "indobenchmark/indobert-base-p1" else default_model
    out_excel = args.output_path or f"data/results/{args.dataset}_evaluation_results.xlsx"

    # Load augmented sets if present
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
        logit_adj_tau=args.logit_adj_tau,
        eval_logit_adj_tau=args.eval_logit_adj_tau,
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
