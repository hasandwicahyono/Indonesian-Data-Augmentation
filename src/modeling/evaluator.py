"""
Phase 3: Classification Performance Evaluation with IndoBERT and Foundation Ensembles.
Finetunes and evaluates IndoBERT (indobenchmark/indobert-base-p1), IndoBERTweet, and ensembles across:
1. Baseline (Raw ERM)
2. Calibrated Baseline (AIC)
3. Euclidean-Augmented
4. Cosine-Augmented
5. Jaccard-Augmented
6. BLEU-Augmented
7. Heterogeneous Ensembles
Computes Accuracy, Precision, Recall, Macro-F1, per-class F1, and Paired statistics.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ or "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:128"
import gc
import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.parallel.scatter_gather")
warnings.filterwarnings("ignore", message=".*Was asked to gather along dimension 0.*")
import random
import copy
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    logging as hf_logging,
)

hf_logging.set_verbosity_error()
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.data.classification import ClassificationDataset
from src.data.preprocessor import IndonesianPreprocessor
from src.modeling.calibrator import AutoImbalanceCalibrator
from src.modeling.adversarial import FGM
from src.augmentation.filtering import is_hallucinatory_or_corrupt
from src.utils.path_utils import resolve_path, ensure_dir


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
        prompt_prefix: Optional[str] = None,
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
        self.prompt_prefix = prompt_prefix

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
        if self.prompt_prefix:
            extra_desc += f", Context-Conditioning: '{self.prompt_prefix.strip()}'"
        print(f"  Enhancements Active -> Preprocessing: {self.use_preprocessing}, Class-Weights: {self.use_class_weights}, Label-Smoothing: {self.label_smoothing}, FGM-Adversarial: {self.use_fgm}{extra_desc}")

    def _set_seed(self, seed: Optional[int] = None):
        target_seed = self.seed if seed is None else seed
        random.seed(target_seed)
        np.random.seed(target_seed)
        torch.manual_seed(target_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(target_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train_epoch(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: Any,
        scheduler: Any,
        criterion: Optional[torch.nn.Module] = None,
        fgm: Optional[FGM] = None,
        log_priors: Optional[torch.Tensor] = None,
        gradient_accumulation_steps: int = 1,
    ) -> float:
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            try:
                # Avoid passing labels directly to model forward to prevent redundant internal HF loss computation graph
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                if log_priors is not None:
                    logits = logits + self.logit_adj_tau * log_priors

                if criterion is not None:
                    loss = criterion(logits, labels)
                else:
                    loss = F.cross_entropy(logits, labels)

                if loss is not None and loss.dim() > 0:
                    loss = loss.mean()

                loss_scaled = loss / max(1, gradient_accumulation_steps)
                loss_scaled.backward()

                if fgm is not None:
                    fgm.attack()
                    outputs_adv = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits_adv = outputs_adv.logits
                    if log_priors is not None:
                        logits_adv = logits_adv + self.logit_adj_tau * log_priors

                    if criterion is not None:
                        loss_adv = criterion(logits_adv, labels)
                    else:
                        loss_adv = F.cross_entropy(logits_adv, labels)

                    if loss_adv is not None and loss_adv.dim() > 0:
                        loss_adv = loss_adv.mean()

                    loss_adv_scaled = loss_adv / max(1, gradient_accumulation_steps)
                    loss_adv_scaled.backward()
                    fgm.restore()
                    del outputs_adv, logits_adv, loss_adv, loss_adv_scaled

                total_loss += loss.item()

                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                del outputs, logits, loss, loss_scaled
            except (torch.OutOfMemoryError, RuntimeError) as err:
                err_str = str(err).lower()
                is_oom = isinstance(err, torch.OutOfMemoryError) or ("out of memory" in err_str) or ("cublas" in err_str)
                if is_oom:
                    print(f"  [CUDA OOM GUARD] Memory allocation error caught during step {step}/{len(dataloader)}: {err}")
                    optimizer.zero_grad()
                    if fgm is not None:
                        try:
                            fgm.restore()
                        except Exception:
                            pass
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    raise torch.OutOfMemoryError(f"CUDA memory allocation failed at step {step}: {err}") from err
                else:
                    raise err
            finally:
                del input_ids, attention_mask, labels

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
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                if log_priors is not None and self.eval_logit_adj_tau > 0.0:
                    logits = logits - self.eval_logit_adj_tau * log_priors
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())
                del outputs, logits, preds, input_ids, attention_mask, labels

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        """Trains and evaluates IndoBERT on a single dataset pair."""
        val_info = f", Val: {len(val_df)}" if val_df is not None else ""
        num_epochs = epochs if epochs is not None else self.epochs
        use_early_stopping = early_stopping if early_stopping is not None else self.early_stopping
        pat = patience if patience is not None else self.patience

        print(f"\nEvaluating: {name} (Train: {len(train_df)}{val_info}, Test: {len(test_df)}, Epochs: {num_epochs}, EarlyStopping: {use_early_stopping} [patience={pat}])")

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

        test_label_counts = test_df["label"].value_counts().to_dict()
        for lbl in label_list:
            if lbl not in test_label_counts:
                print(f"  [CRITICAL WARNING] Label '{lbl}' is absent in test_df! Macro metrics will be distorted.")

        label_map = {lbl: i for i, lbl in enumerate(label_list)}
        id2label = {i: lbl for i, lbl in enumerate(label_list)}
        num_labels = len(label_list)

        self._set_seed(seed)

        if self.target_oversample:
            dfs_to_concat = [train_df]
            for target_class, multiplier in self.target_oversample.items():
                minority_rows = train_df[train_df["label"] == target_class]
                if not minority_rows.empty and multiplier > 1:
                    replicated = pd.concat([minority_rows] * (multiplier - 1), ignore_index=True)
                    dfs_to_concat.append(replicated)
                    print(f"  [TARGET OVERSAMPLER] Oversampled class '{target_class}' by {multiplier}x: {len(minority_rows)} -> {len(minority_rows) * multiplier} samples")
            train_df = pd.concat(dfs_to_concat, ignore_index=True).sample(frac=1.0, random_state=seed or self.seed).reset_index(drop=True)

        if self.use_logit_adjustment:
            class_counts = np.array([sum(train_df["label"] == lbl) for lbl in label_list], dtype=np.float32)
            priors = class_counts / (class_counts.sum() + 1e-6)
            log_priors = torch.tensor(np.log(priors + 1e-8), dtype=torch.float32, device=self.device)
            priors_dict = {lbl: f"{priors[i]:.3f}" for i, lbl in enumerate(label_list)}
            print(f"  [LOGIT ADJUSTMENT] Active (tau={self.logit_adj_tau}, eval_tau={self.eval_logit_adj_tau}) | Priors: {priors_dict}")
        else:
            log_priors = None

        prep = self.preprocessor if self.use_preprocessing else None
        train_ds = ClassificationDataset(train_df["text"].tolist(), train_df["label"].tolist(), self.model_name, self.max_len, label_map, preprocessor=prep, prompt_prefix=self.prompt_prefix)
        test_ds = ClassificationDataset(test_df["text"].tolist(), test_df["label"].tolist(), self.model_name, self.max_len, label_map, preprocessor=prep, prompt_prefix=self.prompt_prefix)

        # Multi-GPU / Single-GPU micro-batching configuration:
        # Use micro-batching with gradient accumulation to prevent DataParallel
        # replica 0 memory aggregation bottlenecks and VRAM OOM on 12GB GPUs.
        effective_batch_size = self.batch_size
        micro_batch_size = effective_batch_size
        grad_accum_steps = 1
        free_mem_gib = 8.0
        if self.device.type == "cuda":
            try:
                free_mem, _ = torch.cuda.mem_get_info(self.device.index or 0)
                free_mem_gib = free_mem / (1024 ** 3)
            except Exception:
                pass

            if free_mem_gib >= 8.0:
                target_micro_batch = 16
            elif free_mem_gib >= 4.0:
                target_micro_batch = 8
            elif free_mem_gib >= 2.0:
                target_micro_batch = 4
            else:
                target_micro_batch = 2

            if effective_batch_size > target_micro_batch:
                micro_batch_size = target_micro_batch
                grad_accum_steps = max(1, math.ceil(effective_batch_size / micro_batch_size))
                print(f"  [Memory Optimization] Available VRAM: {free_mem_gib:.2f} GiB | Effective batch size {effective_batch_size} -> Micro-batch size: {micro_batch_size}, Gradient accumulation steps: {grad_accum_steps}")

        train_loader = DataLoader(train_ds, batch_size=micro_batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=micro_batch_size, shuffle=False)

        if val_df is not None:
            val_ds = ClassificationDataset(val_df["text"].tolist(), val_df["label"].tolist(), self.model_name, self.max_len, label_map, preprocessor=prep, prompt_prefix=self.prompt_prefix)
            val_loader = DataLoader(val_ds, batch_size=micro_batch_size, shuffle=False)
        else:
            val_loader = None

        # Pre-allocation VRAM purge
        from src.utils.hardware import release_model_vram, deep_cuda_purge
        deep_cuda_purge()

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

        if self.device.type == "cuda" and self.num_gpus > 1:
            if free_mem_gib < 3.0 or micro_batch_size < self.num_gpus:
                print(f"  [GPU Selection] Low VRAM ({free_mem_gib:.2f} GiB) or micro-batch ({micro_batch_size}) < GPU count ({self.num_gpus}). Training on primary GPU {self.device} to prevent DataParallel replica 0 bottleneck.")
            else:
                print(f"  [Multi-GPU] Wrapping model with torch.nn.DataParallel across {self.num_gpus} GPUs")
                model = torch.nn.DataParallel(model)

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

        total_steps = math.ceil(len(train_loader) / grad_accum_steps) * num_epochs
        optimizer = AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

        best_val_f1 = -1.0
        best_train_loss = float("inf")
        best_model_weights = None
        best_epoch = 0
        patience_counter = 0
        actual_epochs_trained = 0
        history = {
            "epoch": [],
            "train_loss": [],
            "val_accuracy": [],
            "val_macro_f1": [],
        }

        epoch = 0
        retry_count = 0
        max_retries = 3

        while epoch < num_epochs:
            actual_epochs_trained = epoch + 1
            try:
                loss = self.train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    criterion=criterion,
                    fgm=fgm,
                    log_priors=log_priors,
                    gradient_accumulation_steps=grad_accum_steps,
                )
            except torch.OutOfMemoryError as oom_err:
                retry_count += 1
                if retry_count > max_retries or micro_batch_size <= 1:
                    print(f"  [FATAL OOM] Cannot recover from OutOfMemoryError after {retry_count} retries (micro-batch={micro_batch_size}).")
                    raise oom_err

                new_micro_batch = max(1, micro_batch_size // 2)
                grad_accum_steps = max(1, math.ceil(effective_batch_size / new_micro_batch))
                print(f"  [OOM RECOVERY] Halving micro-batch size {micro_batch_size} -> {new_micro_batch} (grad_accum={grad_accum_steps}) and restarting training from epoch 1...")
                micro_batch_size = new_micro_batch

                # If wrapped in DataParallel and micro-batch size is smaller than num_gpus, unwrap to single GPU
                if hasattr(model, "module") and micro_batch_size < self.num_gpus:
                    print(f"  [OOM RECOVERY] Micro-batch size ({micro_batch_size}) < GPU count ({self.num_gpus}); unwrapping DataParallel to single-GPU execution.")
                    model = model.module

                # Recreate dataloaders with reduced micro_batch_size
                train_loader = DataLoader(train_ds, batch_size=micro_batch_size, shuffle=True)
                if val_loader is not None:
                    val_loader = DataLoader(val_ds, batch_size=micro_batch_size, shuffle=False)
                test_loader = DataLoader(test_ds, batch_size=micro_batch_size, shuffle=False)

                # Rebuild optimizer and scheduler
                optimizer.zero_grad()
                total_steps = math.ceil(len(train_loader) / grad_accum_steps) * num_epochs
                optimizer = AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=1e-8)
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

                # Reset training history to restart cleanly
                epoch = 0
                history["epoch"].clear()
                history["train_loss"].clear()
                history["val_accuracy"].clear()
                history["val_macro_f1"].clear()
                best_val_f1 = -1.0
                best_train_loss = float("inf")
                best_model_weights = None
                patience_counter = 0

                deep_cuda_purge()
                continue

            history["epoch"].append(epoch + 1)
            history["train_loss"].append(float(loss))
            if val_loader is not None:
                v_acc, v_prec, v_rec, v_f1, _, _, _ = self.evaluate(model, val_loader, num_labels=num_labels, log_priors=log_priors)
                history["val_accuracy"].append(float(v_acc))
                history["val_macro_f1"].append(float(v_f1))
                print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f} | Val Acc: {v_acc:.4f} | Val Macro-F1: {v_f1:.4f}")
                if v_f1 > best_val_f1 + self.min_delta:
                    best_val_f1 = v_f1
                    best_epoch = epoch + 1
                    target_model = model.module if hasattr(model, "module") else model
                    # Store optimal checkpoint on CPU to prevent GPU VRAM exhaustion
                    best_model_weights = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if use_early_stopping and patience_counter >= pat:
                        print(f"  [Early Stopping Triggered] No improvement in Val Macro-F1 for {pat} consecutive epochs. Halting at epoch {epoch+1}/{num_epochs}.")
                        break
            else:
                print(f"  Epoch {epoch+1}/{num_epochs} - Train Loss: {loss:.4f}")
                if best_train_loss - loss > self.min_delta:
                    best_train_loss = loss
                    best_epoch = epoch + 1
                    target_model = model.module if hasattr(model, "module") else model
                    best_model_weights = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if use_early_stopping and patience_counter >= pat:
                        print(f"  [Early Stopping Triggered] Training loss did not improve for {pat} consecutive epochs. Halting at epoch {epoch+1}/{num_epochs}.")
                        break

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            epoch += 1

        if best_model_weights is not None:
            target_model = model.module if hasattr(model, "module") else model
            target_model.load_state_dict(best_model_weights)
            del best_model_weights
            print(f"  [Optimal Checkpoint Restored] Best Val Macro-F1: {best_val_f1:.4f} (from epoch {best_epoch})")

        acc, prec, rec, f1, per_class_f1, all_labels, all_preds = self.evaluate(model, test_loader, num_labels=num_labels, log_priors=log_priors)
        test_probs, _ = self.predict_proba(model, test_loader, log_priors=log_priors)

        # Explicit cleanup of model, optimizer, dataloaders from GPU VRAM
        release_model_vram(model)
        del model, optimizer, scheduler, criterion, fgm
        del train_loader, test_loader
        if val_loader is not None:
            del val_loader
        deep_cuda_purge()

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
            "history": history,
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
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                if log_priors is not None and self.eval_logit_adj_tau > 0.0:
                    logits = logits - self.eval_logit_adj_tau * log_priors
                probs = torch.softmax(logits, dim=1)

                all_probs.append(probs.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())
                del outputs, logits, probs, input_ids, attention_mask, labels

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        """Trains and soft-votes across multiple heterogeneous models."""
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
                prompt_prefix=self.prompt_prefix,
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
            del sub_evaluator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
        evaluate_raw_erm: bool = True,
    ) -> pd.DataFrame:
        """Runs comprehensive benchmark comparing all baseline and augmented variants."""
        resolved_output = resolve_path(output_results_path)

        base_text_set = set(str(t).strip() for t in base_train_df["text"].dropna())
        cleaned_aug_dfs = {}
        for metric, aug_df in augmented_dfs.items():
            if "text" in aug_df.columns:
                is_base = aug_df["text"].astype(str).str.strip().isin(base_text_set)
                is_corrupt = aug_df["text"].apply(is_hallucinatory_or_corrupt)
                dropped = is_corrupt & (~is_base)
                if dropped.sum() > 0:
                    print(f"  [QUALITY FILTER] Sanitized {dropped.sum()} hallucinated candidate(s) from {metric} augmented training set.")
                clean_aug = aug_df[is_base | (~is_corrupt)].copy().reset_index(drop=True)
                cleaned_aug_dfs[metric] = clean_aug
            else:
                cleaned_aug_dfs[metric] = aug_df

        results = []

        raw_erm_res = None
        if evaluate_raw_erm and (self.use_logit_adjustment or self.target_oversample or self.use_fgm):
            print(f"\n>>> [DISENTANGLED BASELINE 1/2] Evaluating Standard Raw ERM Baseline (No Calibrator, No Augmentation)...")
            raw_evaluator = IndoBERTClassifierEvaluator(
                model_name=self.model_name,
                max_len=self.max_len,
                batch_size=self.batch_size,
                epochs=epochs if epochs is not None else self.epochs,
                lr=self.lr,
                weight_decay=self.weight_decay,
                device=str(self.device),
                seed=seed,
                early_stopping=early_stopping if early_stopping is not None else self.early_stopping,
                patience=patience if patience is not None else self.patience,
                min_delta=self.min_delta,
                use_preprocessing=self.use_preprocessing,
                use_class_weights=False,
                label_smoothing=0.0,
                use_fgm=False,
                use_logit_adjustment=False,
                logit_adj_tau=0.0,
                eval_logit_adj_tau=0.0,
                target_oversample=None,
                prompt_prefix=self.prompt_prefix,
            )
            raw_erm_res = raw_evaluator.evaluate_dataset(
                base_train_df,
                base_test_df,
                label_list,
                val_df=val_df,
                name=f"{dataset_name}-Baseline-RawERM",
                seed=seed,
                epochs=epochs,
                early_stopping=early_stopping,
                patience=patience,
            )
            results.append(raw_erm_res)
            del raw_evaluator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"\n>>> [DISENTANGLED BASELINE 2/2] Evaluating Calibrated Baseline (With AIC: Logit Adj + Oversampler + FGM)...")
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
        if raw_erm_res is not None:
            aic_gain = base_res["macro_f1"] - raw_erm_res["macro_f1"]
            aic_pct = (aic_gain / raw_erm_res["macro_f1"]) * 100.0 if raw_erm_res["macro_f1"] > 0 else 0.0
            base_res["f1_improvement"] = aic_gain
            base_res["f1_improvement_pct"] = aic_pct
        results.append(base_res)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ref_base_res = base_res

        for metric, aug_train in cleaned_aug_dfs.items():
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
            f1_gain = res["macro_f1"] - ref_base_res["macro_f1"]
            pct_gain = (f1_gain / ref_base_res["macro_f1"]) * 100.0 if ref_base_res["macro_f1"] > 0 else 0.0
            res["f1_improvement"] = f1_gain
            res["f1_improvement_pct"] = pct_gain
            results.append(res)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self.ensemble_configs:
            aug_results = [r for r in results if "Augmented" in r["dataset_name"] and not r["dataset_name"].endswith("Ensemble")]
            if aug_results and len(cleaned_aug_dfs) > 0:
                best_aug_res = max(aug_results, key=lambda x: x["macro_f1"])
                best_variant_key = best_aug_res["dataset_name"].split("-")[-1]
                target_train_df = cleaned_aug_dfs.get(best_variant_key, list(cleaned_aug_dfs.values())[0])
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
            f1_gain = ens_res["macro_f1"] - ref_base_res["macro_f1"]
            pct_gain = (f1_gain / ref_base_res["macro_f1"]) * 100.0 if ref_base_res["macro_f1"] > 0 else 0.0
            ens_res["f1_improvement"] = f1_gain
            ens_res["f1_improvement_pct"] = pct_gain
            results.append(ens_res)

        predictions = {}
        training_histories = {}
        for r in results:
            dname = r["dataset_name"]
            y_t = r.pop("y_true", None)
            y_p = r.pop("y_pred", None)
            r.pop("_test_probs", None)
            hist = r.pop("history", None)
            if hist is not None:
                training_histories[dname] = hist
            if y_t is not None and y_p is not None:
                predictions[f"{dname}__true"] = np.array(y_t)
                predictions[f"{dname}__pred"] = np.array(y_p)

        results_df = pd.DataFrame(results)
        ensure_dir(os.path.dirname(resolved_output))
        results_df.to_excel(resolved_output, index=False)
        print(f"\nFinal Classification Results saved to: {resolved_output}")

        if training_histories:
            hist_path = resolved_output.rsplit(".", 1)[0].replace("_evaluation_results", "") + "_training_history.json"
            with open(hist_path, "w", encoding="utf-8") as f:
                json.dump(training_histories, f, indent=2)
            print(f"Per-epoch training dynamics history saved to: {hist_path}")

        if predictions:
            pred_path = resolved_output.rsplit(".", 1)[0] + "_predictions.npz"
            np.savez(pred_path, **predictions)
            print(f"Predictions for confusion matrix saved to: {pred_path}")

        summary_cols = ["dataset_name", "model_name", "train_size", "epochs_trained", "best_epoch", "accuracy", "macro_f1", "f1_improvement_pct"]
        available_summary_cols = [c for c in summary_cols if c in results_df.columns]
        print(results_df[available_summary_cols])
        return results_df

