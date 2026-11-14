"""
Minimum Bayes Risk (MBR) Calibrator and Decision Engine.
Implements Bayesian risk-optimal decision decoding for imbalanced text classification.

Given model posterior probabilities P(Y=j | x) and a class-dependent cost matrix
C_{ij} = L(predict=i, true=j), the Minimum Bayes Risk decision is:
    \\hat{y}_{MBR}(x) = \\arg\\min_{i \\in {0, ..., K-1}} \\sum_{j=0}^{K-1} C_{ij} P(Y=j | x)

Under symmetric 0-1 loss (C_{ii}=0, C_{ij}=1 for i!=j), MBR decision reduces identically
to standard Maximum A Posteriori (MAP) classification: \\hat{y} = \\arg\\max_i P(Y=i | x).
Under severe class imbalance (e.g. HoASA, EmoT, SmSA), asymmetric false-negative costs
C_{ik} > 1.0 for minority classes k dynamically optimize Macro-F1 while strictly
guarding majority-class accuracy via validation-constrained calibration.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any


class MinimumBayesRiskCalibrator:
    """
    Automated Minimum Bayes Risk (MBR) decision calibrator.
    Supports prior-informed cost matrix derivation, validation-constrained
    cost optimization, and risk-optimal inference decoding.
    """

    def __init__(
        self,
        num_classes: int,
        label_list: Optional[List[str]] = None,
        minority_classes: Optional[List[Union[str, int]]] = None,
        default_cost_matrix: Optional[np.ndarray] = None,
        accuracy_tolerance: float = 0.003,
        alpha: float = 0.25,
        max_minority_cost: float = 5.0,
        verbose: bool = True,
    ):
        self.num_classes = num_classes
        self.label_list = label_list or [str(i) for i in range(num_classes)]
        self.accuracy_tolerance = accuracy_tolerance
        self.alpha = alpha
        self.max_minority_cost = max_minority_cost
        self.verbose = verbose

        # Resolve minority class indices
        self.minority_indices: List[int] = []
        if minority_classes is not None:
            for mc in minority_classes:
                if isinstance(mc, int) and 0 <= mc < num_classes:
                    self.minority_indices.append(mc)
                elif isinstance(mc, str) and mc in self.label_list:
                    self.minority_indices.append(self.label_list.index(mc))

        # Initialize cost matrix (default: symmetric 0-1 loss)
        if default_cost_matrix is not None:
            self.cost_matrix = np.array(default_cost_matrix, dtype=np.float32)
        else:
            self.cost_matrix = self.build_01_cost_matrix(num_classes)

        self.calibrated = False
        self.best_macro_f1 = 0.0
        self.best_accuracy = 0.0

    @staticmethod
    def build_01_cost_matrix(num_classes: int) -> np.ndarray:
        """Returns standard symmetric 0-1 loss matrix (C_ii=0, C_ij=1 for i!=j)."""
        C = np.ones((num_classes, num_classes), dtype=np.float32)
        np.fill_diagonal(C, 0.0)
        return C

    @staticmethod
    def build_cost_matrix_from_penalties(num_classes: int, penalties: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Builds cost matrix from column penalties where C_{ij} = penalties[j] for i != j, and C_{ii} = 0.
        penalties[j] represents the penalty of failing to predict true class j (False Negative cost).
        """
        C = np.ones((num_classes, num_classes), dtype=np.float32)
        for j in range(num_classes):
            C[:, j] = float(penalties[j])
        np.fill_diagonal(C, 0.0)
        return C

    @staticmethod
    def build_valence_cost_matrix(
        label_list: List[str],
        valence_map: Dict[str, List[str]],
        intra_valence_cost: float = 1.0,
        cross_valence_cost: float = 2.0,
        neutral_polar_cost: Optional[float] = None,
    ) -> np.ndarray:
        """Constructs a K x K valence-weighted cost matrix from affective grouping."""
        from src.modeling.valence_loss import build_valence_cost_matrix
        return build_valence_cost_matrix(
            label_list=label_list,
            valence_map=valence_map,
            intra_valence_cost=intra_valence_cost,
            cross_valence_cost=cross_valence_cost,
            neutral_polar_cost=neutral_polar_cost,
        )

    def derive_prior_costs(
        self,
        class_counts: Union[Dict[str, int], List[int], np.ndarray],
    ) -> np.ndarray:
        """
        Derives initial prior-informed false-negative penalty vector directly from class frequencies:
            c_k = (max_j N_j / N_k) ** alpha
        bounded by [1.0, max_minority_cost].
        """
        if isinstance(class_counts, dict):
            counts = np.array([class_counts.get(lbl, 1) for lbl in self.label_list], dtype=np.float32)
        else:
            counts = np.array(class_counts, dtype=np.float32)

        max_c = max(1.0, float(np.max(counts)))
        penalties = np.ones(self.num_classes, dtype=np.float32)

        for k in range(self.num_classes):
            c_k = max(1.0, float(counts[k]))
            ir_k = max_c / c_k
            penalty = float(ir_k ** self.alpha)
            # Only scale if designated minority or significant imbalance
            if k in self.minority_indices or ir_k > 2.0:
                penalties[k] = float(np.clip(penalty, 1.0, self.max_minority_cost))
            else:
                penalties[k] = 1.0

        return penalties

    def compute_risk(self, probs: np.ndarray, cost_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes the expected Bayes risk matrix R = probs @ cost_matrix.T.
        R[n, i] is the expected loss of predicting class i for sample n.
        """
        C = cost_matrix if cost_matrix is not None else self.cost_matrix
        probs_2d = np.atleast_2d(probs)
        return probs_2d @ C.T

    def predict_mbr(self, probs: np.ndarray, cost_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Decodes risk-optimal predictions: y_hat = argmin_i R_i(x).
        """
        risk = self.compute_risk(probs, cost_matrix)
        preds = np.argmin(risk, axis=1)
        if probs.ndim == 1:
            return preds[0]
        return preds

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Tuple[float, float, List[float]]:
        """Fast vectorized computation of Accuracy and Macro-F1."""
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        acc = float(np.sum(np.diag(cm)) / max(1, len(y_true)))
        f1s = []
        for c in range(num_classes):
            tp = cm[c, c]
            fp = np.sum(cm[:, c]) - tp
            fn = np.sum(cm[c, :]) - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(float(f1))
        macro_f1 = float(np.mean(f1s))
        return acc, macro_f1, f1s

    def calibrate(
        self,
        val_probs: np.ndarray,
        val_targets: np.ndarray,
        train_counts: Optional[Union[Dict[str, int], List[int], np.ndarray]] = None,
        search_grid: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Calibrates the optimal cost matrix on validation probabilities to maximize Macro-F1
        subject to an overall accuracy constraint (Accuracy >= Base_Accuracy - accuracy_tolerance).
        Guarantees monotonic non-regression: defaults to 0-1 loss if no improvement is found.
        """
        # Baseline MAP metrics (argmax over probabilities)
        base_preds = np.argmax(val_probs, axis=1)
        base_acc, base_f1, base_f1s = self._compute_metrics(val_targets, base_preds, self.num_classes)
        min_allowed_acc = base_acc - self.accuracy_tolerance

        best_cost_matrix = self.build_01_cost_matrix(self.num_classes)
        best_f1 = base_f1
        best_acc = base_acc
        best_penalties = np.ones(self.num_classes, dtype=np.float32)

        # Prior penalties as candidate baseline
        prior_penalties = np.ones(self.num_classes, dtype=np.float32)
        if train_counts is not None:
            prior_penalties = self.derive_prior_costs(train_counts)

        # Candidate penalty search grid for minority classes
        candidate_values = search_grid or [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

        # Target classes to tune: minority indices if set, else all non-dominant classes
        target_indices = self.minority_indices if self.minority_indices else list(range(self.num_classes))

        # 1. Test prior-informed cost matrix
        C_prior = self.build_cost_matrix_from_penalties(self.num_classes, prior_penalties)
        preds_prior = self.predict_mbr(val_probs, C_prior)
        acc_prior, f1_prior, _ = self._compute_metrics(val_targets, preds_prior, self.num_classes)

        if f1_prior > best_f1 + 1e-4 and acc_prior >= min_allowed_acc:
            best_f1 = f1_prior
            best_acc = acc_prior
            best_cost_matrix = C_prior
            best_penalties = prior_penalties.copy()

        # 2. Coordinate search over target minority penalty parameters
        for _ in range(2):  # 2 coordinate passes
            for m_idx in target_indices:
                current_best_cand = best_penalties[m_idx]
                for cand in candidate_values:
                    test_penalties = best_penalties.copy()
                    test_penalties[m_idx] = cand
                    C_test = self.build_cost_matrix_from_penalties(self.num_classes, test_penalties)
                    test_preds = self.predict_mbr(val_probs, C_test)
                    acc_t, f1_t, _ = self._compute_metrics(val_targets, test_preds, self.num_classes)

                    if f1_t > best_f1 + 1e-4 and acc_t >= min_allowed_acc:
                        best_f1 = f1_t
                        best_acc = acc_t
                        best_cost_matrix = C_test
                        current_best_cand = cand
                best_penalties[m_idx] = current_best_cand

        self.cost_matrix = best_cost_matrix
        self.calibrated = True
        self.best_macro_f1 = best_f1
        self.best_accuracy = best_acc

        if self.verbose:
            gain = best_f1 - base_f1
            if gain > 1e-4:
                print(f"  [MINIMUM BAYES RISK] Calibrated optimal cost matrix:")
                print(f"    - Base Val Acc: {base_acc:.4f} | Base Val Macro-F1: {base_f1:.4f}")
                print(f"    - MBR  Val Acc: {best_acc:.4f} | MBR  Val Macro-F1: {best_f1:.4f} (Delta: {gain:+.4f})")
                print(f"    - Penalties Vector: {np.round(best_penalties, 2).tolist()}")
            else:
                print(f"  [MINIMUM BAYES RISK] Baseline MAP (0-1 loss) is already Bayes-optimal on validation set.")

        return self.cost_matrix
