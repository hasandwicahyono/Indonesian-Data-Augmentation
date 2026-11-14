"""
Valence-Weighted Cost Matrix Loss Function.
Integrates Russell's Circumplex Affective Geometry and Cost-Sensitive Decision Theory
into PyTorch neural representation learning.

Mathematical Formulation:
Given class labels {0, ..., K-1} partitioned into affective valence groups (e.g., positive,
negative, neutral), the Valence Cost Matrix C in R^{K x K} quantifies the asymmetric penalty
of misclassifying true class i as candidate class j:
  - C_{ii} = 0 (correct prediction incurs zero penalty)
  - C_{ij} = c_{intra} for i != j within the same valence manifold (e.g., Anger <-> Sadness)
  - C_{ij} = c_{cross} for opposite valence polarities (e.g., Joy <-> Anger, Love <-> Sadness)
  - C_{ij} = c_{neutral} between neutral and polar categories

The loss function optimizes:
  L(z, y) = L_{cost-margin}(z, y) + lambda_{risk} * R(p, y)
where:
  L_{cost-margin}(z, y) = log( e^{z_y} + sum_{j != y} C_{y,j} * e^{z_j} ) - z_y
  R(p, y) = sum_{j=1}^K C_{y,j} * p_j  (Expected Bayes Risk under posterior p = softmax(z))

Properties:
  1. Under uniform costs (C_{ij} = 1 for all i != j), L_{cost-margin} reduces identically
     to standard multi-class Cross-Entropy loss.
  2. For cross-valence competitors (C_{y,j} = c_{cross} > 1), an effective log-margin
     Delta_{y,j} = log(C_{y,j}) is enforced, compelling the model to separate opposite-valence
     representations and eliminating affective sign-flip errors.
  3. Valence-aware cost-weighted label smoothing distributes smoothing mass inversely
     proportional to misclassification cost, preventing probability leakage into opposite polarities.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


def build_valence_cost_matrix(
    label_list: List[str],
    valence_map: Dict[str, List[str]],
    intra_valence_cost: float = 1.0,
    cross_valence_cost: float = 2.0,
    neutral_polar_cost: Optional[float] = None,
) -> np.ndarray:
    """
    Constructs a K x K cost matrix C from affective valence group annotations.

    Args:
        label_list: List of class names of length K.
        valence_map: Dictionary mapping valence names ('positive', 'negative', 'neutral')
                     to lists of class names.
        intra_valence_cost: Misclassification penalty within the same valence group (default: 1.0).
        cross_valence_cost: Misclassification penalty between opposite polarities (default: 2.0).
        neutral_polar_cost: Penalty between neutral and polar classes. Defaults to average of
                            intra and cross costs (e.g. 1.5).

    Returns:
        np.ndarray: Float32 array of shape (K, K) where C[i, j] is the penalty of
                    predicting class j when true class is i.
    """
    K = len(label_list)
    label2idx = {lbl: idx for idx, lbl in enumerate(label_list)}

    if neutral_polar_cost is None:
        neutral_polar_cost = float(intra_valence_cost + cross_valence_cost) / 2.0

    # Map each class index to its valence category
    class_to_valence: Dict[int, str] = {}
    for v_name, members in valence_map.items():
        v_key = v_name.lower().strip()
        for m in members:
            if m in label2idx:
                class_to_valence[label2idx[m]] = v_key

    C = np.ones((K, K), dtype=np.float32) * float(cross_valence_cost)

    for i in range(K):
        v_i = class_to_valence.get(i, "unknown")
        for j in range(K):
            if i == j:
                C[i, j] = 0.0
                continue

            v_j = class_to_valence.get(j, "unknown")

            if v_i == "unknown" or v_j == "unknown":
                C[i, j] = float(intra_valence_cost)
            elif v_i == v_j:
                # Same valence manifold (e.g. anger <-> sadness, happy <-> love)
                C[i, j] = float(intra_valence_cost)
            elif (v_i == "neutral" and v_j in ("positive", "negative")) or (
                v_j == "neutral" and v_i in ("positive", "negative")
            ):
                # Neutral vs. Polar boundary
                C[i, j] = float(neutral_polar_cost)
            elif (v_i == "positive" and v_j == "negative") or (
                v_i == "negative" and v_j == "positive"
            ):
                # Direct affective contradiction / sign-flip
                C[i, j] = float(cross_valence_cost)
            else:
                C[i, j] = float(cross_valence_cost)

    return C


class ValenceWeightedCostLoss(nn.Module):
    """
    PyTorch Loss Module implementing Valence-Weighted Cost Matrix Optimization.

    Combines:
      1. Cost-Weighted Logit-Margin Softmax:
         L_{margin}(z, y) = log( e^{z_y} + sum_{j != y} C_{y,j} * e^{z_j} ) - z_y
      2. Posterior Expected Bayes Risk Regularization:
         R(p, y) = sum_{j=1}^K C_{y,j} * p_j
      3. Valence-Aware Cost-Weighted Label Smoothing (optional)
      4. Focal Loss Hard-Instance Modulation (optional)
      5. Class Frequency Prior Weighting (optional)
    """

    def __init__(
        self,
        label_list: Optional[List[str]] = None,
        valence_map: Optional[Dict[str, List[str]]] = None,
        cost_matrix: Optional[Union[np.ndarray, torch.Tensor]] = None,
        weight: Optional[torch.Tensor] = None,
        intra_valence_cost: float = 1.0,
        cross_valence_cost: float = 2.0,
        neutral_polar_cost: Optional[float] = None,
        risk_weight: float = 0.5,
        label_smoothing: float = 0.0,
        use_focal_loss: bool = False,
        focal_gamma: float = 1.5,
    ):
        super().__init__()
        self.intra_valence_cost = float(intra_valence_cost)
        self.cross_valence_cost = float(cross_valence_cost)
        self.neutral_polar_cost = neutral_polar_cost
        self.risk_weight = float(risk_weight)
        self.label_smoothing = float(label_smoothing)
        self.use_focal_loss = bool(use_focal_loss)
        self.focal_gamma = float(focal_gamma)

        # Build or register Cost Matrix
        if cost_matrix is not None:
            if isinstance(cost_matrix, np.ndarray):
                C_tensor = torch.tensor(cost_matrix, dtype=torch.float32)
            else:
                C_tensor = cost_matrix.clone().detach().to(torch.float32)
        elif label_list is not None and valence_map is not None:
            C_np = build_valence_cost_matrix(
                label_list=label_list,
                valence_map=valence_map,
                intra_valence_cost=self.intra_valence_cost,
                cross_valence_cost=self.cross_valence_cost,
                neutral_polar_cost=self.neutral_polar_cost,
            )
            C_tensor = torch.tensor(C_np, dtype=torch.float32)
        else:
            # Fallback to 0-1 cost matrix if no valence map is provided
            num_cls = len(label_list) if label_list else 2
            C_tensor = torch.ones((num_cls, num_cls), dtype=torch.float32)
            C_tensor.fill_diagonal_(0.0)

        # Store Cost Matrix buffer (C[y, j] is cost of predicting j when true is y)
        self.register_buffer("cost_matrix", C_tensor)

        # Class weights for class imbalance
        if weight is not None:
            self.register_buffer("class_weights", weight.clone().detach().to(torch.float32))
        else:
            self.class_weights = None

    @property
    def weight(self):
        return self.class_weights

    @weight.setter
    def weight(self, value):
        if value is not None:
            self.class_weights = value.clone().detach().to(torch.float32)
        else:
            self.class_weights = None

    def get_cost_matrix(self) -> torch.Tensor:
        """Returns the active cost matrix tensor."""
        return self.cost_matrix

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the valence-cost-weighted loss.

        Args:
            logits: Predicted raw logits of shape (B, K).
            targets: Ground-truth target indices of shape (B,).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        B, K = logits.shape
        device = logits.device

        # Ensure buffer device alignment
        if self.cost_matrix.device != device:
            self.cost_matrix = self.cost_matrix.to(device)
        if self.class_weights is not None and self.class_weights.device != device:
            self.class_weights = self.class_weights.to(device)

        # -------------------------------------------------------------
        # 1. Cost-Weighted Logit-Margin Softmax
        # -------------------------------------------------------------
        # Gather cost row for each sample in batch: (B, K)
        # batch_costs[b, j] = C_{targets[b], j}
        batch_costs = self.cost_matrix[targets]  # (B, K)

        # In the margin sum:
        # e^{z_y} + sum_{j != y} C_{y,j} * e^{z_j}
        # For j == y, effective multiplier is 1.0 (since C_{y,y} = 0)
        # For j != y, effective multiplier is max(1.0, C_{y,j})
        eff_mult = torch.where(
            targets.unsqueeze(1) == torch.arange(K, device=device).unsqueeze(0),
            torch.ones_like(batch_costs),
            torch.clamp(batch_costs, min=1e-4),
        )
        log_eff_mult = torch.log(eff_mult)  # (B, K)

        # Modified logits: z_b,j + log(C_{y,j})
        adjusted_logits = logits + log_eff_mult
        lse = torch.logsumexp(adjusted_logits, dim=-1)  # (B,)

        # Target true logits
        target_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        margin_loss = lse - target_logits  # (B,)

        # -------------------------------------------------------------
        # 2. Expected Bayes Risk Regularization
        # -------------------------------------------------------------
        probs = F.softmax(logits, dim=-1)  # (B, K)
        if self.risk_weight > 0.0:
            # R_b = sum_{j=1}^K C_{y,j} * p_{b,j}
            expected_risk = (batch_costs * probs).sum(dim=-1)  # (B,)
            per_sample_loss = margin_loss + self.risk_weight * expected_risk
        else:
            per_sample_loss = margin_loss

        # -------------------------------------------------------------
        # 3. Valence-Aware Cost-Weighted Label Smoothing (Optional)
        # -------------------------------------------------------------
        if self.label_smoothing > 0.0:
            eps = self.label_smoothing
            # Distribute smoothing mass inversely proportional to cost
            # Smooth mass S_{y, j} proportional to exp(-C_{y, j})
            smooth_weights = torch.exp(-batch_costs)
            # Zero out true class in smoothing weights
            smooth_weights.scatter_(1, targets.unsqueeze(1), 0.0)
            smooth_sum = smooth_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            smooth_dist = smooth_weights / smooth_sum  # (B, K)

            log_probs = F.log_softmax(logits, dim=-1)
            smooth_loss = -(smooth_dist * log_probs).sum(dim=-1)  # (B,)
            per_sample_loss = (1.0 - eps) * per_sample_loss + eps * smooth_loss

        # -------------------------------------------------------------
        # 4. Focal Loss Hard-Instance Modulation (Optional)
        # -------------------------------------------------------------
        if self.use_focal_loss:
            p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-6, max=1.0)
            focal_modulator = (1.0 - p_t) ** self.focal_gamma
            per_sample_loss = focal_modulator * per_sample_loss

        # -------------------------------------------------------------
        # 5. Class Frequency Weighting & Reduction
        # -------------------------------------------------------------
        if self.class_weights is not None:
            sample_w = self.class_weights[targets]  # (B,)
            total_loss = (sample_w * per_sample_loss).sum() / sample_w.sum().clamp(min=1e-6)
        else:
            total_loss = per_sample_loss.mean()

        return total_loss
