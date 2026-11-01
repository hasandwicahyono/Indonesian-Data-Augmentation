"""
Adversarial Training utilities for neural sequence classifiers.
Implements the Fast Gradient Method (FGM) for adversarial perturbations on embedding weights.
"""

from typing import Dict
import torch


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
