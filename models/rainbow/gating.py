"""Probabilistic gate for RainbowPrompt layer selection."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


class ProbabilisticGate(nn.Module):
    """Learned per-layer gating distribution with Gumbel-Softmax relaxation."""

    def __init__(
        self,
        num_layers: int,
        tau_start: float = 1.0,
        tau_end: float = 0.3,
        harden_epoch_ratio: float = 0.6,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.harden_epoch_ratio = harden_epoch_ratio

        self.logits = nn.Parameter(torch.zeros(num_layers, 2))

    def forward(
        self,
        layer_idx: int,
        epoch: int,
        max_epochs: int,
        training: bool = True,
        temperature: float | None = None,
    ) -> Dict[str, torch.Tensor]:
        if layer_idx >= self.num_layers or layer_idx < 0:
            raise IndexError(f"Layer index {layer_idx} out of bounds for {self.num_layers} layers")

        logits = self.logits[layer_idx]
        # Calculate progress regardless of temperature setting
        progress = min(max(epoch, 0) / max(max_epochs, 1), 1.0)
        
        if temperature is not None:
            tau = temperature
        else:
            tau = self.tau_start + (self.tau_end - self.tau_start) * progress

        use_hard = training and progress >= self.harden_epoch_ratio

        if training:
            # Clamp random values to avoid 0 and 1, which cause NaN in Gumbel noise computation
            u = torch.clamp(torch.rand_like(logits), min=1e-10, max=1-1e-10)
            gumbel_noise = -torch.log(-torch.log(u))
            y = F.softmax((logits + gumbel_noise) / tau, dim=-1)
            if use_hard:
                index = torch.argmax(y, dim=-1)
                y_hard = F.one_hot(index, num_classes=2).float()
                y = (y_hard - y).detach() + y
        else:
            probs_eval = F.softmax(logits, dim=-1)
            index = torch.argmax(probs_eval, dim=-1)
            y = F.one_hot(index, num_classes=2).float()

        probs_soft = F.softmax(logits, dim=-1)
        prob_use = probs_soft[1]
        gate_value = y[1]

        # Sparsity loss: penalize layer usage (negative log encourages sparsity)
        # Use negative log to penalize high prob_use values
        sparsity_loss = -torch.log(prob_use + 1e-8)

        return {
            "gate": gate_value,
            "prob_use": prob_use,
            "logits": logits,
            "sparsity_loss": sparsity_loss,
        }

