from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer


class DeepMetaOptimizer(Optimizer):
    """
    Optimizer that projects fast-path gradients to remain orthogonal
    to cached base gradients while learning a deep momentum memory.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        meta_lr: float = 1e-3,
        hidden_dim: int = 256,
        device: Optional[torch.device] = None,
        eps: float = 1e-8,
    ) -> None:
        params = list(params)
        if len(params) == 0:
            raise ValueError("DeepMetaOptimizer received an empty parameter list.")

        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.meta_lr = meta_lr
        self.eps = eps

        sample_param = params[0]
        self.device = device or sample_param.device
        self._slices: List[slice] = []
        self._param_order: List[torch.nn.Parameter] = []
        start = 0
        for group in self.param_groups:
            for p in group["params"]:
                numel = p.numel()
                self._slices.append(slice(start, start + numel))
                self._param_order.append(p)
                start += numel
        self.total_dim = start

        if self.total_dim == 0:
            raise ValueError("DeepMetaOptimizer parameters must have non-zero elements.")

        self.deep_momentum = nn.Sequential(
            nn.Linear(self.total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.total_dim),
        ).to(self.device)

        self.memory_state = torch.zeros(self.total_dim, device=self.device)
        self.reference_gradient: Optional[torch.Tensor] = None

    def set_reference(self, reference: Optional[torch.Tensor]) -> None:
        if reference is None:
            self.reference_gradient = None
            return
        if reference.numel() != self.total_dim:
            raise ValueError(
                f"Reference gradient has dim {reference.numel()}, expected {self.total_dim}"
            )
        self.reference_gradient = reference.to(self.device)

    def get_memory(self) -> torch.Tensor:
        return self.memory_state.detach().clone()

    def load_memory(self, memory: torch.Tensor) -> None:
        if memory.numel() != self.total_dim:
            raise ValueError("Loaded memory must match optimizer parameter dimensionality.")
        self.memory_state = memory.to(self.device)

    def _flatten_grads(self) -> Optional[torch.Tensor]:
        grads = torch.zeros(self.total_dim, device=self.device)
        has_grad = False
        for slc, param in zip(self._slices, self._param_order):
            if param.grad is None:
                continue
            grads[slc] = param.grad.detach().reshape(-1)
            has_grad = True
        if not has_grad:
            return None
        return grads

    def _apply_updates(self, grad_vec: torch.Tensor) -> None:
        offset = 0
        for group in self.param_groups:
            lr = group["lr"]
            for param in group["params"]:
                numel = param.numel()
                grad_slice = grad_vec[offset : offset + numel].view_as(param)
                param.data.add_(grad_slice, alpha=-lr)
                offset += numel

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        grad_vec = self._flatten_grads()
        if grad_vec is None:
            return None

        grad_vec = grad_vec.to(self.device)
        memory_pred = self.deep_momentum(grad_vec.detach())
        self.memory_state = (1 - self.meta_lr) * self.memory_state + self.meta_lr * memory_pred.detach()

        reference = self.reference_gradient if self.reference_gradient is not None else self.memory_state
        ref_norm_sq = torch.dot(reference, reference).clamp(min=self.eps)
        dot = torch.dot(grad_vec, reference)
        # Orthogonal projection: remove component along reference direction
        # This prevents updates that align with base gradients (both positive and negative alignment)
        if abs(dot) > self.eps:
            grad_vec = grad_vec - (dot / ref_norm_sq) * reference

        self._apply_updates(grad_vec)
        return grad_vec

