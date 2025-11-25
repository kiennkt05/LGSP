import math
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn


class NestedContinuumAdapter(nn.Module):
    """
    Adapter with decoupled slow/fast paths governed by a learnable gate.
    Slow path captures long-term knowledge, fast path captures rapid adaptation.
    """

    def __init__(
        self,
        embed_dim: int,
        bottleneck_dim: int,
        gate_init: float = 0.0,
        activation_fast: nn.Module = None,
    ) -> None:
        super().__init__()
        hidden_dim = bottleneck_dim

        self.slow_path = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.fast_path = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            activation_fast if activation_fast is not None else nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.gate = nn.Parameter(torch.full((1,), gate_init))
        self.reset_fast_path(zero_init=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        slow = self.slow_path(x)
        fast = self.fast_path(x)
        return x + slow + torch.sigmoid(self.gate) * fast

    # -------------------------
    # Utility methods
    # -------------------------
    def _iter_fast_layers(self) -> Iterable[nn.Module]:
        for layer in self.fast_path:
            if hasattr(layer, "reset_parameters"):
                yield layer

    def reset_fast_path(self, zero_init: bool = False) -> None:
        """Re-initialize fast path weights. Use near-zero init when zero_init."""
        for layer in self._iter_fast_layers():
            if isinstance(layer, nn.Linear):
                if zero_init:
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                    layer.weight.data.mul_(1e-2)
                else:
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def freeze_slow(self) -> None:
        for p in self.slow_path.parameters():
            p.requires_grad = False
        self.slow_path.eval()

    def train_fast_only(self) -> None:
        self.freeze_slow()
        for p in self.fast_path.parameters():
            p.requires_grad = True
        self.fast_path.train()
        self.gate.requires_grad = True

    def unfreeze_all(self) -> None:
        for module in (self.slow_path, self.fast_path):
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        self.gate.requires_grad = True

    def fast_parameters(self) -> List[nn.Parameter]:
        return list(self.fast_path.parameters()) + [self.gate]

    def slow_parameters(self) -> List[nn.Parameter]:
        return list(self.slow_path.parameters())

    def split_parameters(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        return self.slow_parameters(), self.fast_parameters()

