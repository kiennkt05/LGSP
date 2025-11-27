"""RainbowPrompt module for FSCIL."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .evolution import RainbowEvolution
from .gating import ProbabilisticGate
from .storage import RainbowPromptStorage


class RainbowPromptModule(nn.Module):
    """Manage RainbowPrompt evolution, gating, and storage across ViT layers."""

    def __init__(
        self,
        embed_dim: int,
        prompt_length: int,
        num_layers: int,
        num_heads: int,
        proj_dim: int,
        align_hidden_dim: int,
        gate_tau_start: float,
        gate_tau_end: float,
        gate_harden_at: float,
        save_dir: str,
        use_task_conditioning: bool = True,
        enable_task_level: bool = True,
        enable_feature_level: bool = True,
        enable_alignment: bool = True,
        use_adaptive_gating: bool = True,
        use_paper_evolution: bool = False,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads for prefix prompts")

        self.embed_dim = embed_dim
        self.prompt_length = prompt_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_task_conditioning = use_task_conditioning
        self.use_adaptive_gating = use_adaptive_gating
        self.use_paper_evolution = use_paper_evolution

        self.evolutions = nn.ModuleList(
            [
                RainbowEvolution(
                    embed_dim=embed_dim,
                    prompt_length=prompt_length,
                    proj_dim=proj_dim,
                    align_hidden_dim=align_hidden_dim,
                    num_heads=num_heads,
                    use_task_conditioning=use_task_conditioning,
                    enable_task_level=enable_task_level,
                    enable_feature_level=enable_feature_level,
                    enable_alignment=enable_alignment,
                    use_paper_evolution=use_paper_evolution,
                )
                for _ in range(num_layers)
            ]
        )

        self.base_prompts = nn.ModuleList([nn.ParameterList() for _ in range(num_layers)])
        self.storage = RainbowPromptStorage(save_dir)

        self.current_task_id: Optional[int] = None
        self.current_gate: Optional[ProbabilisticGate] = None
        self.current_epoch: int = 0
        self.max_epochs: int = 1
        self.task_embedding: Optional[torch.Tensor] = None
        self.training_mode: bool = True

        self._latest_layer_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._aux_losses: Dict[str, torch.Tensor] = {}
        self._inference_cache: Dict[int, torch.Tensor] = {}
        self._inference_task_ids: list[int] = []

        self._gate_config = dict(
            tau_start=gate_tau_start,
            tau_end=gate_tau_end,
            harden_epoch_ratio=gate_harden_at,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def set_epoch(self, epoch: int, max_epochs: int) -> None:
        self.current_epoch = epoch
        self.max_epochs = max_epochs

    def update_task_embedding(self, embedding: Optional[torch.Tensor]) -> None:
        self.task_embedding = embedding

    def set_training(self, mode: bool) -> None:
        self.training_mode = mode
        if mode:
            self.clear_inference_cache()

    def clear_inference_cache(self) -> None:
        """Drop any cached prompt banks used for evaluation."""
        self._inference_cache.clear()
        self._inference_task_ids = []

    def start_task(self, task_id: int) -> None:
        """Initialize base prompts and gating for a new task."""
        self.current_task_id = task_id
        device = self.device

        for layer_idx in range(self.num_layers):
            for prompt in self.base_prompts[layer_idx]:
                prompt.requires_grad = False

            new_prompt = nn.Parameter(torch.zeros(self.prompt_length, self.embed_dim, device=device))
            nn.init.normal_(new_prompt, mean=0.0, std=0.02)
            self.base_prompts[layer_idx].append(new_prompt)

        if self.use_adaptive_gating:
            self.current_gate = ProbabilisticGate(num_layers=self.num_layers, **self._gate_config).to(device)
        else:
            self.current_gate = None
        self._latest_layer_cache.clear()
        self._aux_losses.clear()
        self.clear_inference_cache()

    def _stack_prompts(self, layer_idx: int) -> torch.Tensor:
        prompts = list(self.base_prompts[layer_idx])
        if not prompts:
            raise ValueError(f"No base prompts registered for layer {layer_idx}")
        return torch.stack([p for p in prompts], dim=0)

    def _format_prompt(self, prompt: torch.Tensor, gate_value: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Convert prompt embeddings to prefix attention tensors."""
        if prompt.dim() == 2:
            prompt = prompt.unsqueeze(0)

        num_prompts = prompt.shape[0]
        prompt = prompt.view(num_prompts, self.prompt_length, self.num_heads, self.head_dim)
        key_prompt = prompt.clone()
        value_prompt = prompt.clone()

        # Stack key/value and merge prompt dimension into length axis
        prefix = torch.stack([key_prompt, value_prompt], dim=1)  # [num_prompts, 2, length, heads, dim]
        prefix = prefix.permute(1, 0, 2, 3, 4).contiguous()
        prefix = prefix.view(2, num_prompts * self.prompt_length, self.num_heads, self.head_dim)
        prefix = prefix.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).contiguous()

        gate_scale = gate_value.view(1, 1, 1, 1, 1)
        return prefix * gate_scale

    def _prepare_training_prompt(self, layer_idx: int, batch_size: int) -> Optional[torch.Tensor]:
        base_prompts = self._stack_prompts(layer_idx)
        new_prompt = self.base_prompts[layer_idx][-1]

        task_embedding = None
        if self.use_task_conditioning and self.task_embedding is not None:
            task_embedding = self.task_embedding.to(base_prompts.device)

        evo_out = self.evolutions[layer_idx](base_prompts, new_prompt, task_embedding)
        rainbow_prompt = evo_out["rainbow_prompt"]

        if self.use_adaptive_gating:
            if self.current_gate is None:
                raise RuntimeError("Probabilistic gate not initialized. Call start_task() first.")

            gate_out = self.current_gate(
                layer_idx=layer_idx,
                epoch=self.current_epoch,
                max_epochs=self.max_epochs,
                training=self.training_mode,
            )

            gate_value = gate_out["gate"].to(rainbow_prompt.device)
            self._aux_losses[f"sparsity_{layer_idx}"] = gate_out["sparsity_loss"]
        else:
            gate_value = torch.ones((), device=rainbow_prompt.device)

        formatted_prompt = self._format_prompt(rainbow_prompt, gate_value, batch_size)

        self._latest_layer_cache[layer_idx] = {
            "prompt": rainbow_prompt.detach().cpu(),
            "gate": gate_value.detach().cpu(),
        }

        return formatted_prompt

    def _prepare_inference_prompt(self, task_id: int, layer_idx: int, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        if not self._inference_cache:
            self.prepare_inference(task_ids=None, device=device)

        prompt_bank = self._inference_cache.get(layer_idx)
        if prompt_bank is None:
            return None

        prompt_tensor = prompt_bank.to(device)
        gate_value = torch.ones((), device=device)
        return self._format_prompt(prompt_tensor, gate_value, batch_size)

    def prepare_inference(self, task_ids: Optional[list[int]] = None, device: Optional[torch.device] = None) -> None:
        """Assemble prompt banks from all stored tasks for evaluation."""
        device = device or self.device
        if task_ids is None:
            task_ids = self.storage.list_task_ids()

        stacked_per_layer: Dict[int, torch.Tensor] = {}
        available_tasks: list[int] = []

        for task_id in task_ids:
            try:
                self.storage.load_task(task_id, device=device)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"RainbowPrompt task {task_id} not found; aborting inference preparation."
                ) from exc
            available_tasks.append(task_id)

        for layer_idx in range(self.num_layers):
            layer_prompts = []
            for task_id in available_tasks:
                stored = self.storage.get(task_id, layer_idx)
                if stored is None:
                    if task_id < len(self.base_prompts[layer_idx]):
                        prompt = self.base_prompts[layer_idx][task_id].detach().to(device)
                        gate = torch.ones((), device=device)
                    else:
                        continue
                else:
                    prompt = stored["prompt"].to(device)
                    gate = stored["gate"].to(device)

                gate_scalar = gate.view(1)[0]
                prompt = prompt * gate_scalar
                layer_prompts.append(prompt)

            if layer_prompts:
                stacked_per_layer[layer_idx] = torch.stack(layer_prompts, dim=0)

        self._inference_cache = stacked_per_layer
        self._inference_task_ids = available_tasks

        if not available_tasks:
            print("[RainbowPrompt] Warning: no stored tasks found for inference.")
            return

        missing_layers = [idx for idx in range(self.num_layers) if idx not in stacked_per_layer]
        if missing_layers:
            print(f"[RainbowPrompt] Warning: missing prompts for layers {missing_layers}.")

    def forward(
        self,
        task_id: int,
        layer_idx: int,
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if self.training_mode:
            return self._prepare_training_prompt(layer_idx, batch_size)
        return self._prepare_inference_prompt(task_id, layer_idx, batch_size, device)

    def auxiliary_losses(self) -> Dict[str, torch.Tensor]:
        losses = {k: v for k, v in self._aux_losses.items()}
        self._aux_losses.clear()
        return losses

    def finalize_task(self, task_id: int) -> None:
        if not self._latest_layer_cache:
            return
        for layer_idx, data in self._latest_layer_cache.items():
            self.storage.put(task_id, layer_idx, data["prompt"], data["gate"])
        self.storage.save_task(task_id)
        self._latest_layer_cache.clear()

    def load_task(self, task_id: int, device: torch.device) -> None:
        self.storage.load_task(task_id, device=device)

