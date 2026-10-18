"""
OSKY AI - Optimizer & Learning Rate Scheduler
Fitur:
1. AdamW Optimizer dengan Decoupled Weight Decay:
   - Weight decay (misal 0.01) hanya diterapkan pada bobot matriks 2D (Linear layers).
   - Bias dan layer normalisasi (RMSNorm/LayerNorm) dikecualikan dari weight decay (decay = 0.0).
   - Ini adalah standar wajib dalam pelatihan modern LLM untuk mencegah degradasi representasi.
2. Cosine Annealing Learning Rate Scheduler dengan Linear Warmup:
   - Tahap Warmup: Menaikkan LR secara linier dari 0 hingga target learning rate.
   - Tahap Cosine Decay: Menurunkan LR mengikuti kurva cosinus hingga min_learning_rate.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Any

def configure_optimizers(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """
    Mengonfigurasi AdamW dengan memisahkan parameter yang dikenai weight decay
    dan parameter yang TIDAK dikenai weight decay (bias, norm weights).
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Parameter 1D (bias, norm weights) tidak didecay
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps)
    print(f"[Optimizer] Parameter terdecay: {len(decay_params)} tensor, Tanpa decay: {len(no_decay_params)} tensor")
    return optimizer


class CosineWarmupLRScheduler:
    """
    Learning rate scheduler dengan Linear Warmup dan Cosine Decay.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        max_lr: float,
        min_lr: float = 0.0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def get_lr(self) -> float:
        # 1. Linear Warmup
        if self.current_step < self.warmup_steps:
            return self.max_lr * float(self.current_step) / float(max(1, self.warmup_steps))

        # 2. Melewati batas max_steps -> gunakan min_lr
        if self.current_step > self.max_steps:
            return self.min_lr

        # 3. Cosine Annealing Decay
        decay_ratio = float(self.current_step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "current_step": self.current_step,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.current_step = state["current_step"]
        self.warmup_steps = state.get("warmup_steps", self.warmup_steps)
        self.max_steps = state.get("max_steps", self.max_steps)
        self.max_lr = state.get("max_lr", self.max_lr)
        self.min_lr = state.get("min_lr", self.min_lr)
