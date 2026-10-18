"""
OSKY AI - Transformer Blocks, Normalizations, & Feed-Forward Networks
Dibangun dari nol (from scratch) tanpa pretrained weights.
Komponen:
1. RMSNorm (Root Mean Square Normalization) & LayerNorm
2. SwiGLU / GELU Feed-Forward Network (FFN)
3. Pre-LN Transformer Block dengan Residual Connections
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .attention import CausalSelfAttention

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Komponen modern yang menggantikan LayerNorm konvensional tanpa mean centering,
    meningkatkan kecepatan komputasi dan kestabilan gradien secara signifikan.
    Formula: y = (x / RMS(x)) * gamma, di mana RMS(x) = sqrt(mean(x^2) + eps)
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm


class StandardLayerNorm(nn.Module):
    """Layer Normalization standar dengan mean dan variance."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        variance = (x - mean).pow(2).mean(-1, keepdim=True)
        x_norm = (x - mean) * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm + self.bias


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network (Gated Linear Unit dengan SiLU/Swish).
    Arsitektur modern pilihan LLM generasi baru (LLaMA/Mistral/Qwen):
    FFN(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
    """
    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gated activation: SiLU(gate) * up
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        activated = gate * up
        out = self.down_proj(activated)
        return self.dropout(out)


class GELUFeedForward(nn.Module):
    """MLP klasik dengan aktivasi Gaussian Error Linear Unit (GELU)."""
    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_dim, bias=True)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(x))
        out = self.fc2(h)
        return self.dropout(out)


class TransformerBlock(nn.Module):
    """
    Blok Transformer Lengkap dengan Pre-LayerNorm / Pre-RMSNorm:
    Input x -> Norm 1 -> Multi-Head Causal Attention -> + Residual x
            -> Norm 2 -> Feed-Forward Network -> + Residual
    Pre-LN memastikan gradien dapat mengalir langsung melalui residual highway
    tanpa teredam, membuat training deep network sangat stabil sejak inisialisasi random.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        norm_type: str = "rmsnorm",
        activation: str = "silu",
        use_rope: bool = True,
    ):
        super().__init__()
        self.norm_type = norm_type

        # 1. Normalisasi sebelum Attention (Pre-LN)
        if norm_type == "rmsnorm":
            self.attn_norm = RMSNorm(hidden_dim)
            self.ffn_norm = RMSNorm(hidden_dim)
        else:
            self.attn_norm = StandardLayerNorm(hidden_dim)
            self.ffn_norm = StandardLayerNorm(hidden_dim)

        # 2. Causal Self-Attention
        self.attention = CausalSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            use_rope=use_rope,
        )

        # 3. Feed-Forward Network
        if activation == "silu":
            self.feed_forward = SwiGLUFeedForward(hidden_dim, ffn_dim, dropout)
        else:
            self.feed_forward = GELUFeedForward(hidden_dim, ffn_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Sublayer 1: Causal Self-Attention dengan Residual Connection
        norm_x = self.attn_norm(x)
        attn_out = self.attention(norm_x, rope_cos_sin=rope_cos_sin, padding_mask=padding_mask)
        x = x + attn_out  # Residual connection 1

        # Sublayer 2: Feed-Forward Network dengan Residual Connection
        norm_x = self.ffn_norm(x)
        ffn_out = self.feed_forward(norm_x)
        x = x + ffn_out   # Residual connection 2

        return x
