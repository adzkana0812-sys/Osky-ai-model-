"""
OSKY AI - Multi-Head Causal Self-Attention
Dibangun dari nol (from scratch) tanpa pretrained weights.
Komponen:
1. Query, Key, Value Projections
2. Causal Triangular Attention Mask (autoregressive mask)
3. Scaled Dot-Product Attention: Softmax((Q @ K.T) / sqrt(d_k) + Mask) @ V
4. Support untuk Rotary Position Embeddings (RoPE)
5. Output Linear Projection & Attention Dropout
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .embeddings import apply_rotary_pos_emb

class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        use_rope: bool = True
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) harus habis dibagi num_heads ({num_heads})"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_rope = use_rope
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Matriks proyeksi linier untuk Query, Key, dan Value
        # Menggabungkan Q, K, V dalam satu operasi linear efisien [hidden_dim -> 3 * hidden_dim]
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)

        # Matriks proyeksi keluaran (Output projection W_o)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Buat Causal Attention Mask: lower triangular matrix bernilai 1, upper bernilai 0
        # Token pada posisi t hanya boleh memperhatikan token posisi <= t
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, max_seq_len, max_seq_len), persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        # Inisialisasi bobot Xavier/Kaiming dari nol
        nn.init.normal_(self.qkv_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_heads))

    def forward(
        self,
        x: torch.Tensor,
        rope_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # 1. Proyeksikan ke Query, Key, Value
        qkv = self.qkv_proj(x)  # [batch_size, seq_len, 3 * hidden_dim]
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. Reshape ke format multi-head: [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Terapkan RoPE jika diaktifkan
        if self.use_rope and rope_cos_sin is not None:
            cos, sin = rope_cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 4. Scaled Dot-Product Attention
        # Scores: [batch_size, num_heads, seq_len, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 5. Terapkan Causal Mask (blokir token masa depan dengan -inf)
        sub_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(~sub_mask, float("-inf"))

        # Jika ada padding mask opsional
        if padding_mask is not None:
            # padding_mask: [batch_size, 1, 1, seq_len]
            attn_scores = attn_scores.masked_fill(~padding_mask, float("-inf"))

        # 6. Softmax normalisasi probabilitas atensi
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # 7. Kalikan dengan Value
        # context: [batch_size, num_heads, seq_len, head_dim]
        context = torch.matmul(attn_probs, v)

        # 8. Gabungkan semua head kembali ke [batch_size, seq_len, hidden_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        # 9. Output Linear Projection
        out = self.out_proj(context)
        out = self.resid_dropout(out)
        return out
