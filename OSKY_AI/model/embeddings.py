"""
OSKY AI - Neural Network Embeddings & Positional Representations
Dibangun dari nol (from scratch) tanpa weights pretrained.
Menyediakan:
1. Token Embedding
2. Learned Positional Embedding
3. Modern Rotary Position Embedding (RoPE) - seperti yang digunakan pada LLM modern (LLaMA/Mistral/DeepSeek)
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional

class OskyEmbedding(nn.Module):
    """
    Token embedding layer yang memetakan token ID diskrit ke ruang vektor berdimensi hidden_dim.
    Inisialisasi weights menggunakan distribusi normal dengan mean 0 dan std 0.02.
    """
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.lut = nn.Embedding(vocab_size, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.lut.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len] -> output: [batch_size, seq_len, hidden_dim]
        return self.lut(x)


class LearnedPositionalEmbedding(nn.Module):
    """
    Representasi posisi yang dipelajari (learned positional embeddings).
    Setiap indeks posisi 0..max_seq_len-1 memiliki vektor berdimensi hidden_dim.
    """
    def __init__(self, max_seq_len: int, hidden_dim: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.pos_lut = nn.Embedding(max_seq_len, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.pos_lut.weight, mean=0.0, std=0.02)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # returns [1, seq_len, hidden_dim]
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        return self.pos_lut(positions)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) - Arsitektur Modern.
    Menerapkan rotasi ortogonal pada ruang 2D ke Query dan Key berdasarkan posisi token.
    Keunggulan:
    - Tidak memakan parameter yang harus disimpan di weights
    - Mempertahankan sifat relative distance antar token secara elegan
    - Generalisasi panjang konteks yang lebih stabil
    """
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Hitung frekuensi theta: theta_i = 1.0 / (base ** (2i / dim))
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos dan sin cache
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # [max_seq_len, dim // 2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_seq_len, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cos_cached[:seq_len].to(device),
            self.sin_cached[:seq_len].to(device)
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotasi vektor 2D: [-x2, x1]"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Menerapkan rotasi RoPE ke Query dan Key:
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    """
    # cos, sin: [seq_len, head_dim] -> reshape untuk broadcasting [1, 1, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
