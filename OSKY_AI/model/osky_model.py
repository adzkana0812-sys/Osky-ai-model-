"""
OSKY AI - Full Neural Network Architecture (OskyLanguageModel)
Dibangun 100% dari nol (random weight initialization).
Sama sekali TIDAK menggunakan pretrained weights dari model manapun.
Komponen utama:
1. Token Embedding Layer
2. Positional Representation (RoPE atau Learned Positional Embedding)
3. N-Layer Transformer Blocks (Pre-RMSNorm + Multi-Head Causal Self-Attention + SwiGLU FFN + Residuals)
4. Final Normalization Layer
5. Output Language Model Head (Linear projection ke vocab_size)
6. Cross-Entropy Loss computation dengan masking ignore_index=-100
7. Automatic Parameter Counter
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .embeddings import OskyEmbedding, LearnedPositionalEmbedding, RotaryPositionalEmbedding
from .transformer import TransformerBlock, RMSNorm, StandardLayerNorm

class OskyLanguageModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        self.vocab_size: int = config["vocab_size"]
        self.hidden_dim: int = config["hidden_dim"]
        self.num_layers: int = config["num_layers"]
        self.num_heads: int = config["num_heads"]
        self.max_seq_len: int = config.get("max_seq_len", 512)
        self.ffn_dim: int = config.get("ffn_dim", 4 * self.hidden_dim)
        self.dropout: float = config.get("dropout", 0.1)
        self.norm_type: str = config.get("norm_type", "rmsnorm")
        self.activation: str = config.get("activation", "silu")
        self.use_rope: bool = config.get("use_rope", True)
        self.tie_weights: bool = config.get("tie_weights", True)

        # 1. Token Embeddings
        self.token_emb = OskyEmbedding(self.vocab_size, self.hidden_dim)
        self.emb_dropout = nn.Dropout(self.dropout)

        # 2. Positional Representation
        if self.use_rope:
            head_dim = self.hidden_dim // self.num_heads
            self.rope = RotaryPositionalEmbedding(dim=head_dim, max_seq_len=self.max_seq_len)
            self.pos_emb = None
        else:
            self.rope = None
            self.pos_emb = LearnedPositionalEmbedding(self.max_seq_len, self.hidden_dim)

        # 3. Stack of Transformer Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                ffn_dim=self.ffn_dim,
                max_seq_len=self.max_seq_len,
                dropout=self.dropout,
                norm_type=self.norm_type,
                activation=self.activation,
                use_rope=self.use_rope,
            )
            for _ in range(self.num_layers)
        ])

        # 4. Final Normalization
        if self.norm_type == "rmsnorm":
            self.final_norm = RMSNorm(self.hidden_dim)
        else:
            self.final_norm = StandardLayerNorm(self.hidden_dim)

        # 5. Output LM Head (proyeksi linier ke vocabulary)
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

        # Weight tying: mengikat bobot LM head dengan Token Embedding
        # Mengurangi jumlah parameter dan meningkatkan stabilitas representasi semantik
        if self.tie_weights:
            self.lm_head.weight = self.token_emb.lut.weight

        # 6. Inisialisasi seluruh parameter secara random dari nol
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Inisialisasi bobot Gaussian dari nol (N(0, 0.02)) untuk seluruh layer."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> Dict[str, Any]:
        """
        Menghitung jumlah parameter model secara otomatis dan estimasi memori.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Parameter non-embedding (pure transformer compute)
        emb_params = sum(p.numel() for p in self.token_emb.parameters())
        non_emb_params = total_params - emb_params

        # Ukuran memori (Float32 = 4 bytes per param, Float16/BFloat16 = 2 bytes)
        mem_mb_fp32 = (total_params * 4) / (1024 * 1024)
        mem_mb_fp16 = (total_params * 2) / (1024 * 1024)

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "non_embedding_params": non_emb_params,
            "memory_fp32_mb": round(mem_mb_fp32, 2),
            "memory_fp16_mb": round(mem_mb_fp16, 2),
            "human_readable": f"{total_params / 1e6:.2f}M parameters" if total_params >= 1e6 else f"{total_params / 1e3:.2f}K parameters"
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass neural network.
        Args:
            input_ids: [batch_size, seq_len] tensor token ID
            labels: [batch_size, seq_len] tensor target token ID (opsional untuk training)
            padding_mask: [batch_size, seq_len] tensor boolean
        Returns:
            (loss, logits) jika labels diberikan, atau (None, logits) untuk inference
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} melebihi batas max_seq_len ({self.max_seq_len})")

        # 1. Token Embeddings
        x = self.token_emb(input_ids)  # [batch_size, seq_len, hidden_dim]

        # 2. Positional Embeddings
        rope_cos_sin = None
        if self.use_rope and self.rope is not None:
            rope_cos_sin = self.rope(seq_len, device)
        elif self.pos_emb is not None:
            x = x + self.pos_emb(seq_len, device)

        x = self.emb_dropout(x)

        # Reshape padding mask untuk atensi jika diberikan
        attn_padding_mask = None
        if padding_mask is not None:
            attn_padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]

        # 3. Alirkan melalui N Transformer Blocks
        for layer in self.layers:
            x = layer(x, rope_cos_sin=rope_cos_sin, padding_mask=attn_padding_mask)

        # 4. Final Normalization
        x = self.final_norm(x)

        # 5. Output Projection ke Vocabulary Logits
        # logits: [batch_size, seq_len, vocab_size]
        logits = self.lm_head(x)

        # 6. Hitung Cross-Entropy Loss jika labels diberikan
        loss = None
        if labels is not None:
            # Flatten tensor untuk cross-entropy
            # logits: [batch_size * seq_len, vocab_size]
            # labels: [batch_size * seq_len]
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100  # Masking untuk prompt / padding
            )

        return loss, logits
