"""
OSKY AI - Patch Embedding Layer
Membagi citra 2D (B, C, H, W) menjadi urutan patch non-overlapping (B, N, D),
menambahkan token khusus [CLS], dan menyematkan representasi posisi terpelajar (learnable 1D/2D pos embeddings).
Dibangun 100% dari random initialization (Bukan Pretrained CLIP/ViT).
"""

import math
import torch
import torch.nn as nn
from typing import Tuple

class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 128,
        dropout: float = 0.0
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        assert image_size % patch_size == 0, f"Ukuran gambar ({image_size}) harus habis dibagi patch_size ({patch_size})"

        self.num_patches_h = image_size // patch_size
        self.num_patches_w = image_size // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Proyeksi linier patch menggunakan Conv2d dengan kernel dan stride sama dengan patch_size
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )

        # Token representasi visual global [CLS]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable Positional Embeddings untuk (num_patches + 1 CLS)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        # Inisialisasi acak Gaussian (Random Initialization From Scratch)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.kaiming_normal_(self.projection.weight, mode="fan_out", nonlinearity="relu")
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  x [Batch, Channels, Height, Width]
        Output: [Batch, num_patches + 1, embed_dim]
        """
        B, C, H, W = x.shape
        assert H == self.image_size and W == self.image_size, (
            f"Dimensi gambar input ({H}x{W}) harus sesuai dengan konfigurasi ({self.image_size}x{self.image_size})"
        )

        # Proyeksi patch: (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        feat = self.projection(x)
        # Flatten spasial dan transpose: (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        feat = feat.flatten(2).transpose(1, 2)

        # Gabungkan token CLS di posisi pertama
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_out = torch.cat((cls_tokens, feat), dim=1)

        # Tambahkan positional embedding
        x_out = x_out + self.pos_embedding
        x_out = self.dropout(x_out)
        return x_out
