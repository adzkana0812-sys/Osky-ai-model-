"""
OSKY AI - Vision Model & Image Preprocessor
Membungkus patch embedding, N-layer vision transformer blocks, dan fungsi preprocessing citra.
Mendukung ekstraksi representasi patch spasial dan token global [CLS].
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Union, Optional
from PIL import Image
import numpy as np

from OSKY_AI.vision.patch_embedding import PatchEmbedding
from OSKY_AI.vision.image_encoder import VisionTransformerBlock

def preprocess_image(
    image_input: Union[str, Image.Image, np.ndarray, torch.Tensor],
    image_size: int = 64
) -> torch.Tensor:
    """
    Mengubah input gambar apa saja (path file, objek PIL, array numpy)
    menjadi tensor float32 berbentuk [1, 3, image_size, image_size] ternormalisasi.
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"File gambar tidak ditemukan: {image_input}")
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input.astype('uint8')).convert("RGB")
    elif isinstance(image_input, torch.Tensor):
        if image_input.dim() == 3:
            image_input = image_input.unsqueeze(0)
        return image_input.float()
    else:
        raise TypeError(f"Tipe input gambar tidak didukung: {type(image_input)}")

    # Resize gambar dengan bi-linear interpolation
    pil_img = pil_img.resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = np.array(pil_img, dtype=np.float32) / 255.0  # Range [0.0, 1.0]

    # Normalisasi standar: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    # Ubah [H, W, C] -> [C, H, W] -> [1, C, H, W]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor

class OskyVisionTransformer(nn.Module):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        config = config or {}
        self.image_size = config.get("vision_image_size", 64)
        self.patch_size = config.get("vision_patch_size", 8)
        self.in_channels = config.get("in_channels", 3)
        self.embed_dim = config.get("vision_embed_dim", 128)
        self.num_layers = config.get("vision_layers", 4)
        self.num_heads = config.get("vision_heads", 4)
        self.dropout = config.get("dropout", 0.0)

        # 1. Patch Embedding Layer
        self.patch_embed = PatchEmbedding(
            image_size=self.image_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            dropout=self.dropout
        )

        # 2. Vision Transformer Blocks
        self.blocks = nn.ModuleList([
            VisionTransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=4.0,
                dropout=self.dropout
            )
            for _ in range(self.num_layers)
        ])

        # 3. Final Layer Normalization
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Input: pixel_values [B, C, H, W]
        Returns:
            "cls_token": [B, embed_dim]
            "patch_tokens": [B, num_patches, embed_dim]
            "all_tokens": [B, num_patches + 1, embed_dim]
        """
        x = self.patch_embed(pixel_values)  # [B, N+1, embed_dim]

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_token = x[:, 0]
        patch_tokens = x[:, 1:]

        return {
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
            "all_tokens": x
        }
