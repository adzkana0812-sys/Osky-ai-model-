"""
OSKY AI - Multimodal Projection Layer
Menjembatani ruang fitur visual (Vision Transformer) dengan ruang representasi bahasa (OSKY Language Model).
Mentransformasikan tensor visual berdimensi [B, N_patches, vision_dim]
menjadi tensor token visual [B, N_patches, llm_hidden_dim].
Dibangun dari random initialization (MLP dengan LayerNorm & GELU).
"""

import torch
import torch.nn as nn

class MultimodalProjector(nn.Module):
    def __init__(self, vision_dim: int = 128, llm_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim

        # 2-Layer MLP Projector dengan LayerNorm dan GELU
        self.net = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, llm_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim, llm_dim, bias=True)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Input:  [B, num_patches, vision_dim]
        Output: [B, num_patches, llm_dim]
        """
        return self.net(visual_features)
