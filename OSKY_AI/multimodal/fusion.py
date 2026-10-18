"""
OSKY AI - Multimodal Fusion Engine
Menggabungkan urutan embedding teks dan token visual hasil proyeksi
menjadi satu stream representasi terpadu yang dapat diproses langsung oleh OSKY Language Model.
Format stream:
[visual_tokens (N_patches)] + [text_tokens (L_text)] -> Joint Sequence [B, N_patches + L_text, hidden_dim]
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

class MultimodalFusion(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Layer normalisasi untuk menyeimbangkan magnitudo visual dan teks sebelum masuk transformer
        self.joint_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(
        self,
        projected_visual_tokens: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Input:
            projected_visual_tokens: [B, N_vis, hidden_dim]
            text_embeddings:         [B, L_txt, hidden_dim]
            text_padding_mask:       [B, L_txt] (1 = valid, 0 = pad)
        Output:
            joint_embeddings:        [B, N_vis + L_txt, hidden_dim]
            joint_mask:              [B, N_vis + L_txt]
        """
        B, N_vis, D = projected_visual_tokens.shape
        _, L_txt, _ = text_embeddings.shape

        # Concatenate secara sekuensial di dimensi panjang sequence (dim=1)
        joint_embeddings = torch.cat([projected_visual_tokens, text_embeddings], dim=1)
        joint_embeddings = self.joint_norm(joint_embeddings)

        joint_mask = None
        if text_padding_mask is not None:
            # Token visual selalu valid (nilai 1)
            vis_mask = torch.ones((B, N_vis), dtype=text_padding_mask.dtype, device=text_padding_mask.device)
            joint_mask = torch.cat([vis_mask, text_padding_mask], dim=1)

        return joint_embeddings, joint_mask
