"""
OSKY AI - Vision Evaluation Metrics
Mengukur kualitas representasi fitur visual, patch attention entropy, dan reconstruction MSE.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any

def evaluate_vision_features(vision_model, images: torch.Tensor) -> Dict[str, float]:
    vision_model.eval()
    with torch.no_grad():
        out = vision_model(images)
        patch_tokens = out["patch_tokens"]  # [B, N, D]
        cls_token = out["cls_token"]        # [B, D]

        # 1. Norm magnitudo
        cls_norm = float(torch.norm(cls_token, dim=-1).mean().item())
        # 2. Patch variance
        patch_var = float(torch.var(patch_tokens, dim=-1).mean().item())
        # 3. Pairwise cosine similarity antar patch (mengukur keragaman spasial)
        norm_patches = F.normalize(patch_tokens, p=2, dim=-1)
        sim_matrix = torch.bmm(norm_patches, norm_patches.transpose(1, 2))
        avg_sim = float(sim_matrix.mean().item())

    return {
        "cls_norm": round(cls_norm, 4),
        "patch_variance": round(patch_var, 6),
        "avg_patch_similarity": round(avg_sim, 4)
    }
