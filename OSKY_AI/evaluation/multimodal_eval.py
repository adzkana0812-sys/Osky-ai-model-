"""
OSKY AI - Multimodal Evaluation Metrics
Mengevaluasi performa penalaran visual-ke-bahasa (VQA / Image-to-Text).
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any

def evaluate_multimodal_qa(multimodal_model, pixel_values: torch.Tensor, input_ids: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    multimodal_model.eval()
    with torch.no_grad():
        out = multimodal_model(pixel_values, input_ids, labels=labels)
        loss = out["loss"]
        loss_val = float(loss.item()) if loss is not None else 0.0

    return {
        "multimodal_vqa_loss": round(loss_val, 4),
        "vqa_perplexity": round(float(torch.exp(torch.tensor(min(loss_val, 20.0)))), 2)
    }
