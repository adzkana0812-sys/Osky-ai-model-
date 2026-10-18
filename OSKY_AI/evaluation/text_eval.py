"""
OSKY AI - Text Evaluation Metrics
Menghitung Validation Loss, Perplexity (PPL), token generation speed, dan word overlap.
Kompatibel 100% dengan DataLoader tuple (input_ids, labels, masks) maupun format dict.
"""

import math
import time
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Union, Tuple

def evaluate_text_perplexity(model: torch.nn.Module, dataloader, device: str = "cpu") -> Dict[str, float]:
    """
    Mengevaluasi loss dan perplexity (PPL) secara konsisten pada dataset validasi.
    Mendukung format output DataLoader baik tuple (input_ids, labels, masks)
    maupun dictionary {"input_ids": ..., "target_ids": ..., "loss_mask": ...}.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                if len(batch) >= 3:
                    input_ids, labels, masks = batch[0], batch[1], batch[2]
                elif len(batch) == 2:
                    input_ids, labels = batch[0], batch[1]
                    masks = (labels != -100).long()
                else:
                    input_ids = batch[0]
                    labels = input_ids.clone()
                    masks = torch.ones_like(input_ids)
            elif isinstance(batch, dict):
                input_ids = batch.get("input_ids")
                labels = batch.get("target_ids", batch.get("labels", input_ids))
                masks = batch.get("loss_mask", batch.get("masks", (labels != -100).long()))
            else:
                continue

            input_ids = input_ids.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            # Hitung loss melalui forward pass model
            out = model(input_ids, labels=labels, padding_mask=masks)
            if isinstance(out, tuple):
                loss, logits = out
            elif isinstance(out, dict):
                loss = out.get("loss")
                logits = out.get("logits")
            else:
                loss = None
                logits = out

            if loss is not None and not torch.isnan(loss):
                valid_count = (labels != -100).sum().item() if (labels != -100).any() else masks.sum().item()
                valid_count = max(1, valid_count)
                total_loss += loss.item() * valid_count
                total_tokens += valid_count
                total_batches += 1
            elif logits is not None:
                # Hitung cross-entropy manual jika loss tidak dikembalikan langsung
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                shift_mask = masks[:, 1:].contiguous()

                ce_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                    ignore_index=-100
                )
                masked_loss = (ce_loss * shift_mask.view(-1)).sum()
                valid_count = (shift_mask.sum().item())
                if valid_count > 0:
                    total_loss += masked_loss.item()
                    total_tokens += valid_count
                    total_batches += 1

    avg_loss = (total_loss / max(1, total_tokens)) if total_tokens > 0 else 0.0
    ppl = math.exp(min(avg_loss, 20.0))

    return {
        "val_loss": round(avg_loss, 4),
        "perplexity": round(ppl, 2),
        "total_evaluated_tokens": total_tokens,
        "total_evaluated_batches": total_batches
    }
