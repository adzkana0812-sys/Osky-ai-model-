"""
OSKY AI - Handwriting Training Script
Melatih model OskyHandwritingRecognizer dari nol menggunakan loss CTC (Connectionist Temporal Classification).
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from OSKY_AI.paths import resolve_path
from OSKY_AI.handwriting.dataset import OskyHandwritingDataset, HandwritingCharVocabulary
from OSKY_AI.handwriting.model import OskyHandwritingRecognizer
from OSKY_AI.hardware import get_hardware_info

def collate_handwriting(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)
    texts = [item["text"] for item in batch]
    target_lengths = torch.tensor([item["target_length"] for item in batch], dtype=torch.long)
    targets = torch.cat([item["target"] for item in batch], dim=0)
    return images, targets, target_lengths, texts

def train_handwriting(
    data_path: str = "datasets/handwriting/sample.jsonl",
    epochs: int = 10,
    lr: float = 1e-3,
    save_path: str = "checkpoints/osky_handwriting.pt"
):
    resolved_data = resolve_path(data_path)
    resolved_save = resolve_path(save_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[Handwriting Trainer] Menggunakan device: {device}")

    vocab = HandwritingCharVocabulary()
    dataset = OskyHandwritingDataset(resolved_data, vocab=vocab)
    
    # Jika dataset kosong, buat 1 sampel dummy untuk pengujian
    if len(dataset) == 0:
        print("[Handwriting Trainer] Peringatan: dataset kosong, menggunakan sampel placeholder.")
        dummy_tensor = torch.zeros(1, 32, 128)
        dataset.samples = [{"image": "dummy.png", "text": "OSKY AI 2026"}]

    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_handwriting)

    model = OskyHandwritingRecognizer(vocab_size=vocab.vocab_size)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    print(f"[Handwriting Trainer] Total Parameter: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for images, targets, target_lengths, texts in loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            # forward: [B, W_seq, vocab_size]
            log_probs = model(images)
            # CTCLoss mengharapkan input shape [Input_Length, Batch_Size, Vocab_Size]
            log_probs_t = log_probs.permute(1, 0, 2)
            input_lengths = torch.full((images.size(0),), log_probs.size(1), dtype=torch.long, device=device)

            loss = ctc_loss(log_probs_t, targets, input_lengths, target_lengths)
            if not torch.isnan(loss):
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        if epoch % 2 == 0 or epoch == epochs:
            print(f"[Epoch {epoch:02d}/{epochs:02d}] CTC Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(resolved_save), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "vocab": vocab.chars}, resolved_save)
    print(f"[Handwriting Trainer] Checkpoint tersimpan di {resolved_save}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="datasets/handwriting/sample.jsonl")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    train_handwriting(data_path=args.data, epochs=args.epochs)
