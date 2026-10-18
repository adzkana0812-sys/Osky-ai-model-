"""
OSKY AI - Image Generation Training Loop
Melatih model generatif (OskyDiffusionUNet / OskyConvVAE) dari bobot acak.
Menggunakan Mean Squared Error (MSE) loss antara noise asli dan prediksi noise model.
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from OSKY_AI.image_generation.diffusion import OskyDiffusionUNet
from OSKY_AI.image_generation.generator import LocalTextConditioner
from OSKY_AI.image_generation.dataset import OskyImageGenDataset

def train_diffusion(
    data_path: str = "datasets/image_generation/sample.jsonl",
    epochs: int = 5,
    lr: float = 2e-4,
    save_path: str = "checkpoints/osky_diffusion.pt"
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[Image Gen Trainer] Menggunakan device: {device}")

    dataset = OskyImageGenDataset(data_path, image_size=32)
    if len(dataset) == 0:
        print("[Image Gen Trainer] Memuat dataset sintetis dasar untuk verifikasi.")
        dataset.samples = [{"image": "synthetic.png", "prompt": "lingkaran merah"}]

    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = OskyDiffusionUNet(in_channels=3, out_channels=3, hidden_dim=32, cond_dim=128, num_timesteps=20).to(device)
    conditioner = LocalTextConditioner(cond_dim=128).to(device)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(conditioner.parameters()), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in loader:
            imgs = batch["image"].to(device)
            prompts = batch["prompt"]

            optimizer.zero_grad()
            # Buat kondisi teks
            cond = conditioner(prompts[0], device=device)

            # Pilih timestep acak t
            t = torch.randint(0, model.num_timesteps, (imgs.size(0),), device=device).long()
            noise = torch.randn_like(imgs)

            # Sample forward diffusion
            x_noisy, injected_noise = model.q_sample(imgs, t, noise)

            # Prediksi noise
            pred_noise = model(x_noisy, t, cond)

            loss = F.mse_loss(pred_noise, injected_noise)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        print(f"[Epoch {epoch:02d}/{epochs:02d}] Diffusion MSE Noise Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, save_path)
    print(f"[Image Gen Trainer] Model difusi tersimpan di {save_path}")

if __name__ == "__main__":
    train_diffusion()
