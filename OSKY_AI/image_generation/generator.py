"""
OSKY AI - Unified Image Generator Engine
Menghubungkan Text Conditioner lokal dengan Model Generatif (Diffusion UNet & VAE).
Memproses text prompt dari user -> mengekstraksi learned embedding kondisi teks ->
menjalankan sampling proses generatif -> menghasilkan file citra/objek PIL.
100% Offline & From Scratch dengan weights yang trainable.
"""

import os
import hashlib
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from typing import Optional, Union, List, Dict, Any

from OSKY_AI.paths import resolve_path
from OSKY_AI.image_generation.diffusion import OskyDiffusionUNet
from OSKY_AI.image_generation.autoencoder import OskyConvVAE

class LocalTextConditioner(nn.Module):
    """
    Ekstraktor kondisi teks mandiri berbasis learned token embeddings dan MLP projection.
    100% trainable dari scratch tanpa dependensi pretrained eksternal (CLIP/BERT/OpenAI).
    Bobot embedding dapat di-update melalui backpropagation saat melatih model difusi atau VAE.
    """
    def __init__(
        self,
        vocab_size: int = 2048,
        embed_dim: int = 64,
        cond_dim: int = 128,
        tokenizer: Optional[Any] = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self.tokenizer = tokenizer

        # 1. Trainable Token Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 2. Trainable Projection MLP
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.LayerNorm(cond_dim)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)

    def _text_to_ids(self, text: str) -> List[int]:
        """Konversi teks ke sequence token ID lokal."""
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            ids = self.tokenizer.encode(text)
            return [i % self.vocab_size for i in ids] if ids else [0]

        # Simple whitespace & character-level token mapping
        words = text.lower().strip().split()
        if not words:
            return [0]
        ids = []
        for word in words:
            # Deterministic word hash modulo vocab_size (bukan statis feature hash, melainkan token ID)
            token_id = (int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % (self.vocab_size - 1)) + 1
            ids.append(token_id)
        return ids

    def forward(self, text: Union[str, List[str]], device: str = "cpu") -> torch.Tensor:
        """
        Forward pass ekstraksi kondisi teks.
        Mengembalikan tensor representasi teks: [B, cond_dim] yang fully trainable.
        """
        if isinstance(text, str):
            text_list = [text]
        else:
            text_list = text

        batch_ids = [self._text_to_ids(t) for t in text_list]
        max_len = max(len(ids) for ids in batch_ids)

        # Padding tensor
        padded = torch.zeros((len(batch_ids), max_len), dtype=torch.long, device=device)
        mask = torch.zeros((len(batch_ids), max_len, 1), dtype=torch.float32, device=device)

        for i, ids in enumerate(batch_ids):
            padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            mask[i, :len(ids), 0] = 1.0

        # Lookup embeddings: [B, L, embed_dim]
        embs = self.embedding(padded)

        # Masked average pooling
        sum_embs = (embs * mask).sum(dim=1)  # [B, embed_dim]
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = sum_embs / denom

        # Proyeksikan ke dimensi kondisi difusi
        return self.proj(pooled)

class OskyImageGenerator:
    def __init__(
        self,
        image_size: int = 32,
        cond_dim: int = 128,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.cond_dim = cond_dim

        # Inisialisasi model dari random initialization
        self.text_conditioner = LocalTextConditioner(cond_dim=cond_dim).to(self.device)
        self.diffusion_model = OskyDiffusionUNet(
            in_channels=3,
            out_channels=3,
            hidden_dim=32,
            cond_dim=cond_dim,
            num_timesteps=20
        ).to(self.device)

        self.vae_model = OskyConvVAE(
            in_channels=3,
            latent_dim=64,
            image_size=image_size,
            base_channels=32
        ).to(self.device)

    def load_checkpoint(self, checkpoint_path: str):
        """Memuat bobot model generatif dari file checkpoint."""
        resolved = resolve_path(checkpoint_path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Checkpoint gambar tidak ditemukan: {checkpoint_path} (resolved: {resolved})")

        checkpoint = torch.load(resolved, map_location=self.device)
        if "diffusion_state_dict" in checkpoint:
            self.diffusion_model.load_state_dict(checkpoint["diffusion_state_dict"])
        if "vae_state_dict" in checkpoint:
            self.vae_model.load_state_dict(checkpoint["vae_state_dict"])
        if "conditioner_state_dict" in checkpoint:
            self.text_conditioner.load_state_dict(checkpoint["conditioner_state_dict"])
        print(f"[OskyImageGenerator] Berhasil memuat bobot dari {resolved}")

    def generate_from_prompt(
        self,
        prompt: str,
        mode: str = "diffusion",
        save_path: Optional[str] = None
    ) -> Image.Image:
        """
        Menghasilkan gambar berdasarkan prompt teks.
        Mode: 'diffusion' atau 'vae'
        """
        self.text_conditioner.eval()
        with torch.no_grad():
            cond = self.text_conditioner(prompt, device=self.device)

            if mode == "diffusion":
                self.diffusion_model.eval()
                img_tensor = self.diffusion_model.p_sample_loop(
                    shape=(1, 3, self.image_size, self.image_size),
                    text_cond=cond,
                    device=self.device
                )  # Nilai [0, 1]
            else:
                # Mode VAE latent sampling
                self.vae_model.eval()
                z = cond[:, :self.vae_model.latent_dim]
                img_tensor = self.vae_model.decode(z)

            # Konversi tensor [1, 3, H, W] -> PIL Image
            arr = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(arr)

            if save_path:
                resolved_save = resolve_path(save_path)
                dir_name = os.path.dirname(resolved_save)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)
                pil_img.save(resolved_save)

            return pil_img
