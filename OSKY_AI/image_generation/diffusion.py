"""
OSKY AI - Lightweight Diffusion Model (Score-based DDPM from Scratch)
Mengimplementasikan model difusi bersyarat (Conditioned Denoising Diffusion Probabilistic Model).
Menerima tensor citra ber-noise x_t, langkah waktu t, dan vektor kondisi teks c.
Memprediksi komponen noise yang diinjeksikan secara iteratif.
100% Random Initialization tanpa model pretrained Stable Diffusion / Midjourney.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.cond_proj = nn.Linear(cond_dim, out_channels)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        # Injeksi kondisi teks & waktu ke feature map: [B, C, 1, 1]
        h = h + self.cond_proj(cond)[:, :, None, None]
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.shortcut(x)

class OskyDiffusionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_dim: int = 64,
        cond_dim: int = 128,
        num_timesteps: int = 100
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        # Encoder Blocks
        self.init_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.block1 = ResidualBlock(hidden_dim, hidden_dim, cond_dim)
        self.down1 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)

        self.block2 = ResidualBlock(hidden_dim * 2, hidden_dim * 2, cond_dim)
        self.down2 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1)

        # Bottleneck
        self.mid_block = ResidualBlock(hidden_dim * 4, hidden_dim * 4, cond_dim)

        # Decoder Blocks with Skips
        self.up1 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.block3 = ResidualBlock(hidden_dim * 4, hidden_dim * 2, cond_dim)

        self.up2 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.block4 = ResidualBlock(hidden_dim * 2, hidden_dim, cond_dim)

        # Final Output Conv
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
        )

        # Setup DDPM Schedules (Linear Betas)
        betas = torch.linspace(1e-4, 0.02, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def forward(self, x: torch.Tensor, t: torch.Tensor, text_cond: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x:         [B, C, H, W]
            t:         [B] timestep integer (0 .. num_timesteps - 1)
            text_cond: [B, cond_dim] representasi teks
        Output:
            predicted_noise: [B, C, H, W]
        """
        # Gabungkan embedding waktu dan kondisi teks
        t_emb = self.time_embed(t)
        cond = t_emb + text_cond

        # Encoder
        x0 = self.init_conv(x)
        h1 = self.block1(x0, cond)
        d1 = self.down1(h1)

        h2 = self.block2(d1, cond)
        d2 = self.down2(h2)

        # Bottleneck
        mid = self.mid_block(d2, cond)

        # Decoder with Skips
        u1 = self.up1(mid)
        cat1 = torch.cat([u1, h2], dim=1)
        h3 = self.block3(cat1, cond)

        u2 = self.up2(h3)
        cat2 = torch.cat([u2, h1], dim=1)
        h4 = self.block4(cat2, cond)

        out_noise = self.final_conv(h4)
        return out_noise

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward Diffusion: tambahkan noise ke x_0 sesuai timestep t"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_noisy, noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape: Tuple[int, int, int, int],
        text_cond: torch.Tensor,
        device: str = "cpu"
    ) -> torch.Tensor:
        """Reverse Diffusion: generate citra dari pure gaussian noise secara bertahap"""
        img = torch.randn(shape, device=device)
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            predicted_noise = self.forward(img, t, text_cond)

            beta = self.betas[i]
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]

            # Mean computation
            mean = (1.0 / torch.sqrt(alpha)) * (img - (beta / torch.sqrt(1.0 - alpha_cumprod)) * predicted_noise)
            if i > 0:
                noise = torch.randn_like(img)
                variance = torch.sqrt(beta)
                img = mean + variance * noise
            else:
                img = mean

        # Clip ke rentang [-1, 1] lalu normalisasi ke [0, 1]
        img = torch.clamp(img, -1.0, 1.0)
        img = (img + 1.0) / 2.0
        return img
