"""
OSKY AI - Generative Autoencoder (Conv-VAE from Scratch)
Mengimplementasikan Variational Autoencoder berbasis Convolutional untuk penelitian generatif lokal.
Terdiri dari Encoder, Latent Bottleneck dengan Reparameterization Trick, dan Decoder Transpose.
100% Random Initialization tanpa bobot pretrained luar.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class OskyConvVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 64,
        image_size: int = 32,
        base_channels: int = 32
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.base_channels = base_channels

        # 1. Convolutional Encoder: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),  # [B, 32, 16, 16]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),  # [B, 64, 8, 8]
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1), # [B, 128, 4, 4]
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.flatten_dim = (base_channels * 4) * 4 * 4  # 128 * 16 = 2048

        # 2. Latent Projection (Mean & Log-Variance)
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # 3. Decoder Projection: latent_dim -> flatten_dim
        self.fc_decoder = nn.Linear(latent_dim, self.flatten_dim)

        # 4. Transposed Convolution Decoder: 4x4 -> 8x8 -> 16x16 -> 32x32
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),     # 16x16
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(base_channels, in_channels, kernel_size=4, stride=2, padding=1),           # 32x32
            nn.Sigmoid()  # Output direntang [0, 1]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        feat_flat = feat.view(feat.size(0), -1)
        mu = self.fc_mu(feat_flat)
        logvar = self.fc_logvar(feat_flat)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        feat_flat = self.fc_decoder(z)
        feat = feat_flat.view(feat_flat.size(0), self.base_channels * 4, 4, 4)
        return self.decoder(feat)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)

        # 1. Reconstruction Loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        # 2. KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        total_loss = recon_loss + 0.001 * kl_loss

        return {
            "recon_x": recon_x,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss
        }
