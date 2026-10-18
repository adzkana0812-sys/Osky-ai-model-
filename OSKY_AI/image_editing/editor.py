"""
OSKY AI - Neural Image Editor & Inpainter
Model neural inpainting mandiri dari nol.
Menerima citra RGB [B, 3, H, W] + Binary Mask [B, 1, H, W] = [B, 4, H, W]
dan merekonstruksi pixel yang hilang atau dihapus secara mulus.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

class OskyInpaintingNet(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 3, hidden_dim: int = 32):
        super().__init__()
        # Encoder Gated Convolutions
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Decoder Convolutions
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Mask out area yang akan diinpaint (jadikan 0)
        masked_img = img * (1.0 - mask)
        # Gabungkan citra ber-mask dan channel mask: [B, 4, H, W]
        x = torch.cat([masked_img, mask], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d1 = self.dec1(b)
        out = self.out_conv(d1)

        # Kompositkan: pertahankan area asli yang tidak di-mask, isi area di-mask dengan prediksi
        reconstructed = img * (1.0 - mask) + out * mask
        return reconstructed

class OskyImageEditor:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.inpainting_net = OskyInpaintingNet().to(device)

    def inpaint(self, pil_img: Image.Image, mask_tensor: torch.Tensor) -> Image.Image:
        """Menjalankan neural inpainting pada area mask"""
        self.inpainting_net.eval()
        w, h = pil_img.size
        # Resize sementara ke 64x64 untuk proses neural
        resized = pil_img.resize((64, 64), Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        if mask_tensor.shape[-2:] != (64, 64):
            mask_tensor = F.interpolate(mask_tensor, size=(64, 64), mode="nearest").to(self.device)

        with torch.no_grad():
            out_tensor = self.inpainting_net(img_tensor, mask_tensor)
            out_arr = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            out_arr = np.clip(out_arr * 255.0, 0, 255).astype(np.uint8)
            result_pil = Image.fromarray(out_arr).resize((w, h), Image.Resampling.BILINEAR)
            return result_pil
