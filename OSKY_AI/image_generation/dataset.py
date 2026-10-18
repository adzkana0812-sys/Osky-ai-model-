"""
OSKY AI - Image Generation Dataset Loader & Synthetic Generator
Memuat pasangan citra dan prompt deskripsi teks:
{"image": "datasets/image_generation/img_001.png", "prompt": "lingkaran merah di tengah"}
Jika citra belum ada di disk, generator sintetis membuat gambar geometris
yang 100% konsisten dengan isi prompt (warna dan bentuk), bukan canvas kosong!
"""

import os
import json
import re
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw
import numpy as np

from OSKY_AI.paths import resolve_path

COLOR_MAP = {
    "merah": (230, 30, 30),
    "red": (230, 30, 30),
    "biru": (30, 80, 230),
    "blue": (30, 80, 230),
    "hijau": (30, 200, 50),
    "green": (30, 200, 50),
    "kuning": (240, 220, 30),
    "yellow": (240, 220, 30),
    "putih": (240, 240, 240),
    "white": (240, 240, 240),
    "hitam": (20, 20, 20),
    "black": (20, 20, 20),
    "oranye": (240, 130, 30),
    "orange": (240, 130, 30),
    "ungu": (160, 40, 200),
    "purple": (160, 40, 200),
}

def generate_synthetic_image_for_prompt(prompt: str, size: int = 32) -> Image.Image:
    """
    Menghasilkan citra sintetis PIL yang sesuai dengan kata kunci warna dan bentuk dalam prompt.
    Menjamin pasangan (Prompt <-> Image) konsisten secara semantik untuk pelatihan difusi/VAE.
    """
    p_lower = prompt.lower()

    # Tentukan warna utama
    fill_color = (220, 40, 40)  # default merah
    for color_name, rgb in COLOR_MAP.items():
        if color_name in p_lower:
            fill_color = rgb
            break

    # Background color berlawanan kontras
    bg_color = (15, 15, 20) if fill_color != (20, 20, 20) else (230, 230, 230)

    img = Image.new("RGB", (size, size), color=bg_color)
    draw = ImageDraw.Draw(img)

    margin = max(2, int(size * 0.15))
    x0, y0 = margin, margin
    x1, y1 = size - margin, size - margin

    # Tentukan bentuk geometris
    if any(k in p_lower for k in ["lingkaran", "circle", "round", "oval"]):
        draw.ellipse([x0, y0, x1, y1], fill=fill_color)
    elif any(k in p_lower for k in ["segitiga", "triangle"]):
        points = [(size // 2, y0), (x0, y1), (x1, y1)]
        draw.polygon(points, fill=fill_color)
    elif any(k in p_lower for k in ["silang", "cross", "tanda tambah", "plus"]):
        thickness = max(2, size // 8)
        cx, cy = size // 2, size // 2
        # Garis horizontal
        draw.rectangle([x0, cy - thickness // 2, x1, cy + thickness // 2], fill=fill_color)
        # Garis vertikal
        draw.rectangle([cx - thickness // 2, y0, cx + thickness // 2, y1], fill=fill_color)
    else:
        # Default: Persegi / rectangle
        draw.rectangle([x0, y0, x1, y1], fill=fill_color)

    return img

class OskyImageGenDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        image_size: int = 32,
        channels: int = 3
    ):
        self.image_size = image_size
        self.channels = channels
        self.samples: List[Dict[str, str]] = []

        resolved_path = resolve_path(data_path)
        if os.path.exists(resolved_path):
            with open(resolved_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            if "prompt" in item:
                                self.samples.append(item)
                        except Exception:
                            pass

        if not self.samples:
            # Sediakan data sampel sintetis geometris bawaan
            default_prompts = [
                "lingkaran merah di tengah",
                "persegi biru terang",
                "segitiga hijau padat",
                "lingkaran kuning bersinar",
                "persegi oranye menyala",
                "silang ungu kontras",
                "red circle centered",
                "blue square geometric",
                "green triangle pattern"
            ]
            for i, p in enumerate(default_prompts):
                self.samples.append({
                    "image": f"synthetic_{i}.png",
                    "prompt": p
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        img_path = item.get("image", "")
        prompt = item["prompt"]

        resolved_img = resolve_path(img_path) if img_path else ""

        if resolved_img and os.path.exists(resolved_img):
            pil_img = Image.open(resolved_img).convert("RGB")
        else:
            # Hasilkan gambar sintetis yang konsisten dengan prompt!
            pil_img = generate_synthetic_image_for_prompt(prompt, size=self.image_size)

        pil_img = pil_img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        arr = np.array(pil_img, dtype=np.float32) / 255.0  # [0, 1]
        # Normalisasi ke rentang [-1.0, 1.0] untuk diffusion
        arr = (arr * 2.0) - 1.0
        img_tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]

        return {
            "image": img_tensor,
            "prompt": prompt
        }
