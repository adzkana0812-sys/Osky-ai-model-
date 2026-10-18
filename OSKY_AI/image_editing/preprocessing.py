"""
OSKY AI - Image Editing Preprocessing & Layout Analyzer
Menganalisis komposisi visual gambar:
- Alignment (penjajaran sumbu X & Y)
- Spacing & Margins (jarak tepi)
- Symmetry & Proportions (keseimbangan simetri kiri-kanan)
- Object Center of Mass & Bounding Boxes
- Masking generator untuk inpainting/penghapusan objek
"""

import numpy as np
from PIL import Image
import torch
from typing import Dict, Any, Tuple, Optional

def analyze_image_layout(pil_img: Image.Image) -> Dict[str, Any]:
    """
    Melakukan inspeksi struktural layout dan kerapian gambar secara kuantitatif.
    """
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32)
    h, w, c = arr.shape

    # Konversi ke grayscale untuk deteksi intensitas dan massa objek
    gray = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
    # Thresholding sederhana untuk segmentasi area aktif vs latar belakang
    bg_val = np.median(gray)
    diff = np.abs(gray - bg_val)
    mask = diff > 20.0

    if np.sum(mask) > 0:
        y_indices, x_indices = np.where(mask)
        min_x, max_x = int(np.min(x_indices)), int(np.max(x_indices))
        min_y, max_y = int(np.min(y_indices)), int(np.max(y_indices))
        
        # Hitung margin
        left_margin = min_x
        right_margin = w - max_x
        top_margin = min_y
        bottom_margin = h - max_y

        # Center of mass
        center_x = float(np.mean(x_indices)) / w
        center_y = float(np.mean(y_indices)) / h

        # Keseimbangan horizontal & vertikal (0.0 = sempurna di tengah)
        horizontal_imbalance = abs(center_x - 0.5)
        vertical_imbalance = abs(center_y - 0.5)

        # Simetri kiri vs kanan
        left_half = gray[:, :w//2]
        right_half_flipped = np.fliplr(gray[:, w//2:w//2 + left_half.shape[1]])
        symmetry_score = 1.0 - (float(np.mean(np.abs(left_half - right_half_flipped))) / 255.0)
    else:
        min_x, max_x, min_y, max_y = 0, w, 0, h
        left_margin = right_margin = top_margin = bottom_margin = 0
        center_x, center_y = 0.5, 0.5
        horizontal_imbalance = vertical_imbalance = 0.0
        symmetry_score = 1.0

    return {
        "width": w,
        "height": h,
        "aspect_ratio": round(w / max(1, h), 3),
        "margins": {
            "top": top_margin,
            "bottom": bottom_margin,
            "left": left_margin,
            "right": right_margin
        },
        "center_of_mass": {"x": round(center_x, 3), "y": round(center_y, 3)},
        "horizontal_imbalance": round(horizontal_imbalance, 3),
        "vertical_imbalance": round(vertical_imbalance, 3),
        "symmetry_score": round(max(0.0, min(1.0, symmetry_score)), 3),
        "bounding_box": [min_x, min_y, max_x, max_y]
    }

def create_center_mask(image_size: int, mask_ratio: float = 0.4) -> torch.Tensor:
    """Membuat binary mask untuk inpainting objek di tengah gambar"""
    mask = torch.zeros((1, 1, image_size, image_size), dtype=torch.float32)
    start = int(image_size * (1.0 - mask_ratio) / 2)
    end = int(image_size * (1.0 + mask_ratio) / 2)
    mask[:, :, start:end, start:end] = 1.0
    return mask
