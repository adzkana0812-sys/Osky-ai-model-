"""
OSKY AI - Handwriting & OCR Dataset Pipeline
Memuat dataset tulisan tangan/dokumen dalam format:
{"image": "path/ke/gambar.png", "text": "Halo OSKY"}
Mendukung normalisasi karakter Indonesia, angka, huruf, dan simbol matematika.
Jika file gambar belum ada, generator sintetis menggambar karakter secara akurat
sehingga pipeline pelatihan CRNN CTC dapat langsung dilatih.
"""

import os
import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw
import numpy as np

from OSKY_AI.paths import resolve_path

# Daftar karakter standar tulisan tangan & simbol matematika
DEFAULT_CHARACTERS = (
    " 0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ".,-+=/*()[]_{}:;!?%@#$&'\""
)

class HandwritingCharVocabulary:
    def __init__(self, characters: str = DEFAULT_CHARACTERS):
        self.blank_token = "<blank>"  # Token CTC blank (indeks 0)
        self.unk_token = "<unk>"

        self.chars = [self.blank_token, self.unk_token] + list(characters)
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text: str) -> List[int]:
        return [self.char2idx.get(ch, self.char2idx[self.unk_token]) for ch in text]

    def decode(self, indices: List[int], merge_repeated: bool = True) -> str:
        """Decode output CTC indices menjadi teks terbaca"""
        res = []
        prev_idx = -1
        blank_idx = self.char2idx[self.blank_token]
        unk_idx = self.char2idx[self.unk_token]

        for idx in indices:
            if idx == blank_idx:
                prev_idx = -1
                continue
            if merge_repeated and idx == prev_idx:
                continue

            char = self.idx2char.get(idx, "")
            if char != self.unk_token:
                res.append(char)
            prev_idx = idx

        return "".join(res)

def render_synthetic_handwriting(text: str, width: int = 128, height: int = 32) -> Image.Image:
    """Menggambar karakter teks hitam di atas canvas putih (255) untuk OCR."""
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)
    # Pusatkan teks secara perkiraan
    char_w = 12
    total_w = len(text) * char_w
    start_x = max(4, (width - total_w) // 2)
    start_y = max(4, (height - 18) // 2)
    draw.text((start_x, start_y), text, fill=0)
    return img

class OskyHandwritingDataset(Dataset):
    def __init__(
        self,
        data_path: str = "datasets/handwriting/sample.jsonl",
        vocab: Optional[HandwritingCharVocabulary] = None,
        img_height: int = 32,
        img_width: int = 128
    ):
        self.vocab = vocab or HandwritingCharVocabulary()
        self.img_height = img_height
        self.img_width = img_width
        self.samples: List[Dict[str, str]] = []

        resolved_path = resolve_path(data_path)
        if os.path.exists(resolved_path):
            with open(resolved_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            if "text" in item:
                                self.samples.append(item)
                        except Exception:
                            pass

        if not self.samples:
            # Sediakan sampel angka dan huruf default untuk melatih recognizer
            char_samples = [str(i) for i in range(10)] + [chr(c) for c in range(ord('A'), ord('Z') + 1)] + [chr(c) for c in range(ord('a'), ord('z') + 1)]
            for ch in char_samples:
                self.samples.append({
                    "image": f"synthetic_{ch}.png",
                    "text": ch
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        img_path = item.get("image", "")
        text = item["text"]

        resolved_img = resolve_path(img_path) if img_path else ""

        # Muat gambar atau buat gambar sintetis jika file belum ada
        if resolved_img and os.path.exists(resolved_img):
            pil_img = Image.open(resolved_img).convert("L")
        else:
            # Gambar teks sintetis nyata
            pil_img = render_synthetic_handwriting(text, width=self.img_width, height=self.img_height)

        # Resize ke dimensi target
        pil_img = pil_img.resize((self.img_width, self.img_height), Image.Resampling.BILINEAR)
        img_arr = np.array(pil_img, dtype=np.float32) / 255.0  # [0, 1]
        # Invert agar background gelap (0) dan tulisan terang (1): [1, H, W]
        img_arr = 1.0 - img_arr
        img_tensor = torch.from_numpy(img_arr).unsqueeze(0)  # [1, H, W]

        encoded_text = self.vocab.encode(text)
        return {
            "image": img_tensor,
            "text": text,
            "target": torch.tensor(encoded_text, dtype=torch.long),
            "target_length": len(encoded_text)
        }
