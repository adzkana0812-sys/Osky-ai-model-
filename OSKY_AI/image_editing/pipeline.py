"""
OSKY AI - Instruction-based Image Editing Pipeline
Menerima:
1. Citra Masukan (PIL Image)
2. Instruksi Pengguna (misal: "Rapikan gambar", "Hapus objek di tengah", "Seimbangkan margin")

Tahapan Pipeline:
Image -> Vision/Layout Analysis -> Object Positions & Imbalance -> Editing Plan -> Neural Execution -> Clean Result
"""

from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import torch
from typing import Dict, Any, Tuple

from OSKY_AI.image_editing.preprocessing import analyze_image_layout, create_center_mask
from OSKY_AI.image_editing.editor import OskyImageEditor

class OskyEditingPipeline:
    def __init__(self, device: str = "cpu"):
        self.editor = OskyImageEditor(device=device)

    def execute_instruction(
        self,
        image: Image.Image,
        instruction: str
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Menganalisis gambar dan mengeksekusi rencana perbaikan sesuai instruksi.
        """
        inst_lower = instruction.lower()
        # 1. Analisis layout awal
        layout_before = analyze_image_layout(image)

        editing_plan = []
        result_img = image.copy()

        # 2. Logika Pemrosesan Berdasarkan Instruksi
        if any(w in inst_lower for w in ["rapikan", "tata", "seimbang", "center", "margin"]):
            editing_plan.append("Langkah 1: Deteksi ketidakseimbangan horizontal/vertikal.")
            # Koreksi margin dengan auto-centering
            bbox = layout_before["bounding_box"]
            w, h = image.size
            cropped = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

            # Buat kanvas baru persegi berlatar belakang bersih dengan objek di tengah
            max_side = max(cropped.size) + 40
            bg_color = (255, 255, 255)
            centered_canvas = Image.new("RGB", (max_side, max_side), bg_color)
            offset_x = (max_side - cropped.size[0]) // 2
            offset_y = (max_side - cropped.size[1]) // 2
            centered_canvas.paste(cropped, (offset_x, offset_y))
            result_img = centered_canvas
            editing_plan.append("Langkah 2: Menyeimbangkan margin kiri-kanan & atas-bawah (Auto-Centering).")

        if any(w in inst_lower for w in ["hapus", "hilangkan", "inpaint"]):
            editing_plan.append("Langkah: Membuat mask koordinat dan menjalankan neural inpainting.")
            mask = create_center_mask(64, mask_ratio=0.35)
            result_img = self.editor.inpaint(result_img, mask)

        if any(w in inst_lower for w in ["bersih", "kontras", "terang"]):
            editing_plan.append("Langkah: Meningkatkan kejernihan kontras dan ketajaman pixel.")
            enhancer = ImageEnhance.Contrast(result_img)
            result_img = enhancer.enhance(1.15)
            sharpener = ImageEnhance.Sharpness(result_img)
            result_img = sharpener.enhance(1.1)

        # Analisis layout setelah perbaikan
        layout_after = analyze_image_layout(result_img)

        report = {
            "instruction": instruction,
            "editing_plan": editing_plan,
            "layout_before": layout_before,
            "layout_after": layout_after,
            "symmetry_improvement": round(layout_after["symmetry_score"] - layout_before["symmetry_score"], 3)
        }

        return result_img, report
