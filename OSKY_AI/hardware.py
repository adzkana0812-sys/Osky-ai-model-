"""
OSKY AI - Hardware Detection & Resource Profiles
Mendeteksi kapabilitas CPU, RAM, CUDA GPU, dan VRAM secara presisi.
Menyediakan profil hardware-aware (CPU_TEST, GPU_SMALL, GPU_MEDIUM, GPU_LARGE)
untuk memastikan training vision & image generation tidak melebihi memori sistem.
"""

import os
import torch
from typing import Dict, Any


def get_hardware_info() -> Dict[str, Any]:
    info = {
        "device_type": "cpu",
        "device_name": "CPU",
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": None,
        "vram_total_gb": 0.0,
        "vram_free_gb": 0.0,
        "recommended_dtype": torch.float32,
        "recommended_profile": "CPU_TEST"
    }

    # RAM detection via os/sys
    try:
        total_ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
        info["system_ram_gb"] = round(total_ram, 1)
    except Exception:
        info["system_ram_gb"] = 4.0

    if torch.cuda.is_available():
        info["device_type"] = "cuda"
        info["device_name"] = torch.cuda.get_device_name(0)
        info["gpu_name"] = info["device_name"]
        
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        info["vram_total_gb"] = round(vram_bytes / (1024**3), 2)
        
        # Determine profile based on VRAM
        if info["vram_total_gb"] >= 14.0:
            info["recommended_profile"] = "GPU_LARGE"
            info["recommended_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif info["vram_total_gb"] >= 7.0:
            info["recommended_profile"] = "GPU_MEDIUM"
            info["recommended_dtype"] = torch.float16
        else:
            info["recommended_profile"] = "GPU_SMALL"
            info["recommended_dtype"] = torch.float16
    else:
        info["recommended_profile"] = "CPU_TEST"
        info["recommended_dtype"] = torch.float32

    return info

HARDWARE_PROFILES = {
    "CPU_TEST": {
        "vision_image_size": 64,
        "vision_patch_size": 8,
        "vision_embed_dim": 128,
        "vision_layers": 2,
        "vision_heads": 2,
        "gen_image_size": 32,
        "gen_latent_dim": 64,
        "batch_size": 4,
        "description": "Profil sangat ringan untuk CPU, laptop biasa, dan testing fungsionalitas."
    },
    "GPU_SMALL": {
        "vision_image_size": 128,
        "vision_patch_size": 16,
        "vision_embed_dim": 256,
        "vision_layers": 4,
        "vision_heads": 4,
        "gen_image_size": 64,
        "gen_latent_dim": 128,
        "batch_size": 16,
        "description": "Profil optimal untuk GPU 4GB VRAM (GTX 1650, RTX 3050 Laptop)."
    },
    "GPU_MEDIUM": {
        "vision_image_size": 224,
        "vision_patch_size": 16,
        "vision_embed_dim": 512,
        "vision_layers": 8,
        "vision_heads": 8,
        "gen_image_size": 128,
        "gen_latent_dim": 256,
        "batch_size": 32,
        "description": "Profil seimbang untuk GPU 8GB-12GB VRAM (RTX 3060, RTX 4060, T4)."
    },
    "GPU_LARGE": {
        "vision_image_size": 256,
        "vision_patch_size": 16,
        "vision_embed_dim": 768,
        "vision_layers": 12,
        "vision_heads": 12,
        "gen_image_size": 256,
        "gen_latent_dim": 512,
        "batch_size": 64,
        "description": "Profil resolusi penuh untuk GPU 16GB-24GB+ VRAM (RTX 3090, 4090, A10G, V100)."
    }
}
