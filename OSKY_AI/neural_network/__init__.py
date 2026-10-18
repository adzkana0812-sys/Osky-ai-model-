"""
OSKY AI - Neural Network Architectures Registry
Menyatukan seluruh arsitektur jaringan saraf mandiri OSKY AI:
- language: Transformer Causal Decoder-Only
- vision: Vision Transformer (ViT)
- handwriting: CRNN + CTC Recognizer
- multimodal: Visual-Language Projector & Fusion Model
- image_generation: Conv-VAE & Score Diffusion UNet
- image_editing: Neural Inpainting Net & Layout Engine
"""

from OSKY_AI.neural_network.language import OskyLanguageModel
from OSKY_AI.neural_network.vision import OskyVisionTransformer
from OSKY_AI.neural_network.handwriting import OskyHandwritingRecognizer
from OSKY_AI.neural_network.multimodal import OskyMultimodalModel
from OSKY_AI.neural_network.image_generation import OskyConvVAE, OskyDiffusionUNet, OskyImageGenerator
from OSKY_AI.neural_network.image_editing import OskyInpaintingNet, OskyImageEditor, OskyEditingPipeline

__all__ = [
    "OskyLanguageModel",
    "OskyVisionTransformer",
    "OskyHandwritingRecognizer",
    "OskyMultimodalModel",
    "OskyConvVAE",
    "OskyDiffusionUNet",
    "OskyImageGenerator",
    "OskyInpaintingNet",
    "OskyImageEditor",
    "OskyEditingPipeline"
]
