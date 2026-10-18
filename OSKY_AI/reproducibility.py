"""
OSKY AI - Reproducibility & Random Seed Controller
Mengontrol keacakan pada Python random, Numpy, dan PyTorch (CPU & CUDA)
untuk memastikan eksperimen training dapat direproduksi dengan konsisten.
"""

import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42, deterministic_cuda: bool = False):
    """
    Mengatur seed untuk semua generator angka acak.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
