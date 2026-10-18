"""
OSKY AI - Path Resolution Utility
Menyediakan path resolution deterministik berbasis __file__ sehingga project
dapat dijalankan dari current working directory (CWD) manapun dengan aman.
"""

import os
from typing import Optional

# Direktori root OSKY_AI (lokasi folder OSKY_AI itu sendiri)
OSKY_DIR = os.path.dirname(os.path.abspath(__file__))

# Direktori root project (parent dari OSKY_AI)
PROJECT_ROOT = os.path.dirname(OSKY_DIR)

def resolve_path(relative_or_absolute_path: str, base_dir: Optional[str] = None) -> str:
    """
    Menyelesaikan path relatif terhadap base_dir, OSKY_DIR, atau PROJECT_ROOT.
    Jika path sudah absolut, dikembalikan langsung.
    """
    if not relative_or_absolute_path:
        return ""

    if os.path.isabs(relative_or_absolute_path):
        return os.path.normpath(relative_or_absolute_path)

    # 1. Cek terhadap base_dir jika diberikan
    if base_dir:
        candidate = os.path.join(base_dir, relative_or_absolute_path)
        if os.path.exists(candidate):
            return os.path.normpath(candidate)

    # 2. Cek apakah ada di CWD saat ini
    if os.path.exists(relative_or_absolute_path):
        return os.path.abspath(relative_or_absolute_path)

    # 3. Cek di dalam OSKY_DIR
    candidate_osky = os.path.join(OSKY_DIR, relative_or_absolute_path)
    if os.path.exists(candidate_osky):
        return os.path.normpath(candidate_osky)

    # 4. Cek di dalam PROJECT_ROOT
    candidate_root = os.path.join(PROJECT_ROOT, relative_or_absolute_path)
    if os.path.exists(candidate_root):
        return os.path.normpath(candidate_root)

    # Default fallback: gabungkan dengan OSKY_DIR jika path dimulai dengan modul internal
    first_part = relative_or_absolute_path.split(os.sep)[0]
    if first_part in ("checkpoints", "configs", "data", "tokenizer", "neural_network", "datasets"):
        return os.path.normpath(os.path.join(OSKY_DIR, relative_or_absolute_path))

    return os.path.normpath(os.path.join(PROJECT_ROOT, relative_or_absolute_path))
