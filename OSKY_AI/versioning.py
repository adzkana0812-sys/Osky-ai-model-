"""
OSKY AI - Model & Dataset Versioning & Continual Learning Manager
Mengelola siklus hidup pembaruan model secara aman:
1. Versioning Model: osky_model_v1, osky_model_v2, osky_model_v3
2. Versioning Dataset: dataset_v1, dataset_v2, dataset_v3
3. Tiga Pilar Pembelajaran OSKY AI:
   - Training (Update Bobot Melalui Dataset Terverifikasi)
   - Context (Memori Kerja Inferensi Sementara)
   - Long-term Knowledge (Indeks Retrieval Dokumen Lokal)
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from OSKY_AI.paths import resolve_path

VERSION_REGISTRY_FILE = "checkpoints/version_registry.json"

class ModelVersionManager:
    def __init__(self, registry_file: str = VERSION_REGISTRY_FILE):
        self.registry_file = resolve_path(registry_file)
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "current_active_version": "osky_model_v1",
            "models": {
                "osky_model_v1": {
                    "version": "osky_model_v1",
                    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "architecture": "Pre-LN Transformer with RoPE & SwiGLU",
                    "dataset_version": "dataset_v1",
                    "eval_perplexity": 5.4,
                    "checkpoint_path": "checkpoints/latest.pt",
                    "description": "Model inisialisasi awal from scratch."
                }
            },
            "datasets": {
                "dataset_v1": {
                    "version": "dataset_v1",
                    "sample_count": 120,
                    "token_count": 8500,
                    "categories": ["percakapan_umum", "identitas_osky", "pengetahuan_dasar"],
                    "cleaning_version": "clean_v1.0"
                }
            }
        }

    def _save_registry(self):
        os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2)

    def register_new_model_version(
        self,
        version_name: str,
        checkpoint_path: str,
        dataset_version: str,
        eval_metrics: Dict[str, float],
        description: str
    ):
        """Mendaftarkan versi model baru yang telah divalidasi"""
        self.registry["models"][version_name] = {
            "version": version_name,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "dataset_version": dataset_version,
            "metrics": eval_metrics,
            "checkpoint_path": checkpoint_path,
            "description": description
        }
        self.registry["current_active_version"] = version_name
        self._save_registry()

    def register_dataset_version(
        self,
        dataset_name: str,
        sample_count: int,
        token_count: int,
        categories: List[str]
    ):
        """Mendaftarkan versi dataset baru yang telah dibersihkan dan difilter"""
        self.registry["datasets"][dataset_name] = {
            "version": dataset_name,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "sample_count": sample_count,
            "token_count": token_count,
            "categories": categories
        }
        self._save_registry()

    def get_summary(self) -> Dict[str, Any]:
        return {
            "active_version": self.registry.get("current_active_version"),
            "total_model_versions": len(self.registry.get("models", {})),
            "total_dataset_versions": len(self.registry.get("datasets", {})),
            "all_models": list(self.registry.get("models", {}).keys())
        }
