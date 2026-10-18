"""
OSKY AI - Checkpoint Manager
Menyimpan dan memuat:
1. Model weights (state_dict)
2. Optimizer state (optimizer_state_dict)
3. LR Scheduler state (scheduler_state_dict)
4. Epoch dan Training Step (global_step)
5. Train Loss dan Validation Loss secara terpisah (TIDAK PERNAH mencampuradukkan keduanya)
6. Best Validation Loss
7. Konfigurasi model lengkap
8. Versi Tokenizer, Vocab Size, dan Vocab Hash
Mendukung resume pelatihan dan best-checkpoint tracking.
"""

import os
import time
import torch
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from OSKY_AI.paths import resolve_path

class CheckpointManager:
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = resolve_path(checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        step: int,
        config: Dict[str, Any],
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        best_val_loss: Optional[float] = None,
        tokenizer: Optional[Any] = None,
        filename: Optional[str] = None,
        is_best: bool = False,
    ) -> str:
        """Menyimpan checkpoint lengkap ke disk dengan metadata akurat."""
        if filename is None:
            filename = f"osky_step_{step}.pt"

        save_path = os.path.join(self.checkpoint_dir, filename)

        # Metadata tokenizer jika tersedia
        vocab_hash = None
        vocab_size = config.get("vocab_size", None)
        tokenizer_version = "1.0"
        if tokenizer is not None:
            if hasattr(tokenizer, "get_vocab_hash"):
                vocab_hash = tokenizer.get_vocab_hash()
            if hasattr(tokenizer, "vocab_size"):
                vocab_size = tokenizer.vocab_size

        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
            "epoch": epoch,
            "step": step,
            "global_step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "config": config,
            "tokenizer_version": tokenizer_version,
            "vocab_hash": vocab_hash,
            "vocab_size": vocab_size,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

        torch.save(checkpoint_data, save_path)
        loss_info = []
        if train_loss is not None:
            loss_info.append(f"Train Loss: {train_loss:.4f}")
        if val_loss is not None:
            loss_info.append(f"Val Loss: {val_loss:.4f}")
        loss_str = ", ".join(loss_info) if loss_info else "Loss: N/A"

        print(f"[Checkpoint] Tersimpan di: {save_path} (Step {step}, {loss_str})")

        # Copy/save ke latest.pt
        latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
        torch.save(checkpoint_data, latest_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pt")
            torch.save(checkpoint_data, best_path)
            print(f"[Checkpoint] Model terbaik baru disimpan di: {best_path}")

        return save_path

    def load(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu",
        expected_tokenizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Memuat checkpoint untuk resume training atau evaluasi."""
        resolved_path = resolve_path(checkpoint_path, self.checkpoint_dir)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"File checkpoint tidak ditemukan: {checkpoint_path} (resolved: {resolved_path})")

        print(f"[Checkpoint] Memuat checkpoint dari: {resolved_path}...")
        checkpoint = torch.load(resolved_path, map_location=device)

        # Verifikasi vocab hash jika tokenizer diberikan
        if expected_tokenizer is not None and hasattr(expected_tokenizer, "get_vocab_hash"):
            chk_hash = checkpoint.get("vocab_hash")
            current_hash = expected_tokenizer.get_vocab_hash()
            if chk_hash is not None and chk_hash != current_hash:
                print(f"[WARNING Checkpoint] Peringatan mismatch vocabulary hash:")
                print(f"  Checkpoint vocab_hash: {chk_hash}")
                print(f"  Current vocab_hash   : {current_hash}")

        # Muat bobot model
        model.load_state_dict(checkpoint["model_state_dict"])
        print("[Checkpoint] Bobot neural network berhasil dimuat.")

        # Muat optimizer jika diminta (mode resume training)
        if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("[Checkpoint] State optimizer berhasil dipulihkan.")

        # Muat scheduler jika diminta
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            if hasattr(scheduler, "load_state_dict"):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                print("[Checkpoint] State scheduler berhasil dipulihkan.")

        return {
            "epoch": checkpoint.get("epoch", 0),
            "step": checkpoint.get("step", checkpoint.get("global_step", 0)),
            "global_step": checkpoint.get("global_step", checkpoint.get("step", 0)),
            "train_loss": checkpoint.get("train_loss", None),
            "val_loss": checkpoint.get("val_loss", None),
            "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
            "config": checkpoint.get("config", {}),
            "vocab_hash": checkpoint.get("vocab_hash", None),
            "vocab_size": checkpoint.get("vocab_size", None),
            "timestamp": checkpoint.get("timestamp", ""),
        }
