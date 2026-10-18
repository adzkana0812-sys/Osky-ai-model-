"""
OSKY AI - Full Training Engine
Mendukung:
1. Deteksi hardware otomatis (CUDA GPU / CPU, RAM, VRAM, Device, Dtype)
2. Forward pass, Cross-Entropy loss calculation, Backpropagation
3. AdamW optimizer dengan decoupled weight decay
4. Gradient clipping & learning rate schedule dengan warmup
5. Gradient accumulation & Automatic Mixed Precision (AMP) jika GPU mendukung
6. Validasi, Perplexity calculation, dan Test Generation berkala
7. Checkpoint otomatis & Resume capability
8. Pelacakan tokens processed & throughput (tokens/sec)
"""

import os
import time
import math
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader

from OSKY_AI.model import OskyLanguageModel
from OSKY_AI.tokenizer import OskyTokenizer
from .optimizer import configure_optimizers, CosineWarmupLRScheduler
from .checkpoint import CheckpointManager

def detect_hardware() -> Dict[str, Any]:
    """
    Mendeteksi hardware secara otomatis dan memeriksa RAM, VRAM, dan kapabilitas komputasi.
    """
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_available else "cpu")

    info = {
        "device": str(device),
        "cuda_available": cuda_available,
        "device_name": "CPU",
        "vram_gb": 0.0,
        "recommended_dtype": torch.float32,
        "amp_supported": False,
        "ram_gb": 0.0,
    }

    # Cek System RAM
    try:
        if hasattr(os, "sysconf"):
            mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            info["ram_gb"] = round(mem_bytes / (1024 ** 3), 2)
    except Exception:
        info["ram_gb"] = 0.0

    if cuda_available:
        info["device_name"] = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        info["vram_gb"] = round(vram_bytes / (1024 ** 3), 2)
        info["amp_supported"] = True
        # bfloat16 jika GPU mendukung Ampere atau lebih baru
        if torch.cuda.is_bf16_supported():
            info["recommended_dtype"] = torch.bfloat16
        else:
            info["recommended_dtype"] = torch.float16

    return info


class OskyTrainer:
    def __init__(
        self,
        model: OskyLanguageModel,
        tokenizer: OskyTokenizer,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: str = "checkpoints",
        resume_from: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_config = config.get("training", {})
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)

        # 1. Hardware setup
        self.hw_info = detect_hardware()
        self.device = torch.device(self.hw_info["device"])
        self.model.to(self.device)

        print("=" * 60)
        print("  OSKY AI - HARDWARE & TRAINING CONFIGURATION")
        print("=" * 60)
        print(f"Device: {self.hw_info['device']} ({self.hw_info['device_name']})")
        print(f"System RAM: {self.hw_info['ram_gb']} GB")
        if self.hw_info['cuda_available']:
            print(f"GPU VRAM: {self.hw_info['vram_gb']} GB")
            print(f"Mixed Precision: {self.hw_info['recommended_dtype']}")
        else:
            print("GPU VRAM: Tidak ada (Menggunakan CPU)")
            print("Mixed Precision: Disabled (FP32 pada CPU)")
        print(f"Model Parameters: {self.model.count_parameters()['human_readable']}")
        print(f"Memory Footprint (FP32): {self.model.count_parameters()['memory_fp32_mb']} MB")
        print("=" * 60)

        # 2. Hyperparameters
        self.lr = self.train_config.get("learning_rate", 5e-4)
        self.min_lr = self.train_config.get("min_learning_rate", 5e-5)
        self.warmup_steps = self.train_config.get("warmup_steps", 100)
        self.max_steps = self.train_config.get("max_steps", 2000)
        self.weight_decay = self.train_config.get("weight_decay", 0.01)
        self.max_grad_norm = self.train_config.get("max_grad_norm", 1.0)
        self.grad_accum_steps = self.train_config.get("gradient_accumulation_steps", 1)
        self.eval_interval = self.train_config.get("eval_interval", 100)
        self.eval_steps = self.train_config.get("eval_steps", 20)
        self.save_interval = self.train_config.get("save_interval", 250)

        # 3. Optimizer & Scheduler
        self.optimizer = configure_optimizers(self.model, self.lr, self.weight_decay)
        self.scheduler = CosineWarmupLRScheduler(
            self.optimizer,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            max_lr=self.lr,
            min_lr=self.min_lr,
        )

        # 4. Mixed Precision Scaler
        self.use_amp = self.hw_info["amp_supported"]
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # 5. Tracking variables
        self.global_step = 0
        self.epoch = 0
        self.total_tokens_processed = 0
        self.best_val_loss = float("inf")

        # 6. Resume jika diberikan
        if resume_from:
            self.resume(resume_from)

    def resume(self, checkpoint_path: str):
        state = self.checkpoint_manager.load(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler,
            device=str(self.device),
            expected_tokenizer=self.tokenizer,
        )
        self.global_step = state.get("global_step", state.get("step", 0))
        self.epoch = state.get("epoch", 0)
        self.best_val_loss = state.get("best_val_loss", state.get("val_loss", float("inf")))
        print(f"[Trainer] Melanjutkan pelatihan dari step {self.global_step}, epoch {self.epoch} (Best Val Loss: {self.best_val_loss:.4f})")

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Menjalankan evaluasi pada validation split."""
        if not self.val_loader:
            return {"val_loss": 0.0, "perplexity": 1.0}

        self.model.eval()
        total_loss = 0.0
        steps = 0

        for i, (input_ids, labels, masks) in enumerate(self.val_loader):
            if i >= self.eval_steps:
                break
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            masks = masks.to(self.device)

            loss, _ = self.model(input_ids, labels=labels, padding_mask=masks)
            if loss is not None and not torch.isnan(loss):
                total_loss += loss.item()
                steps += 1

        self.model.train()
        avg_loss = total_loss / max(1, steps)
        # Perplexity = exp(loss)
        perplexity = math.exp(min(20.0, avg_loss))
        return {"val_loss": avg_loss, "perplexity": perplexity}

    @torch.no_grad()
    def test_sample_generation(self, prompt: str = "Halo") -> str:
        """Menghasilkan teks pengujian sederhana menggunakan weights model saat ini."""
        self.model.eval()
        input_ids = self.tokenizer.encode(f"<user>{prompt}<assistant>", add_bos=True)
        x = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        generated_ids = list(input_ids)
        for _ in range(30):
            seq_len = x.shape[1]
            if seq_len >= self.model.max_seq_len:
                break
            _, logits = self.model(x)
            next_token_logits = logits[:, -1, :]
            # Greedy sampling
            next_token = torch.argmax(next_token_logits, dim=-1).item()
            generated_ids.append(next_token)
            if next_token == self.tokenizer.eos_token_id:
                break
            x = torch.cat([x, torch.tensor([[next_token]], device=self.device)], dim=1)

        self.model.train()
        return self.tokenizer.decode(generated_ids)

    def train(self):
        """Memulai loop pelatihan neural network dari nol."""
        print(f"\n[Trainer] Memulai pelatihan OSKY AI dari Step {self.global_step} ke {self.max_steps}...")
        self.model.train()
        self.optimizer.zero_grad()

        start_time = time.time()
        step_start_time = time.time()
        accumulated_loss = 0.0

        while self.global_step < self.max_steps:
            self.epoch += 1

            for batch_idx, (input_ids, labels, masks) in enumerate(self.train_loader):
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                masks = masks.to(self.device)

                # Hitung token untuk tracking throughput
                valid_tokens = (labels != -100).sum().item()
                self.total_tokens_processed += valid_tokens

                # Forward pass dengan Automatic Mixed Precision jika ada GPU
                with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.hw_info["recommended_dtype"]):
                    loss, _ = self.model(input_ids, labels=labels, padding_mask=masks)
                    # Skalakan loss untuk gradient accumulation
                    scaled_loss = loss / self.grad_accum_steps

                # Backpropagation
                self.scaler.scale(scaled_loss).backward()
                accumulated_loss += loss.item()

                # Step Optimizer setelah gradient accumulation terpenuhi
                if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    # Unscale gradien sebelum clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    # Optimizer & Scheduler Step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    current_lr = self.scheduler.step()

                    self.global_step += 1
                    avg_loss = accumulated_loss / self.grad_accum_steps
                    accumulated_loss = 0.0

                    # Logging berkala
                    if self.global_step % 10 == 0 or self.global_step == 1:
                        elapsed = time.time() - step_start_time
                        step_start_time = time.time()
                        tok_per_sec = self.total_tokens_processed / max(0.1, time.time() - start_time)

                        print(
                            f"[Step {self.global_step:5d}/{self.max_steps:5d}] "
                            f"Loss: {avg_loss:6.4f} | "
                            f"LR: {current_lr:.2e} | "
                            f"Tok/s: {tok_per_sec:7.1f} | "
                            f"Tokens: {self.total_tokens_processed:8d} | "
                            f"Epoch: {self.epoch}"
                        )

                    # Evaluasi berkala
                    if self.global_step % self.eval_interval == 0:
                        eval_res = self.evaluate()
                        val_loss = eval_res["val_loss"]
                        ppl = eval_res["perplexity"]
                        is_best = val_loss < self.best_val_loss
                        if is_best:
                            self.best_val_loss = val_loss

                        print("\n" + "-" * 50)
                        print(f"[EVALUATION Step {self.global_step}] Val Loss: {val_loss:.4f} | Perplexity: {ppl:.2f}")

                        # Uji generasi teks setelah evaluasi
                        sample_output = self.test_sample_generation("Halo")
                        print(f"[Test Sample Output]:\n{sample_output}")
                        print("-" * 50 + "\n")

                        # Simpan checkpoint jika ada peningkatan
                        if is_best:
                            self.checkpoint_manager.save(
                                model=self.model,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                epoch=self.epoch,
                                step=self.global_step,
                                config=self.config,
                                train_loss=avg_loss,
                                val_loss=val_loss,
                                best_val_loss=self.best_val_loss,
                                tokenizer=self.tokenizer,
                                is_best=True
                            )

                    # Simpan periodic checkpoint
                    if self.global_step % self.save_interval == 0:
                        self.checkpoint_manager.save(
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            epoch=self.epoch,
                            step=self.global_step,
                            config=self.config,
                            train_loss=avg_loss,
                            val_loss=None,
                            best_val_loss=self.best_val_loss,
                            tokenizer=self.tokenizer
                        )

                    if self.global_step >= self.max_steps:
                        break

        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"[Trainer] Pelatihan selesai!")
        print(f"Total Step: {self.global_step} | Total Tokens: {self.total_tokens_processed}")
        print(f"Total Waktu: {total_time:.1f} detik ({total_time / 60:.2f} menit)")
        print("=" * 60)

        # Simpan final checkpoint
        final_eval = self.evaluate()
        self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epoch,
            step=self.global_step,
            config=self.config,
            train_loss=avg_loss if 'avg_loss' in locals() else None,
            val_loss=final_eval["val_loss"],
            best_val_loss=self.best_val_loss,
            tokenizer=self.tokenizer,
            filename="osky_final.pt"
        )
