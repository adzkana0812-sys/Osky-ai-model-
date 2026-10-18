"""
OSKY AI - Main Training Script
Melatih model neural network OSKY AI dari nol (100% scratch, random weights).
Contoh eksekusi:
python train.py --config configs/small.json
python train.py --config configs/small.json --resume checkpoints/latest.pt
"""

import os
import sys
import argparse
import json
import torch
from torch.utils.data import DataLoader

# Tambahkan direktori root OSKY_AI ke sys.path jika diperlukan
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from OSKY_AI.paths import resolve_path
from OSKY_AI.reproducibility import set_seed
from OSKY_AI.model import OskyLanguageModel
from OSKY_AI.tokenizer import OskyTokenizer
from OSKY_AI.training.dataset import OskyDatasetPipeline, OskyTorchDataset, collate_fn
from OSKY_AI.training.trainer import OskyTrainer

def main():
    parser = argparse.ArgumentParser(description="Pelatihan OSKY AI dari Nol")
    parser.add_argument("--config", type=str, default="configs/small.json", help="Path ke konfigurasi model")
    parser.add_argument("--data", type=str, default="data/sample_conversations.jsonl", help="Path ke dataset JSONL/TXT")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer/vocab.json", help="Path ke file vocabulary tokenizer")
    parser.add_argument("--resume", type=str, default=None, help="Path ke checkpoint untuk melanjutkan training")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Direktori penyimpanan checkpoint")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Rasio data validasi")
    parser.add_argument("--seed", type=int, default=42, help="Random seed untuk reproduksibilitas")
    parser.add_argument("--max_steps", type=int, default=None, help="Override jumlah langkah pelatihan maksimum")
    parser.add_argument("--batch_size", type=int, default=None, help="Override ukuran batch pelatihan")
    args = parser.parse_args()

    # 1. Set Random Seed untuk Reproduksibilitas Lengkap
    set_seed(args.seed)

    # Selesaikan semua path secara deterministik
    config_path = resolve_path(args.config)
    data_path = resolve_path(args.data)
    tokenizer_path = resolve_path(args.tokenizer_path)
    resume_path = resolve_path(args.resume) if args.resume else None
    checkpoint_dir = resolve_path(args.checkpoint_dir)

    print("=" * 65)
    print("  OSKY AI (Osky Artificial Intelligence)")
    print("  Sistem Pelatihan Neural Network Bahasa dari Nol (From Scratch)")
    print("=" * 65)

    # 2. Muat Konfigurasi
    if not os.path.exists(config_path):
        print(f"Error: File konfigurasi {config_path} tidak ditemukan!")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if args.max_steps is not None:
        config.setdefault("training", {})["max_steps"] = args.max_steps
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size

    # 3. Setup Tokenizer & Dataset
    tokenizer = OskyTokenizer()
    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
    else:
        print(f"[Setup] File tokenizer {tokenizer_path} belum ada. Membangun tokenizer dari dataset...")
        # Baca teks mentah dari file data untuk training tokenizer
        sample_texts = []
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        sample_texts.append(line)
        if not sample_texts:
            sample_texts = [
                "Halo! Saya OSKY AI, kecerdasan buatan yang dilatih dari nol.",
                "OSKY AI menggunakan arsitektur transformer modern dengan RoPE dan RMSNorm.",
                "Bagaimana saya bisa membantu Anda hari ini?",
                "Kecerdasan buatan mandiri tanpa ketergantungan API pihak ketiga.",
            ]
        target_vsize = config.get("vocab_size", 1024)
        tokenizer.train_from_corpus(sample_texts, target_vocab_size=target_vsize)
        tokenizer.save(tokenizer_path)

    # Update config vocab_size agar sinkron dengan tokenizer
    config["vocab_size"] = tokenizer.vocab_size

    # 4. Muat Dataset Percakapan / Teks
    pipeline = OskyDatasetPipeline(tokenizer=tokenizer, max_seq_len=config.get("max_seq_len", 512), seed=args.seed)
    records = pipeline.load_jsonl(data_path)
    if not records:
        print("[Setup] Mencoba memuat sebagai teks biasa...")
        records = pipeline.load_raw_txt(data_path)

    if not records:
        print("[Setup] Menyiapkan contoh percakapan internal untuk testing awal...")
        records = [
            type("Rec", (), {"prompt": "Halo", "response": "Halo! Saya OSKY AI, model bahasa buatan sendiri."})(),
            type("Rec", (), {"prompt": "Siapa kamu?", "response": "Saya OSKY AI, sebuah neural network yang dibangun dari nol."})(),
            type("Rec", (), {"prompt": "Apa itu transformer?", "response": "Transformer adalah arsitektur neural network berbasis mekanisme atensi."})(),
        ]

    train_records, val_records = pipeline.split_dataset(records, val_ratio=args.val_ratio)

    train_dataset = OskyTorchDataset(train_records, tokenizer=tokenizer, max_seq_len=config.get("max_seq_len", 512))
    val_dataset = OskyTorchDataset(val_records, tokenizer=tokenizer, max_seq_len=config.get("max_seq_len", 512))

    batch_size = config.get("training", {}).get("batch_size", 8)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 5. Inisialisasi Model Neural Network OSKY AI dari Random Weights
    print("\n[Setup] Menginisialisasi OSKY AI Neural Network dari nol (Random Weights)...")
    model = OskyLanguageModel(config)

    param_info = model.count_parameters()
    print(f"[Model Parameter] Total Parameter: {param_info['total_params']:,} ({param_info['human_readable']})")
    print(f"[Model Parameter] Trainable Parameter: {param_info['trainable_params']:,}")
    print(f"[Model Parameter] Non-Embedding Parameter: {param_info['non_embedding_params']:,}")
    print(f"[Memory Footprint] FP32: {param_info['memory_fp32_mb']} MB | FP16: {param_info['memory_fp16_mb']} MB")

    # 6. Inisialisasi Trainer & Jalankan Pelatihan
    trainer = OskyTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_path,
    )

    trainer.train()

if __name__ == "__main__":
    main()
