"""
OSKY AI - Model Evaluation & Benchmark Script
Mengevaluasi checkpoint OSKY AI:
1. Training/Validation Loss
2. Perplexity (PPL)
3. Benchmark Generation pada berbagai prompt bahasa Indonesia
Contoh eksekusi:
python evaluate.py --checkpoint checkpoints/latest.pt
"""

import os
import sys
import argparse
import math
import time
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from OSKY_AI.paths import resolve_path
from OSKY_AI.model import OskyLanguageModel
from OSKY_AI.tokenizer import OskyTokenizer
from OSKY_AI.training.dataset import OskyDatasetPipeline, OskyTorchDataset, collate_fn
from OSKY_AI.inference import OskyInferenceEngine

def main():
    parser = argparse.ArgumentParser(description="Evaluasi Model OSKY AI")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt", help="Path ke checkpoint model")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/vocab.json", help="Path ke tokenizer")
    parser.add_argument("--data", type=str, default="data/sample_conversations.jsonl", help="Path ke data evaluasi")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    checkpoint_path = resolve_path(args.checkpoint)
    tokenizer_path = resolve_path(args.tokenizer)
    data_path = resolve_path(args.data)

    if not os.path.exists(checkpoint_path):
        print(f"Error: File checkpoint {checkpoint_path} tidak ditemukan!")
        sys.exit(1)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"  OSKY AI - EVALUASI MODEL & BENCHMARK")
    print("=" * 60)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Device     : {device}")

    # 1. Muat Checkpoint & Model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    step = checkpoint.get("step", checkpoint.get("global_step", 0))
    epoch = checkpoint.get("epoch", 0)

    model = OskyLanguageModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = OskyTokenizer(tokenizer_path)
    param_info = model.count_parameters()

    print(f"Langkah Training : Step {step} (Epoch {epoch})")
    print(f"Jumlah Parameter : {param_info['human_readable']} ({param_info['total_params']:,} params)")
    print(f"Waktu Checkpoint : {checkpoint.get('timestamp', 'N/A')}")
    print("=" * 60)

    # 2. Hitung Loss & Perplexity pada Dataset Evaluasi
    if os.path.exists(args.data):
        pipeline = OskyDatasetPipeline(tokenizer=tokenizer, max_seq_len=config.get("max_seq_len", 512))
        records = pipeline.load_jsonl(args.data)
        if records:
            dataset = OskyTorchDataset(records, tokenizer=tokenizer, max_seq_len=config.get("max_seq_len", 512))
            loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

            total_loss = 0.0
            total_tokens = 0
            steps = 0

            with torch.no_grad():
                for input_ids, labels, masks in loader:
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)
                    masks = masks.to(device)

                    loss, _ = model(input_ids, labels=labels, padding_mask=masks)
                    if loss is not None and not torch.isnan(loss):
                        total_loss += loss.item()
                        steps += 1
                        total_tokens += (labels != -100).sum().item()

            avg_loss = total_loss / max(1, steps)
            perplexity = math.exp(min(20.0, avg_loss))

            print("\n[HASIL METRIK LOSS & PERPLEXITY]")
            print(f"- Evaluated Batches : {steps}")
            print(f"- Tokens Evaluated  : {total_tokens:,}")
            print(f"- Average Loss      : {avg_loss:.4f}")
            print(f"- Perplexity (PPL)  : {perplexity:.4f}")
            print("  (Catatan: Semakin rendah nilai Perplexity, semakin baik pemodelan probabilitas bahasa model)")

    # 3. Uji Generasi Teks Sampel
    print("\n" + "=" * 60)
    print("  UJI GENERASI TEKS OSKY AI (SAMPLE BENCHMARKS)")
    print("=" * 60)

    engine = OskyInferenceEngine(args.checkpoint, args.tokenizer, device=device)
    test_prompts = [
        "Halo",
        "Siapa kamu?",
        "Jelaskan apa itu OSKY AI.",
        "Apa fungsi transformer dalam AI?",
    ]

    for prompt in test_prompts:
        start_t = time.time()
        output = engine.generate(prompt, max_new_tokens=60, temperature=0.7, top_p=0.9)
        elapsed = time.time() - start_t
        print(f"\n[User]   : {prompt}")
        print(f"[OSKY AI]: {output}")
        print(f"[Waktu]  : {elapsed:.2f} detik")

    print("\n" + "=" * 60)
    print("  EVALUASI SELESAI")
    print("=" * 60)

if __name__ == "__main__":
    main()
