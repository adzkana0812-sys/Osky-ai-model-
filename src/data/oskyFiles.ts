import { FileItem } from "../types";

export const OSKY_FILES: FileItem[] = [
  {
    path: "OSKY_AI/model/osky_model.py",
    name: "osky_model.py",
    category: "model",
    description: "Neural network bahasa utama (OskyLanguageModel) yang menghubungkan embedding, RoPE, N-layer transformer blocks, RMSNorm, LM Head, Cross-Entropy Loss, dan penghitung parameter otomatis.",
    content: `# OSKY AI - Full Neural Network Architecture (OskyLanguageModel)
# Dibangun 100% dari nol (random weight initialization).
# Sama sekali TIDAK menggunakan pretrained weights dari model manapun.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class OskyLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.hidden_dim = config["hidden_dim"]
        self.num_layers = config["num_layers"]
        self.num_heads = config["num_heads"]
        self.max_seq_len = config.get("max_seq_len", 512)
        self.ffn_dim = config.get("ffn_dim", 4 * self.hidden_dim)

        self.token_emb = OskyEmbedding(self.vocab_size, self.hidden_dim)
        self.rope = RotaryPositionalEmbedding(dim=self.hidden_dim // self.num_heads)
        self.layers = nn.ModuleList([TransformerBlock(...) for _ in range(self.num_layers)])
        self.final_norm = RMSNorm(self.hidden_dim)
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

        if config.get("tie_weights", True):
            self.lm_head.weight = self.token_emb.lut.weight

        self.apply(self._init_weights)`
  },
  {
    path: "OSKY_AI/model/attention.py",
    name: "attention.py",
    category: "model",
    description: "Causal Multi-Head Self-Attention dengan Q, K, V linear projections, RoPE complex 2D rotation, triangular causal mask, dan scaled dot-product attention.",
    content: `# OSKY AI - Multi-Head Causal Self-Attention
# Scaled Dot-Product: Softmax((Q @ K.T) / sqrt(d_k) + CausalMask) @ V`
  },
  {
    path: "OSKY_AI/model/embeddings.py",
    name: "embeddings.py",
    category: "model",
    description: "Token embedding matrix dan Rotary Position Embeddings (RoPE) untuk menyematkan posisi token secara rotasi koordinat 2D.",
    content: `# OSKY AI - Token Embedding & Rotary Position Embeddings (RoPE)`
  },
  {
    path: "OSKY_AI/model/transformer.py",
    name: "transformer.py",
    category: "model",
    description: "Pre-LN Transformer Block dengan RMSNorm (Root Mean Square Layer Normalization), SwiGLU Gated Feed-Forward Network, dan Residual Connections.",
    content: `# OSKY AI - TransformerBlock, RMSNorm, & SwiGLU FFN`
  },
  {
    path: "OSKY_AI/tokenizer/bpe_char_tokenizer.py",
    name: "bpe_char_tokenizer.py",
    category: "tokenizer",
    description: "Custom local tokenizer mandiri: encode, decode, vocab builder, BPE merges, unknown byte fallback handling, dan percakapan special tokens.",
    content: `# OSKY AI - Custom Local Tokenizer from Scratch (No Hugging Face required)`
  },
  {
    path: "OSKY_AI/training/dataset.py",
    name: "dataset.py",
    category: "training",
    description: "Pipeline data: JSONL, TXT, pembersihan teks, deduplikasi MD5 hash, tokenisasi, dynamic batching, padding, dan masking prompt untuk SFT.",
    content: `# OSKY AI - Dataset Pipeline & Preprocessing`
  },
  {
    path: "OSKY_AI/training/trainer.py",
    name: "trainer.py",
    category: "training",
    description: "Mesin pelatihan lengkap: deteksi CPU/CUDA GPU, forward pass, loss, backprop, gradient clipping, Cosine Warmup scheduler, checkpointing, dan test generation.",
    content: `# OSKY AI - Training Engine with Mixed Precision & Hardware Detection`
  },
  {
    path: "OSKY_AI/training/optimizer.py",
    name: "optimizer.py",
    category: "training",
    description: "AdamW dengan decoupled weight decay (memisahkan bobot matriks 2D dari bias/norm 1D) dan Cosine Annealing Learning Rate Scheduler dengan Linear Warmup.",
    content: `# OSKY AI - Decoupled AdamW & Cosine Annealing Scheduler`
  },
  {
    path: "OSKY_AI/training/checkpoint.py",
    name: "checkpoint.py",
    category: "training",
    description: "Pengelola checkpoint: menyimpan dan memuat bobot model, optimizer state, scheduler state, epoch, step, val_loss, dan konfigurasi.",
    content: `# OSKY AI - Checkpoint Manager & Resume Engine`
  },
  {
    path: "OSKY_AI/inference/generate.py",
    name: "generate.py",
    category: "inference",
    description: "Mesin inferensi mandiri lokal: Temperature, Top-K, Top-P (nucleus), Repetition Penalty, Stop Tokens, dan multi-turn conversation session.",
    content: `# OSKY AI - Local Autoregressive Inference & Chat Engine`
  },
  {
    path: "OSKY_AI/api/server.py",
    name: "server.py",
    category: "api",
    description: "Server HTTP REST API lokal mandiri (POST /v1/chat/completions, GET /health) dengan verifikasi API key server dan rate limiting.",
    content: `# OSKY AI - Local REST API Server`
  },
  {
    path: "OSKY_AI/api/auth.py",
    name: "auth.py",
    category: "api",
    description: "Manajemen keamanan server lokal: pembuatan API key osky_live_..., hashing SHA-256 + salt, token bucket rate limiter, dan proteksi server.",
    content: `# OSKY AI - Local API Key Security & Token-Bucket Rate Limiter`
  },
  {
    path: "OSKY_AI/train.py",
    name: "train.py",
    category: "root",
    description: "Script utama eksekusi training: python train.py --config configs/small.json",
    content: `# OSKY AI - Main Training CLI Script`
  },
  {
    path: "OSKY_AI/evaluate.py",
    name: "evaluate.py",
    category: "root",
    description: "Script evaluasi model: python evaluate.py --checkpoint checkpoints/latest.pt",
    content: `# OSKY AI - Model Evaluation & Benchmark Script`
  },
  {
    path: "OSKY_AI/tests/test_all.py",
    name: "test_all.py",
    category: "tests",
    description: "Unit test lengkap: Tokenizer, RoPE, Attention, TransformerBlock, Forward, Loss, Checkpoint, dan Weight Update Verification.",
    content: `# OSKY AI - Comprehensive Unit Test Suite`
  }
];
