# OSKY AI (Osky Artificial Intelligence)
### Neural Network Bahasa Transformer Modern dari Nol (100% From Scratch)

OSKY AI adalah model kecerdasan buatan bahasa mandiri yang dibangun sepenuhnya dari **random initialization** tanpa mengunduh, memakai, atau men-download pretrained weights dari model komersial seperti ChatGPT, Gemini, Claude, LLaMA, Qwen, Mistral, DeepSeek, atau model lainnya.

Seluruh bobot (weights) OSKY AI diperoleh murni melalui proses training mandiri pada neural network OSKY AI sendiri.

---

## 1. Diagram Alur Arsitektur Neural Network

```
[ Input Teks: "Halo OSKY AI" ]
              │
              ▼
    [ OskyTokenizer ]  (Byte-level / Subword Encoding)
              │  Token IDs: [266, 263, 267, 268, ...]
              ▼
   ┌─────────────────────────────────────────────────────────┐
   │                  OSKY NEURAL NETWORK                    │
   │                                                         │
   │  [ Token Embedding Layer ]  (vocab_size -> hidden_dim)  │
   │              │                                          │
   │              ▼                                          │
   │      [ + Dropout ]                                      │
   │              │                                          │
   │              ▼                                          │
   │   ┌──────────────────────────────────────────────────┐  │
   │   │  TRANSFORMER BLOCK (Diulang sebanyak N Layer)     │  │
   │   │                                                  │  │
   │   │   x ───┬──► [ RMSNorm 1 ]                        │  │
   │   │        │          │                              │  │
   │   │        │          ▼                              │  │
   │   │        │     [ Q, K, V Projections ]             │  │
   │   │        │          │                              │  │
   │   │        │          ▼                              │  │
   │   │        │     [ RoPE Positional Rotation ]        │  │
   │   │        │          │                              │  │
   │   │        │          ▼                              │  │
   │   │        │     [ Scaled Dot-Product Attention ]    │  │
   │   │        │     (Scores = Softmax(Q·K^T/√d + Mask)) │  │
   │   │        │          │                              │  │
   │   │        │          ▼                              │  │
   │   │        │     [ Output Projection W_o ]           │  │
   │   │        │          │                              │  │
   │   │        └──► ( + ) ◄┘  (Residual Connection 1)    │  │
   │   │               │                                  │  │
   │   │               ├─────┐                            │  │
   │   │               │     ▼                            │  │
   │   │               │  [ RMSNorm 2 ]                   │  │
   │   │               │     │                            │  │
   │   │               │     ▼                            │  │
   │   │               │  [ SwiGLU Feed-Forward Network ] │  │
   │   │               │  (SiLU(x·W_gate) * (x·W_up))·W_d │  │
   │   │               │     │                            │  │
   │   │               └──►( + )  (Residual Connection 2) │  │
   │   │                     │                            │  │
   │   └─────────────────────┼────────────────────────────┘  │
   │                         ▼                               │
   │               [ Final RMSNorm Layer ]                   │
   │                         │                               │
   │                         ▼                               │
   │           [ Language Model Head (Linear) ]              │
   │                         │                               │
   │                         ▼                               │
   │       Logits Output: [batch, seq_len, vocab_size]       │
   └─────────────────────────┬───────────────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [ Mode Training ]                 [ Mode Inference ]
   Cross-Entropy Loss                Sampling (Temp, Top-P, Top-K)
   Backpropagation (AdamW)           Next Token -> Output Text
```

---

## 2. Penjelasan Komponen Arsitektur

1. **OskyEmbedding**: Memetakan token integer diskrit ke dalam representasi vektor kontinu berdimensi `hidden_dim`.
2. **Rotary Position Embedding (RoPE)**: Pendekatan modern yang memutar vektor Query dan Key secara ortogonal di ruang 2D sesuai posisi token. Menggantikan positional embedding konvensional, menghemat parameter, dan meningkatkan ekstrapolasi panjang urutan konteks.
3. **RMSNorm (Root Mean Square Normalization)**: Normalisasi modern tanpa pemusatan rata-rata (mean-centering) yang lebih efisien secara komputasi dan memberikan gradien yang sangat stabil selama training dari nol.
4. **Multi-Head Causal Self-Attention**:
   - Memproyeksikan hidden state ke dalam $Q$ (Query), $K$ (Key), dan $V$ (Value).
   - Menghitung kecocokan atensi antar token: $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$.
   - **Causal Mask ($M$)**: Matriks segitiga bawah yang bernilai $-\infty$ pada posisi masa depan, memastikan model tidak bisa melihat token setelahnya saat memprediksi kata berikutnya.
5. **SwiGLU Feed-Forward Network**:
   - Lapisan non-linear gating: $\text{FFN}(x) = (\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})) W_{\text{down}}$.
   - Memberikan kapasitas representasi pengetahuan yang jauh lebih baik daripada FFN ReLU/GELU biasa.
6. **Residual Highway (Pre-LN)**: Normalisasi diterapkan sebelum sub-layer sehingga gradien backpropagation dapat mengalir bebas tanpa hambatan ke layer awal.
7. **Weight Tying**: Mengikat bobot matriks proyeksi output (LM Head) dengan matriks token embedding untuk efisiensi memori dan regularisasi representasi semantik.

---

## 3. Konfigurasi Bertingkat OSKY AI

| Parameter | OSKY AI Small (Testing & Belajar) | OSKY AI Medium (Server Ringan) | OSKY AI Large (GPU Server Mandiri) |
|---|---|---|---|
| **Vocab Size** | 1,024 | 4,096 | 8,192 |
| **Hidden Dim** | 256 | 512 | 1,024 |
| **Num Layers** | 4 | 8 | 16 |
| **Attention Heads**| 4 (head dim: 64) | 8 (head dim: 64) | 16 (head dim: 64) |
| **Context Length** | 256 tokens | 512 tokens | 1,024 tokens |
| **FFN Dim** | 1,024 (SwiGLU) | 2,048 (SwiGLU) | 4,096 (SwiGLU) |
| **Total Parameter**| **~3.5 Juta (3.5M)** | **~25.8 Juta (25.8M)** | **~195 Juta (195M)** |
| **VRAM FP16** | ~35 MB | ~180 MB | ~1.2 GB |
| **Rekomendasi Hardware** | CPU Biasa / Laptop | GPU 4GB VRAM / CPU Cepat | GPU 8GB - 16GB VRAM |

---

## 4. Perbedaan Konseptual Krusial

### A. Tokenisasi Bahasa vs. API Key/Token
- **Token Bahasa**: Unit linguistik matematis (karakter, subword, atau byte) yang diubah menjadi angka integer agar dapat diproses oleh perkalian matriks neural network (misal kata "Halo" -> token ID `266`). OSKY AI menggunakan custom tokenizer lokal tanpa API key apapun.
- **API Key / Security Token**: String kriptografi acak (seperti `osky_live_9f8a...`) yang digunakan oleh server HTTP lokal untuk mengotentikasi pengguna dan mencegah akses tidak sah ke server Anda.

### B. Context Window vs. Model Memory vs. Dataset Training
- **Context Window**: Memori kerja jangka pendek sementara. Batas jumlah token maksimum yang dapat dilihat model dalam satu kali giliran inferensi (misal 512 token). Begitu prompt baru melebihi batas ini, token terlama akan tergeser keluar (sliding window).
- **Model Memory (Weights)**: Pengetahuan permanen model yang tersimpan di dalam miliaran angka floating-point pada parameter neural network hasil training.
- **Dataset Training**: File data mentah di disk yang digunakan optimizer untuk mengubah bobot model selama fase training. Model tidak lagi membaca dataset saat tahap inferensi berjalan.

---

## 5. Panduan Instalasi & Eksekusi

### 1. Persiapan Lingkungan
```bash
cd OSKY_AI
pip install -r requirements.txt
```

### 2. Menjalankan Unit Test (Verifikasi Sistem)
```bash
python -m unittest tests/test_all.py
```
Test ini mencakup:
- Tokenizer encode/decode dan penanganan byte fallback
- RoPE and causal mask autoregressive
- Forward pass dan CrossEntropyLoss
- **Verifikasi pembaruan bobot** (memastikan bobot neural network benar-benar berubah saat training)
- Checkpoint save dan load

### 3. Memulai Training dari Nol
```bash
python train.py --config configs/small.json --data data/sample_conversations.jsonl
```
Sistem akan:
1. Mendeteksi hardware (CPU / CUDA GPU, RAM, VRAM).
2. Membangun vocabulary tokenizer lokal dan menyimpannya di `tokenizer/vocab.json`.
3. Menginisialisasi bobot neural network dari acak (Gaussian distribution).
4. Menampilkan hitungan parameter total dan footprint memori.
5. Melakukan forward, backpropagation, optimizer step, gradient clipping, dan evaluasi berkala.
6. Menyimpan checkpoint otomatis ke `checkpoints/latest.pt` dan `checkpoints/best.pt`.

### 4. Melanjutkan Training (Resume Checkpoint)
Jika proses training terhenti, Anda dapat melanjutkannya tanpa mengulang:
```bash
python train.py --config configs/small.json --resume checkpoints/latest.pt
```

### 5. Evaluasi Model & Benchmark
```bash
python evaluate.py --checkpoint checkpoints/latest.pt --data data/sample_conversations.jsonl
```
Menampilkan:
- Validation Loss
- Nilai Perplexity (PPL)
- Jumlah token yang dievaluasi
- Output pengujian teks benchmark

### 6. Menjalankan Inferensi Lokal (CLI)
Inferensi prompt tunggal:
```bash
python inference/generate.py --checkpoint checkpoints/latest.pt --prompt "Siapa kamu?"
```

Menjalankan sesi Chatbot Interaktif:
```bash
python inference/generate.py --checkpoint checkpoints/latest.pt --interactive
```

### 7. Menjalankan REST API Server Lokal
Buat API key server baru:
```bash
python -m api.auth "admin_lokal"
```

Jalankan server API lokal:
```bash
python api/server.py --checkpoint checkpoints/latest.pt --port 8000
```

Uji API menggunakan `curl`:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <MASUKKAN_API_KEY_ANDA>" \
  -d '{
    "model": "osky-ai",
    "messages": [
      {"role": "user", "content": "Halo! Jelaskan siapa dirimu."}
    ],
    "temperature": 0.7,
    "max_tokens": 80
  }'
```

---

## 7. Ekosistem Multimodal OSKY AI (Vision, Image, Web, Retrieval)

OSKY AI dirancang dengan arsitektur modular yang dapat dikembangkan dan dilatih bertahap tanpa mencampuradukkan weights neural network dengan web tools:

```
                OSKY AI
                   │
    ┌──────────────┼──────────────┐
    │              │              │
   TEXT          VISION          WEB
    │              │              │
Transformer    Vision Model   Research Tool
    │              │              │
    └──────────────┼──────────────┘
                   │
           MULTIMODAL FUSION
                   │
          OSKY REASONING CORE
                   │
    ┌──────────────┼──────────────┐
    │              │              │
   TEXT          IMAGE          TOOLS
GENERATION    GENERATION      /RESEARCH
    │              │              │
    └──────────────┼──────────────┘
                   │
              OSKY OUTPUT
```

### Panduan Modul Multimodal:
1. **OSKY Vision (`OSKY_AI/vision/`)**:
   - `PatchEmbedding`: Membagi citra ke dalam ukuran patch (misal 8x8 atau 16x16) dan menambahkan posisi spasial 2D.
   - `OskyVisionTransformer`: Vision Transformer dua arah (bidirectional ViT) dari random initialization.
2. **Handwriting / OCR (`OSKY_AI/handwriting/`)**:
   - `OskyHandwritingRecognizer`: CNN + Bidirectional GRU/Sequence layer + CTC loss untuk membaca tulisan tangan dan dokumen.
3. **Multimodal Fusion (`OSKY_AI/multimodal/`)**:
   - `MultimodalProjector` & `MultimodalFusion`: Memproyeksikan representasi visual langsung ke deret token embedding Language Model.
4. **Image Generation (`OSKY_AI/image_generation/`)**:
   - `OskyDiffusionUNet` & `OskyConvVAE`: Model difusi dan Variational Autoencoder lokal dari bobot acak dengan pengkondisian teks lokal (`LocalTextConditioner`).
5. **Image Editing (`OSKY_AI/image_editing/`)**:
   - `OskyEditingPipeline` & `OskyInpaintingNet`: Menganalisis layout (keseimbangan simetri, margin, alignment) dan mengeksekusi rencana editing instruksi (auto-centering, inpainting objek, perbaikan kontras).
6. **Web Research (`OSKY_AI/web/`)**:
   - `OskyWebResearcher`: Tool pencarian, penarikan halaman, ekstraksi artikel, dan format sitasi web terverifikasi (dipisahkan secara tegas dari weights neural network).
7. **Local Retrieval / RAG (`OSKY_AI/retrieval/`)**:
   - `OskyLocalVectorIndex`: Indeks vektor dense-sparse mandiri (TF-IDF + Subword Hashing) tanpa model embedding eksternal.
8. **Continual Learning & Versioning (`OSKY_AI/versioning.py`)**:
   - Melacak `osky_model_v1`, `v2`, `v3` dan `dataset_v1`, `v2` dengan metadata audit trail yang transparan.

### Menjalankan Unit Test Lengkap Seluruh Sistem:
```bash
# Uji model bahasa inti (Core Transformer, RoPE, RMSNorm, SwiGLU, Checkpoint)
python -m unittest tests/test_all.py

# Uji sistem multimodal (Vision, Handwriting, Multimodal, Diffusion, Editing, Web, Retrieval, Versioning)
python -m unittest tests/test_all_multimodal.py
```


Sangat penting untuk memahami batas kapabilitas model:
1. **Bukan Pengganti Langsung LLM 70B Komersial**: OSKY AI Small (~3.5M params) atau Medium (~25.8M params) yang dilatih dengan dataset kecil adalah model miniatur yang sangat baik untuk memahami cara kerja mendalam neural network, representasi bahasa, dan inferensi lokal. Model ini tidak dapat menjawab pertanyaan ensiklopedis dunia layaknya model triliunan token seperti ChatGPT/Claude yang dilatih menggunakan ribuan cluster GPU H100 selama berbulan-bulan dengan biaya jutaan dolar.
2. **Kualitas Jawaban Bergantung Penuh pada Data**: Karena OSKY AI diinisialisasi dari bobot acak (random weights), model awalnya hanya akan mengeluarkan huruf/token acak. Setelah ribuan langkah training pada korpus berkualitas, model akan mulai membentuk pola kata, tata bahasa, dan gaya percakapan yang konsisten.
3. **Reproduksibilitas Penuh**: Kode ini memberikan kedaulatan 100% atas arsitektur, data, dan bobot kecerdasan buatan Anda sendiri tanpa back-door atau pelacakan telemetri pihak ketiga.
