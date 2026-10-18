# OSKY AI — Dokumentasi Teknis Neural Network Architecture

Dokumentasi ini menjelaskan secara menyeluruh arsitektur, cara kerja internal, matematika dasar, siklus pelatihan (*training*), evaluasi, hingga proses inferensi dari seluruh modul **neural network OSKY AI (Osky Artificial Intelligence)**.

---

## Tiga Pertanyaan Utama

### 1. NEURAL NETWORK ADA DI MANA?
> **Jawaban:**  
> Seluruh arsitektur neural network OSKY AI berada di dalam direktori:  
> **`OSKY_AI/neural_network/`** (serta modul terkait di `OSKY_AI/model/`, `OSKY_AI/vision/`, `OSKY_AI/handwriting/`, `OSKY_AI/multimodal/`, `OSKY_AI/image_generation/`, dan `OSKY_AI/image_editing/`).  
> Modul ini berisi struktur kelas PyTorch (`nn.Module`), kalkulasi tensor, *forward pass*, *attention mechanisms*, *patch embeddings*, *convolutions*, dan proyeksi output.

### 2. AI DILATIH DI MANA?
> **Jawaban:**  
> Neural network tidak melatih dirinya sendiri. Proses pelatihan dijalankan oleh **Training Engine** proyek OSKY AI yang berpusat di:  
> **`OSKY_AI/training/`**  
> dengan berkas eksekusi utama (*entry point*):  
> **`OSKY_AI/train.py`**  
> 
> - **Neural Network** = Cetak biru arsitektur, parameter bobot, dan fungsi kalkulasi (*The Brain*).  
> - **Training Engine** = Sistem orkestrasi yang mengalirkan data, menghitung loss, melakukan backpropagation, dan menjalankan optimasi (*The Learning System*).  
> - **Dataset** = Bahan ajar berisi sampel teks, gambar, atau pasangan instruksi-jawaban (*The Study Material*).  
> - **Optimizer** = Mekanisme pembaruan nilai bobot numerik (*The Weight Updater*).  
> - **Checkpoint** = Berkas penyimpanan status bobot hasil belajar ke media disk (*The Memory Snapshot*).

### 3. CARA MELATIHNYA BAGAIMANA?
> **Jawaban:**  
> Pelatihan dilakukan melalui siklus iteratif matematis tertutup:  
> **Dataset** $\rightarrow$ **Tokenizer/Preprocessing** $\rightarrow$ **DataLoader** $\rightarrow$ **Neural Network** $\rightarrow$ **Forward Pass** $\rightarrow$ **Loss Computation** $\rightarrow$ **Backward Pass (Gradient)** $\rightarrow$ **Optimizer** $\rightarrow$ **Weight Update** $\rightarrow$ **Checkpointing** $\rightarrow$ **Validation** $\rightarrow$ **Repeat**.

---

## 1. Penjelasan OSKY AI

### Apa itu OSKY AI?
**OSKY AI** adalah sistem model kecerdasan buatan independen yang dibangun **100% dari nol (*from scratch*)** menggunakan pustaka tensor PyTorch. OSKY AI **TIDAK** menggunakan bobot (*weights*) pra-latih (*pretrained*) dari model luar (seperti OpenAI GPT, Meta LLaMA, Mistral, atau Google Gemini).

### Apa Fungsi Neural Network OSKY?
Neural network OSKY berfungsi sebagai mesin transformasi representasi matematis:
1. Menerima representasi numerik dari teks, gambar, atau sinyal tulisan tangan.
2. Memproses relasi antar-elemen melalui lapisan perhatian (*self-attention*) dan transformasi non-linear (*SwiGLU feed-forward*).
3. Memprediksi distribusi probabilitas untuk elemen berikutnya (*next token prediction* pada bahasa, *patch features* pada visi, atau *noise estimation* pada generasi citra).

### Bagaimana Neural Network Bekerja?
Neural network terdiri dari jutaan angka desimal (*floating point*) yang terorganisir dalam bentuk matriks bobot ($W$) dan bias ($b$). Saat input masuk, input tersebut dikalikan dengan bobot, dijumlahkan, dinormalisasi, dan dilewatkan pada fungsi aktivasi non-linear. 

> ⚠️ **Prinsip Kejujuran Teknis:**  
> Neural network OSKY **dimulai dari inisialisasi acak (*random initialization*)**. Saat baru diinstansiasi di memori komputer, bobotnya hanyalah bilangan acak Gaussian/Kaiming. **Model belum memiliki pengetahuan apa pun sebelum dilatih.** Output pertama yang dihasilkan sebelum training adalah teks acak (*gibberish*). Model tidak memiliki kesadaran, intuisi, atau kepintaran gaib.

### Perbedaan Istilah-Istilah Kunci
| Istilah | Penjelasan Teknis |
| :--- | :--- |
| **Architecture** | Rancangan struktural jaringan (misal: Decoder-Only Transformer 4 layer, 4 attention heads, dimensi embedding 256). Ini adalah kodenya, bukan nilainya. |
| **Parameters** | Total variabel numerik yang dapat diatur di dalam model (jumlah elemen bobot $W$ dan bias $b$). |
| **Weights** | Nilai numerik riil saat ini yang tersimpan di dalam parameter tensor (misal: `0.0231`, `-0.4120`). |
| **Training** | Proses iteratif mengoptimalkan nilai weights agar error (loss) antara prediksi model dan data sebenarnya mengecil. |
| **Inference** | Penggunaan model yang sudah memiliki bobot untuk memproses input dan menghasilkan jawaban tanpa mengubah bobot (tanpa backpropagation). |
| **Dataset** | Kumpulan contoh teks/gambar terkurasi yang disiapkan sebagai target acuan belajar model. |
| **Tokenizer** | Algoritma konversi teks alfabet/kata menjadi bilangan bulat indeks (*Token IDs*) yang dimengerti matriks neural network. |
| **Checkpoint** | Berkas biner (`.pt`) di disk yang menyimpan nilai bobot jaringan, status optimizer, dan nomor langkah training saat itu. |
| **Context Window** | Batas panjang urutan token maksimum yang dapat diproses bersamaan oleh model dalam satu jendela komputasi (misal 256 token). |
| **Model Memory** | *Parametric Memory* (pengetahuan yang melekat di dalam jutaan bobot hasil training) vs *Context Memory* (teks yang sedang dimasukkan user di prompt saat inferensi). |

---

## 2. Cara Kerja Language AI

Model bahasa OSKY AI (`OskyLanguageModel`) menggunakan arsitektur **Causal Autoregressive Decoder-Only Transformer**.

```
Input Text: "halo dunia"
    │
    ▼
Tokenizer
    │  Mengubah teks menjadi daftar ID numerik
    ▼
Token IDs: [104, 215]
    │
    ▼
Token Embedding (Lookup Table)
    │  Memetakan setiap ID ke vektor dense berdimensi D (hidden_dim)
    ▼
Positional Representation (RoPE: Rotary Position Embedding)
    │  Menyematkan informasi urutan dan jarak posisi token via rotasi sudut
    ▼
Transformer Blocks (L Layers)
    │  ├─ RMSNorm (Pre-Layer Normalization)
    │  ├─ Multi-Head Causal Self-Attention (Q, K, V + Causal Mask)
    │  ├─ Residual Addition (Skip Connection)
    │  ├─ RMSNorm
    │  ├─ SwiGLU Feed-Forward Network
    │  └─ Residual Addition
    ▼
Final RMSNorm
    │  Menstabilkan variansi aktivasi akhir
    ▼
Final Projection (LM Head / Unembedding)
    │  Memproyeksikan vektor D kembali ke ruang kosakata V (vocab_size)
    ▼
Logits [B, T, vocab_size]
    │  Nilai skor mentah tak ternormalisasi untuk setiap kandidat token
    ▼
Softmax & Sampling
    │
    ▼
Next Token Prediction
```

### Penjelasan Setiap Tahap
1. **Tokenizer**: Memecah karakter atau subkata teks menjadi nomor indeks integer (*Token IDs*). Contoh: kata `"halo"` dipetakan menjadi nomor ID tertentu sesuai vocabulary yang telah dilatih.
2. **Token Embedding**: Mengubah nomor ID menjadi vektor berdimensi kontinu (misal: ID `104` menjadi vektor 256 angka desimal).
3. **Rotary Position Embedding (RoPE)**: Menerapkan rotasi trigonometrik pada vektor Query dan Key sehingga jarak relatif antar-posisi kata dapat dipahami oleh dot-product attention tanpa batas posisi absolut.
4. **Q, K, V (Query, Key, Value)**:
   - **Query ($Q$)**: "Apa informasi yang sedang dicari oleh token ini?"
   - **Key ($K$)**: "Informasi apa yang dimiliki oleh token ini untuk ditawarkan?"
   - **Value ($V$)**: "Kandungan fitur representasi riil dari token ini."
5. **Multi-Head Self-Attention**:
   Menghitung skor kecocokan antara Query token saat ini dengan Key seluruh token lain:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$
6. **Causal Mask ($M$)**: Matriks segitiga atas berisi $-\infty$ untuk memastikan token pada posisi $t$ **hanya boleh melihat token $1 \dots t$**, dan tidak boleh mengintip kata-kata masa depan ($t+1 \dots T$).
7. **SwiGLU Feed-Forward Network**:
   Transformasi non-linear dua cabang yang memberikan kapasitas representasi fakta:
   $$\text{SwiGLU}(x) = (xW_{gate} \cdot \text{SiLU}(xW_{gate})) \otimes (xW_{up}) \cdot W_{down}$$
8. **Residual Connections**: Menjumlahkan input blok dengan output blok ($x + f(x)$) untuk menjamin gradien dapat mengalir tanpa hambatan (*vanishing gradient prevention*).
9. **Logits & Softmax**: Logits adalah skor mentah untuk setiap kata dalam kamus. Fungsi Softmax mengubah logits menjadi distribusi probabilitas $[0, 1]$ yang totalnya berjumlah 1.0.

---

## 3. Cara Kerja Training (Proses Belajar)

Training adalah siklus optimasi numerik di mana bobot acak disesuaikan sedikit demi sedikit agar mendekati pola data target.

```
Dataset (File JSONL/TXT)
   │
   ▼
Cleaning & Normalization (Pembersihan Teks)
   │
   ▼
Tokenizer Encoding
   │
   ▼
Training Batch [Input IDs, Target IDs, Loss Mask]
   │
   ▼
OSKY Neural Network (Forward Pass)
   │
   ▼
Predicted Logits
   │
   ▼
Loss Calculation (Cross-Entropy Loss vs Target IDs)
   │
   ▼
Backpropagation (Reverse Automatic Differentiation)
   │
   ▼
Gradients (∇W = dLoss / dW)
   │
   ▼
Gradient Clipping (Mencegah Exploding Gradients)
   │
   ▼
Optimizer Step (AdamW: W = W - lr * m_hat / (sqrt(v_hat) + eps))
   │
   ▼
Weight Update (Bobot Jaringan Diperbarui)
   │
   ▼
Checkpoint Saving & Validation
   │
   ▼
Repeat (Langkah Selanjutnya)
```

### Apa yang Terjadi pada Weight di Setiap Training Step?
- Pada awal training, bobot acak memprediksi semua kata dengan peluang seragam atau salah. Nilai **Loss sangat tinggi** (misal $\ln(1024) \approx 6.93$).
- Saat backpropagation, turunan parsial $\frac{\partial \mathcal{L}}{\partial W}$ dihitung. Angka ini memberitahu: *"Jika bobot ini dinaikkan sedikit, apakah loss akan naik atau turun?"*
- Optimizer AdamW menggeser bobot ke arah yang berlawanan dengan gradien. Nilai bobot berubah dalam orde kecil (misal berubah sebesar $0.0001$).
- Seiring berjalannya ribuan langkah, bobot-bobot ini secara kolektif mengkristalkan relasi statistik bahasa dari dataset.
- **Peringatan Ilmiah:** Penurunan loss **TIDAK OTOMATIS** berarti model pintar atau benar secara faktual. Penurunan loss hanya berarti model semakin mahir memprediksi urutan token pada dataset yang diberikan. Jika datasetnya buruk, bias, atau terjadi *overfitting*, model akan tetap menghasilkan jawaban yang keliru meskipun loss-nya rendah.

---

## 4. Cara Melatih OSKY AI

### Struktur Modul & Tanggung Jawab
```
OSKY_AI/
├── data/                    <-- Bahan ajar (Dataset teks/percakapan)
├── tokenizer/               <-- Pemetaan karakter/subkata ke Token IDs
├── neural_network/          <-- Otak / Arsitektur PyTorch murni
│   ├── language/            <-- Transformer Causal Decoder
│   ├── vision/              <-- Vision Transformer (ViT)
│   ├── handwriting/         <-- CRNN + CTC Recognizer
│   ├── multimodal/          <-- Projector & Visual-Language Fusion
│   ├── image_generation/    <-- Conv-VAE & Diffusion UNet
│   └── image_editing/       <-- Neural Inpainter & Layout Engine
├── training/                <-- Mesin pelatih (Trainer, DataLoader, Loss)
├── inference/               <-- Mesin penjawab (Sampler, Generator)
├── checkpoints/             <-- Tempat penyimpanan file bobot (.pt)
├── configs/                 <-- Konfigurasi hyperparameter model (.json)
├── train.py                 <-- Entry point terminal untuk training
└── evaluate.py              <-- Entry point terminal untuk evaluasi
```

---

## 5. Cara Menyiapkan Dataset

Dataset diletakkan di dalam folder **`OSKY_AI/data/`** atau **`OSKY_AI/datasets/`**.

### Contoh Format Dataset Instruksi (`data/train.jsonl`)
```jsonl
{"instruction": "Apa itu matahari?", "response": "Matahari adalah bintang di pusat tata surya kita."}
{"instruction": "Berapa 15 ditambah 25?", "response": "15 ditambah 25 adalah 40."}
{"instruction": "Apa ibu kota Indonesia?", "response": "Ibu kota Indonesia adalah Nusantara."}
```

### Contoh Format Percakapan Berantai (`data/conversations.jsonl`)
```jsonl
{"messages": [{"role": "user", "content": "Halo OSKY"}, {"role": "assistant", "content": "Halo! Ada yang bisa saya bantu hari ini?"}]}
{"messages": [{"role": "user", "content": "Jelaskan apa itu atom."}, {"role": "assistant", "content": "Atom adalah unit dasar materi yang terdiri dari inti atom dan elektron."}]}
```

> 🔒 **Etika & Keamanan Data:**
> - Dataset harus disiapkan dan dikurasi sendiri secara sadar oleh pengguna.
> - Sistem tidak melakukan scraping liar di internet secara otomatis.
> - **DILARANG KERAS** memasukkan data pribadi sensitif, kata sandi, atau API Key ke dalam file dataset.

---

## 6. Cara Training dari Terminal

Untuk memulai proses training, buka terminal di direktori proyek dan jalankan perintah:

```bash
# Melatih model dari awal menggunakan konfigurasi standar
python OSKY_AI/train.py --config OSKY_AI/configs/small.json --data OSKY_AI/data/sample_conversations.jsonl

# Melanjutkan training dari checkpoint sebelumnya (Resume Training)
python OSKY_AI/train.py --config OSKY_AI/configs/small.json --resume OSKY_AI/checkpoints/latest.pt
```

### Penjelasan Parameter Training
- `--config`: File JSON konfigurasi dimensi model dan hyperparameter.
- `--data`: Path ke file dataset (`.jsonl` atau `.txt`).
- `batch_size`: Jumlah sampel kalimat yang diproses sekaligus dalam satu langkah komputasi (misal: 16).
- `learning_rate`: Besaran langkah pembaruan bobot optimizer (misal: $5 \times 10^{-4}$).
- `gradient_accumulation_steps`: Akumulasi gradien sebelum melakukan update bobot untuk menyimulasikan batch size besar pada memory kecil.
- `max_seq_len`: Panjang sequence konteks token maksimum (misal: 256).
- `save_interval`: Frekuensi penyimpanan file checkpoint (misal: setiap 250 langkah).
- `eval_interval`: Frekuensi pengujian loss pada data validasi terpisah (misal: setiap 100 langkah).

---

## 7. Konfigurasi Model Kecil untuk CPU Testing

Untuk pengujian cepat pada laptop standar atau lingkungan CPU tanpa kartu grafis khusus, gunakan konfigurasi minimal (`OSKY_AI/configs/test_quick.json`):

```json
{
  "model_name": "OSKY-Quick-Test",
  "vocab_size": 256,
  "hidden_dim": 128,
  "num_layers": 2,
  "num_heads": 4,
  "max_seq_len": 128,
  "ffn_dim": 512,
  "dropout": 0.1,
  "norm_type": "rmsnorm",
  "use_rope": true
}
```
*Catatan:* Konfigurasi di atas hanya untuk verifikasi sintaksis, alur kode, dan pipeline pengujian, **bukan** untuk menghasilkan model percakapan berskala produksi.

---

## 8. Training pada GPU vs CPU

### Memeriksa Ketersediaan Akselerator Hardware
Jalankan perintah berikut di terminal:
```bash
python -c "import torch; print('CUDA Tersedia:', torch.cuda.is_available()); print('Jumlah GPU:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
```
- Jika menghasilkan `CUDA Tersedia: True`, maka engine training akan otomatis menggunakan GPU Nvidia untuk komputasi tensor paralel berkecepatan tinggi.
- Jika menghasilkan `False`, training akan berjalan di atas CPU menggunakan multithreading PyTorch standar.
- **Kejujuran Perangkat:** Model berparameter puluhan juta atau ratusan juta membutuhkan VRAM dan daya komputasi besar. Jangan pernah berekspektasi dapat melatih model besar di smartphone atau prosesor laptop entry-level.

---

## 9. Checkpoint & Pemulihan State

Checkpoint adalah berkas arsip biner PyTorch (`.pt`) yang menyimpan seluruh status pelatihan. File ini disimpan di folder `checkpoints/latest.pt`.

Isi minimal sebuah checkpoint OSKY AI:
- `model_state_dict`: Nilai seluruh matriks bobot ($W$) neural network.
- `optimizer_state_dict`: Nilai momentum ($m$) dan variansi ($v$) dari AdamW.
- `scheduler_state_dict`: Status langkah kurva learning rate.
- `step` & `epoch`: Angka iterasi terakhir yang telah dicapai.
- `best_val_loss`: Nilai validation loss terendah yang pernah dicapai.
- `config`: Konfigurasi arsitektur model agar tidak salah ukuran saat dimuat kembali.

---

## 10. Evaluasi Model

Evaluasi dilakukan untuk memastikan model tidak sekadar menghafal data latihan (*overfitting*):

```bash
python OSKY_AI/evaluate.py --checkpoint OSKY_AI/checkpoints/latest.pt --data OSKY_AI/data/validation.jsonl
```

### Metrik Evaluasi
- **Validation Loss**: Rata-rata error pada dataset validasi yang **belum pernah dilihat** oleh model selama training.
- **Perplexity (PPL)**: Eksponensial dari cross-entropy loss ($e^{\mathcal{L}}$). Menunjukkan seberapa bingung model memilih kata berikutnya. Nilai PPL yang lebih rendah menandakan kepastian prediksi yang lebih baik.
- **Validasi Independen**: Dataset validasi **HARUS** dipisahkan secara ketat dari dataset training.

---

## 11. Cara Menjalankan Inferensi (Generasi Teks)

Setelah model memiliki checkpoint bobot hasil training, inferensi dijalankan secara murni lokal tanpa internet:

```
User Prompt: "Jelaskan fotosintesis"
   │
   ▼
Tokenizer Encode -> Token IDs: [45, 128, 98]
   │
   ▼
OSKY Neural Network Forward Pass -> Logits [1, T, vocab_size]
   │
   ▼
Logit Filtering (Repetition Penalty -> Temperature -> Top-K -> Top-P / Nucleus)
   │
   ▼
Probabilitas Multinominal Sampling -> Token Baru Dipilih
   │
   ▼
Append Token ke Sequence & Ulangi Forward Pass
   │
   ▼
Selesai saat mencapai <|eos|> (End-of-Sequence) atau max_new_tokens
```

### Parameter Inferensi
- `temperature`: Mengontrol keacakan (suhu rendah = deterministik & fokus, suhu tinggi = kreatif).
- `top_k`: Membatasi pemilihan hanya pada $K$ token dengan probabilitas tertinggi.
- `top_p`: Membatasi pemilihan pada kumpulan token yang probabilitas kumulatifnya mencapai nilai $P$ (misal $0.90$).
- `repetition_penalty`: Memberikan penalti pada token yang sudah pernah muncul agar model tidak mengulang kalimat yang sama berulang-ulang.
- `stop_tokens`: Token khusus penanda berhenti generasi (seperti `<|im_end|>` atau `<|eos|>`).

---

## 12. Vision Training (Vision Transformer)

Model visi OSKY AI (`OskyVisionTransformer` di `neural_network/vision/`) memproses gambar dari random initialization:

```
Citra Masukan [B, 3, H, W]
   │
   ▼
Preprocessing (Resize & Normalisasi Kanal RGB ke [-1, 1])
   │
   ▼
Patch Embedding (Konvolusi Strided membagi citra ke N patches berukuran PxP)
   │
   ▼
Concat [CLS] Token + Tambahkan 2D Spatial Positional Embedding
   │
   ▼
L-Layer Bidirectional Vision Transformer (Pre-LN Self-Attention)
   │
   ▼
Visual Tokens [B, N, D] & CLS Representation [B, D]
   │
   ▼
Loss Computation & Backpropagation -> Weight Update
```

---

## 13. Handwriting / OCR Training

Model pembacaan tulisan tangan (`OskyHandwritingRecognizer` di `neural_network/handwriting/`):

```
Citra Tulisan Tangan [B, 1, 32, W]
   │
   ▼
CNN Feature Extractor (VGG-style Convolutions + MaxPool mereduksi tinggi)
   │
   ▼
Bidirectional Sequence Layer (Bidirectional GRU menangani dependensi kontekstual horizontal)
   │
   ▼
Linear Projection ke Karakter Vocab Logits [B, T_seq, vocab_size]
   │
   ▼
CTC Loss (Connectionist Temporal Classification Loss)
   │
   ▼
Backpropagation & Weight Update
```

---

## 14. Multimodal Training (Visual Question Answering)

Model multimodal (`OskyMultimodalModel` di `neural_network/multimodal/`):

```
Image [B, 3, H, W]                    Question Text [B, T_text]
       │                                         │
       ▼                                         ▼
Vision Transformer Encoder               Text Token Embedding
       │                                         │
Patch Tokens [B, N, D_vis]                       │
       │                                         │
Multimodal Projector (Linear + GELU)             │
       │                                         │
Projected Visual Tokens [B, N, D_llm]            │
       └──────────────────┬──────────────────────┘
                          ▼
           Multimodal Fusion Concatenation:
               [Visual Tokens || Text Tokens]
                          ▼
            OSKY Causal Transformer Backbone
                          ▼
            Next-Token Autoregressive Loss
```

### Format Dataset Multimodal
```jsonl
{"image": "datasets/multimodal/kucing.jpg", "question": "Hewan apa yang ada di gambar?", "answer": "Ini adalah seekor kucing."}
```

---

## 15. Image Generation Training (Conv-VAE & Diffusion)

Generasi citra dari nol (`neural_network/image_generation/`) menggunakan pendekatan prinsip fisika difusi:
1. **Autoencoder Latent (`OskyConvVAE`)**: Mengompresi piksel resolusi tinggi ke dalam representasi laten berdimensi rendah.
2. **Diffusion UNet (`OskyDiffusionUNet`)**: Menambahkan noise Gaussian bertahap pada citra di forward process ($t=0 \dots T$), lalu melatih jaringan neural untuk memprediksi dan membuang noise tersebut (*reverse denoising process*) dengan pengkondisian teks lokal.

---

## 16. Image Editing Training (Neural Inpainting & Layout)

Editing citra berbasis instruksi (`neural_network/image_editing/`):
- Menerima citra masukan, koordinat area perbaikan (*mask*), dan instruksi pengguna.
- Model inpainting memproses tensor berkanal 4 (RGB + Mask) untuk merekonstruksi piksel yang hilang secara halus dan mulus.

---

## 17. Apa yang Membuat OSKY "Lebih Pintar"?

Secara ilmiah, tidak ada keajaiban dalam AI. Model OSKY menjadi "lebih pintar" hanya melalui faktor-faktor teknis yang terukur:
1. **Kualitas Dataset**: Data yang bersih, faktual, bebas typo, dan memiliki struktur penjelasan yang logis.
2. **Volume Token**: Jumlah eksposur token unik yang dilatihkan kepada model.
3. **Kapasitas Arsitektur**: Ukuran dimensi hidden, jumlah head attention, dan kedalaman layer yang seimbang.
4. **Optimasi Hyperparameter**: Penggunaan learning rate schedule dengan *cosine annealing* dan *warmup* yang stabil.
5. **Kombinasi Multimodal**: Pengetahuan visual yang melengkapi pemahaman teks.
6. **Retrieval Knowledge (RAG)**: Penyediaan konteks dokumen lokal yang relevan saat inferensi untuk mencegah halusinasi data.

---

## 18. Continual Learning & Model Versioning

OSKY AI memisahkan secara tegas antara proses inferensi dan penambahan pengetahuan:
- **Obrolan biasa tidak mengubah bobot**: Berbicara dengan model tidak akan mengubah angka-angka weight secara langsung di memori.
- **Prosedur Pembaruan Bobot yang Benar**:
  1. Data percakapan atau dokumen baru dikumpulkan dan dibersihkan dari kesalahan.
  2. Data dimasukkan ke pipeline kurasi dan didaftarkan versinya (`dataset_v2`).
  3. Dilakukan *fine-tuning* terencana menggunakan `train.py`.
  4. Hasilnya divalidasi dan disimpan sebagai versi model baru (`osky_model_v2.pt`).

---

## 19. Operasional 100% Offline Tanpa Ketergantungan Eksternal

Setelah dependensi PyTorch dan checkpoint model tersimpan secara lokal di mesin Anda:
- Inferensi teks, visi, pembacaan OCR, editing gambar, dan pencarian dokumen lokal berjalan **100% tanpa internet**.
- Sistem **tidak membutuhkan** API key komersial (seperti OpenAI, Google Cloud, atau Anthropic) untuk menjalankan komputasi neural network internalnya.

---

## 20. Panduan Pemecahan Masalah (Troubleshooting)

| Gejala Masalah | Penyebab Teknis | Solusi yang Teruji |
| :--- | :--- | :--- |
| `CUDA is not available` | Driver Nvidia belum terpasang atau PyTorch terinstall versi CPU-only. | Jalankan pada CPU, atau install PyTorch dengan perintah CUDA yang sesuai dari situs pytorch.org. |
| `CUDA Out of Memory (OOM)` | Alokasi tensor melebihi kapasitas VRAM kartu grafis. | Turunkan `batch_size`, perpendek `max_seq_len`, atau tingkatkan `gradient_accumulation_steps`. |
| `Loss bernilai NaN` | Terjadi lonjakan gradien (*exploding gradients*) atau learning rate terlalu besar. | Aktifkan `max_grad_norm: 1.0` (gradient clipping), turunkan `learning_rate`, dan periksa apakah ada nilai 0 pada pembagian di dataset. |
| `Model tidak belajar (Loss stagnan)` | Target IDs tidak bergeser (*shifting*) atau learning rate terlalu kecil. | Periksa apakah `target_ids` digeser 1 posisi ke kanan terhadap `input_ids`, periksa aktivasi, dan pastikan bobot berubah di setiap step. |
| `RuntimeError: Sequence length exceeded` | Panjang token input melebihi kapasitas `max_seq_len` di RoPE. | Potong (*truncate*) input agar panjangnya $\le$ nilai `max_seq_len` pada konfigurasi model. |

---

## 21. Hubungan Antar Direktori Proyek

```
OSKY_AI/
├── data/                    <-- Tempat meletakkan file dataset latih & validasi
├── tokenizer/               <-- Logika Byte-Pair / Char Tokenizer dan file vocab.json
├── neural_network/          <-- Definisi layer, block, dan model PyTorch
├── training/                <-- Loop training, perhitungan loss, dan scheduling
├── inference/               <-- Algoritma decoding, sampling, dan text generation
├── checkpoints/             <-- Tempat penyimpanan snapshot bobot model .pt
├── configs/                 <-- Arsip konfigurasi hyperparameter model .json
├── web/                     <-- Tool riset pencarian web terpisah (External Tool)
├── retrieval/               <-- Mesin pencarian dokumen lokal RAG mandiri
├── evaluation/              <-- Metrik perplexity, loss, dan validasi model
├── train.py                 <-- Script CLI utama untuk memulai training
└── evaluate.py              <-- Script CLI utama untuk evaluasi model
```

---

## 22. Ringkasan Akhir

Modul `OSKY_AI/neural_network/` adalah fondasi murni komputasi matematika jaringan saraf tiruan OSKY AI. Dengan memisahkan secara disiplin antara **arsitektur neural**, **mesin pelatihan**, **dataset**, dan **alat bantu eksternal**, sistem ini dapat dipelajari, dilatih, dan dikembangkan secara transparan, ilmiah, dan berdaya guna tinggi dari baris kode pertama.
