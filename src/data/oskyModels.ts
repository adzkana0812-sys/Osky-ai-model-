import { NeuralNetworkSpec, HandwritingSample, VisionCategory } from '../types';

export const OSKY_NEURAL_NETWORKS: NeuralNetworkSpec[] = [
  {
    id: 'llm',
    name: 'OSKY Language Model (Causal Decoder)',
    category: 'Language',
    params: 4458752,
    formattedParams: '4,458,752 parameter (4.46M)',
    architecture: 'Causal Multi-Head Attention, RoPE, SwiGLU, RMSNorm',
    description: 'Model inti pemahaman bahasa, penalaran logis, dan dialog teks next-token prediction dari nol.',
    status: 'Trained',
    lossMetric: 'Cross-Entropy Loss',
    targetAccuracy: 'Perplexity: 1.84',
  },
  {
    id: 'vit',
    name: 'Vision Transformer (ViT Encoder)',
    category: 'Vision',
    params: 826496,
    formattedParams: '826,496 parameter (826K)',
    architecture: 'Patch Embedding Conv2D + Transformer Encoder Layers + Self-Attention',
    description: 'Memecah citra gambar menjadi patch grid 8x8 dan mengekstraksi representasi semantik visual.',
    status: 'Ready',
    lossMetric: 'Cross-Entropy Loss',
    targetAccuracy: 'Akurasi: 96.4%',
  },
  {
    id: 'crnn',
    name: 'Handwriting & OCR Recognizer (CRNN)',
    category: 'OCR',
    params: 756112,
    formattedParams: '756,112 parameter (756K)',
    architecture: 'Deep Conv2D Feature Extractor + Bidirectional GRU + CTC Decoder',
    description: 'Mengenali guratan tulisan tangan manusia (angka, alfabet, simbol rumus matematika) secara berurutan.',
    status: 'Ready',
    lossMetric: 'CTC Loss',
    targetAccuracy: 'CER: < 1.9%',
  },
  {
    id: 'projector',
    name: 'Multimodal Cross-Modal Projector',
    category: 'Multimodal',
    params: 99072,
    formattedParams: '99,072 parameter (99K)',
    architecture: 'Two-Layer Non-linear GeLU MLP + LayerNorm',
    description: 'Menjembatani ruang representasi visual Vision Transformer ke ruang dimensi embedding bahasa LLM.',
    status: 'Trained',
    lossMetric: 'Projection MSE / Contrastive Loss',
    targetAccuracy: 'Alignment Cosine: 0.94',
  },
  {
    id: 'fusion',
    name: 'Multimodal Joint Fusion Layer',
    category: 'Multimodal',
    params: 512,
    formattedParams: '512 parameter',
    architecture: 'Joint Normalization & Feature Alignment Layer',
    description: 'Mengintegrasikan token gambar dan teks secara simultan untuk tugas Visual Question Answering (VQA).',
    status: 'Trained',
    lossMetric: 'Joint Cross-Entropy',
    targetAccuracy: 'VQA Score: 91.2%',
  },
  {
    id: 'vae',
    name: 'Generative Conv-VAE (Variational Autoencoder)',
    category: 'Generative',
    params: 530371,
    formattedParams: '530,371 parameter (530K)',
    architecture: 'Convolutional Encoder-Decoder dengan Gaussian Reparameterization',
    description: 'Mengompresi gambar ke dalam manifold laten kontinyu dan merekonstruksi tekstur visual.',
    status: 'Trained',
    lossMetric: 'Reconstruction + KL Divergence',
    targetAccuracy: 'SSIM: 0.92',
  },
  {
    id: 'diffusion',
    name: 'Denoising Diffusion UNet',
    category: 'Generative',
    params: 929571,
    formattedParams: '929,571 parameter (930K)',
    architecture: 'U-Net Residual Down/Up Blocks + Sinusoidal Time Embedding',
    description: 'Menghasilkan citra gambar baru dari noise gaussian murni melalui proses denoising iteratif.',
    status: 'Ready',
    lossMetric: 'Noise MSE Loss',
    targetAccuracy: 'FID Score: 18.4',
  },
  {
    id: 'inpainting',
    name: 'Neural Image Inpainting & Editing Network',
    category: 'Generative',
    params: 90659,
    formattedParams: '90,659 parameter (91K)',
    architecture: 'Gated / Masked Residual Autoencoder (4-Channel RGB + Binary Mask)',
    description: 'Merekonstruksi dan memperbaiki bagian gambar yang terhapus, terpotong, atau diedit.',
    status: 'Trained',
    lossMetric: 'L1 + Perceptual Loss',
    targetAccuracy: 'PSNR: 32.5 dB',
  },
];

export const TOTAL_OSKY_PARAMETERS = 7691545;

export const DEFAULT_HANDWRITING_SAMPLES: HandwritingSample[] = [
  {
    id: 'hw-digit-7',
    label: '7',
    category: 'Angka',
    description: 'Angka 7 tulisan tangan dengan garis tengah miring khas manusia',
    svgStrokes: ['M 25 20 L 75 20 L 40 85', 'M 35 50 L 60 50'],
  },
  {
    id: 'hw-digit-3',
    label: '3',
    category: 'Angka',
    description: 'Angka 3 tulisan tangan melingkar',
    svgStrokes: ['M 30 25 C 50 15, 70 25, 55 45 C 75 55, 65 85, 30 80'],
  },
  {
    id: 'hw-math-1',
    label: '4 + 5 = 9',
    category: 'Matematika',
    description: 'Persamaan operasi aritmatika tulisan tangan',
    svgStrokes: [
      'M 15 25 L 15 50 L 30 50 M 25 20 L 25 70', // 4
      'M 38 45 L 50 45 M 44 38 L 44 52', // +
      'M 65 30 L 55 30 L 55 48 C 65 45, 68 65, 55 68', // 5
      'M 75 42 L 87 42 M 75 50 L 87 50', // =
      'M 105 45 C 105 32, 95 32, 95 45 C 95 58, 105 58, 105 45 L 105 70', // 9
    ],
  },
  {
    id: 'hw-word-osky',
    label: 'OSKY',
    category: 'Kata',
    description: 'Teks kapital tulisan tangan OSKY',
    svgStrokes: [
      'M 20 45 C 20 25, 35 25, 35 45 C 35 65, 20 65, 20 45', // O
      'M 48 30 C 40 25, 40 42, 48 48 C 55 54, 55 70, 42 68', // S
      'M 60 25 L 60 70 M 75 25 L 60 48 L 76 70', // K
      'M 85 25 L 93 48 L 101 25 M 93 48 L 93 70', // Y
    ],
  },
  {
    id: 'hw-letter-a',
    label: 'a',
    category: 'Huruf',
    description: 'Huruf kecil a tulisan tangan bersambung',
    svgStrokes: ['M 65 45 C 65 35, 45 35, 45 55 C 45 75, 65 75, 65 55 L 65 75 C 65 78, 70 78, 75 75'],
  },
  {
    id: 'hw-digit-8',
    label: '8',
    category: 'Angka',
    description: 'Angka 8 kurva tertutup berkelok ganda',
    svgStrokes: ['M 50 25 C 65 25, 65 45, 50 50 C 35 55, 35 78, 50 78 C 65 78, 65 55, 50 50 C 35 45, 35 25, 50 25'],
  },
];

export const DEFAULT_VISION_CATEGORIES: VisionCategory[] = [
  {
    id: 'kucing',
    name: 'Kucing (Hewan)',
    iconName: 'Cat',
    description: 'Kategori mamalia berkaki empat dengan sepasang telinga segitiga, kumis, dan mata elips.',
    sampleCount: 140,
    features: ['Bentuk Telinga Runcing', 'Tekstur Bulu Lembut', 'Pola Kumis & Hidung'],
  },
  {
    id: 'mobil',
    name: 'Mobil (Kendaraan)',
    iconName: 'Car',
    description: 'Kategori kendaraan bermotor dengan roda simetris, kaca depan, dan bodi horizontal aerodinamis.',
    sampleCount: 160,
    features: ['Kontur Roda Melingkar', 'Garis Bodi Logam', 'Refleksi Lampu Depan'],
  },
  {
    id: 'gedung',
    name: 'Gedung (Arsitektur)',
    iconName: 'Building',
    description: 'Kategori bangunan buatan manusia dengan pola kisi jendela vertikal dan garis tegak lurus.',
    sampleCount: 120,
    features: ['Garis Kisi Vertikal', 'Sudut Tegak Lurus 90°', 'Blok Geometris'],
  },
  {
    id: 'lingkaran',
    name: 'Bentuk Lingkaran',
    iconName: 'Circle',
    description: 'Bentuk geometris fundamental 2D dengan jari-jari seragam dari titik pusat simetri.',
    sampleCount: 100,
    features: ['Kelengkungan Kurva 360°', 'Simetri Radial', 'Ketiadaan Sudut Siku'],
  },
  {
    id: 'pohon',
    name: 'Pohon (Alam)',
    iconName: 'Tree',
    description: 'Vegetasi alam dengan pola kanopi dedaunan fraktal dan batang kayu vertikal silindris.',
    sampleCount: 130,
    features: ['Pola Kanopi Fraktal', 'Tekstur Batang Kasar', 'Warna Daun Hijau/Rimbun'],
  },
];
