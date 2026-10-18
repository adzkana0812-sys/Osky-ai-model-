"""
OSKY AI - Handwriting Recognition & OCR Neural Network
Model CRNN (Convolutional + Recurrent/Sequence + CTC) dibangun 100% dari nol.
Mengekstraksi fitur spasial karakter dari gambar tulisan tangan,
memproyeksikannya ke deret sekuensial horizontal, dan memprediksi probabilitas karakter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from OSKY_AI.handwriting.dataset import HandwritingCharVocabulary

class OskyHandwritingRecognizer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 90,
        in_channels: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size

        # 1. Convolutional Visual Feature Extractor
        self.conv_net = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),  # H: 32 -> 16, W: 128 -> 64

            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),  # H: 16 -> 8, W: 64 -> 32

            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 1)), # H: 8 -> 4, W: 32 -> 32

            # Layer 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((4, 1)), # H: 4 -> 1, W: 32 -> 32
        )

        # 2. Sequence Modeling (Bidirectional GRU/LSTM)
        # Input per timestep width: 128 channels * 1 height = 128
        self.rnn = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 3. Output Projection to Vocabulary (2 * hidden_dim karena bidirectional)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Input:  images [B, 1, H, W]  (misal [B, 1, 32, 128])
        Output: log_probs [B, Seq_Len_W, vocab_size]
        """
        B = images.size(0)
        # Ekstraksi CNN: [B, 128, 1, W_out]
        conv_feats = self.conv_net(images)
        # Squeeze dimensi H: [B, 128, W_out] -> Permute: [B, W_out, 128]
        seq_feats = conv_feats.squeeze(2).permute(0, 2, 1)

        # RNN Temporal Sequence
        rnn_out, _ = self.rnn(seq_feats)  # [B, W_out, hidden_dim * 2]

        # Linear Logits: [B, W_out, vocab_size]
        logits = self.fc(rnn_out)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    def recognize(self, image_tensor: torch.Tensor, vocab: HandwritingCharVocabulary, beam_width: int = 1) -> str:
        """
        Decode output log_probs.
        Mendukung Greedy Argmax (beam_width=1) dan Top-K Beam Search sederhana untuk akurasi tinggi.
        """
        self.eval()
        with torch.no_grad():
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            log_probs = self.forward(image_tensor)  # [1, W, vocab_size]
            
            if beam_width <= 1:
                preds = torch.argmax(log_probs, dim=-1)[0].tolist()
                return vocab.decode(preds)
            
            # Simple Beam Search Decoding
            seq_len = log_probs.size(1)
            beams = [(0.0, [])]  # (score, path)
            for t in range(seq_len):
                new_beams = []
                step_log_probs = log_probs[0, t]  # [vocab_size]
                top_log_probs, top_indices = torch.topk(step_log_probs, k=min(beam_width, step_log_probs.size(0)))
                for score, path in beams:
                    for lp, idx in zip(top_log_probs.tolist(), top_indices.tolist()):
                        new_beams.append((score + lp, path + [idx]))
                # Sort dan ambil top-k
                new_beams.sort(key=lambda x: x[0], reverse=True)
                beams = new_beams[:beam_width]

            best_path = beams[0][1]
            return vocab.decode(best_path)
