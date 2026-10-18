"""
OSKY AI - Unified Multimodal Model (Vision + Language)
Mengintegrasikan:
1. OskyVisionTransformer (Vision Encoder)
2. MultimodalProjector (Visual-to-Text Projection)
3. MultimodalFusion (Stream Concatenation)
4. OskyLanguageModel (Transformer Decoder Backbone)

Seluruh arsitektur dibangun 100% dari random initialization tanpa model pihak ketiga.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from OSKY_AI.vision.vision_model import OskyVisionTransformer, preprocess_image
from OSKY_AI.multimodal.projector import MultimodalProjector
from OSKY_AI.multimodal.fusion import MultimodalFusion
from OSKY_AI.model.osky_model import OskyLanguageModel

class OskyMultimodalModel(nn.Module):
    def __init__(
        self,
        vision_config: Optional[Dict[str, Any]] = None,
        language_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.vision_config = vision_config or {
            "vision_image_size": 64,
            "vision_patch_size": 8,
            "vision_embed_dim": 128,
            "vision_layers": 2,
            "vision_heads": 2
        }
        self.language_config = language_config or {
            "vocab_size": 1024,
            "hidden_dim": 256,
            "num_layers": 4,
            "num_heads": 4,
            "context_length": 256
        }

        # 1. Vision Encoder
        self.vision_encoder = OskyVisionTransformer(self.vision_config)

        # 2. Multimodal Projector
        self.projector = MultimodalProjector(
            vision_dim=self.vision_config.get("vision_embed_dim", 128),
            llm_dim=self.language_config.get("hidden_dim", 256)
        )

        # 3. Fusion Layer
        self.fusion = MultimodalFusion(
            hidden_dim=self.language_config.get("hidden_dim", 256)
        )

        # 4. Language Model Backbone
        self.language_model = OskyLanguageModel(self.language_config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        pixel_values: [B, 3, H, W]
        input_ids:    [B, L_txt]
        labels:       [B, L_txt] (opsional, -100 untuk masking)
        """
        # 1. Vision encoding: [B, N_patches, vision_dim]
        vision_out = self.vision_encoder(pixel_values)
        patch_tokens = vision_out["patch_tokens"]

        # 2. Project visual features ke LLM dimension: [B, N_patches, llm_dim]
        projected_visual = self.projector(patch_tokens)
        B, N_vis, D = projected_visual.shape

        # 3. Dapatkan text token embeddings: [B, L_txt, llm_dim]
        text_emb = self.language_model.token_emb(input_ids)

        # 4. Fuse visual + text sequence: [B, N_vis + L_txt, llm_dim]
        # Validasi panjang urutan gabungan terhadap max_seq_len backbone
        max_seq_len = getattr(self.language_model, "max_seq_len", self.language_config.get("context_length", 256))
        total_len = N_vis + text_emb.shape[1]
        if total_len > max_seq_len:
            # Truncate segmen teks secara aman agar tidak menyebabkan overflow RoPE/pos_emb
            allowed_text_len = max(1, max_seq_len - N_vis)
            text_emb = text_emb[:, :allowed_text_len, :]
            if labels is not None:
                labels = labels[:, :allowed_text_len]

        joint_emb, _ = self.fusion(projected_visual, text_emb)
        total_len = joint_emb.shape[1]

        # 5. Teruskan melewati transformer layers
        rope_cos_sin = None
        if self.language_model.use_rope and self.language_model.rope is not None:
            rope_cos_sin = self.language_model.rope(total_len, joint_emb.device)
        elif self.language_model.pos_emb is not None:
            joint_emb = joint_emb + self.language_model.pos_emb(total_len, joint_emb.device)

        x = self.language_model.emb_dropout(joint_emb)
        for layer in self.language_model.layers:
            x = layer(x, rope_cos_sin=rope_cos_sin)
        x = self.language_model.final_norm(x)

        # 6. Proyeksi logits: [B, total_len, vocab_size]
        logits = self.language_model.lm_head(x)

        loss = None
        if labels is not None:
            # Ambil logits hanya pada segmen teks (setelah token visual)
            text_logits = logits[:, N_vis:, :]  # [B, L_txt, vocab_size]
            
            # Autoregressive shift: token t memprediksi token t+1
            shift_logits = text_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.language_config["vocab_size"]),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return {"logits": logits, "loss": loss}

    def generate_response(
        self,
        pixel_values: torch.Tensor,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 30,
        temperature: float = 0.7,
        top_k: int = 40,
        eos_token_id: Optional[int] = 3
    ) -> torch.Tensor:
        """
        Inferensi autoregresif multimodal.
        eos_token_id default adalah 3 (token </s> pada OskyTokenizer).
        """
        self.eval()
        with torch.no_grad():
            curr_ids = prompt_ids.clone()
            for _ in range(max_new_tokens):
                out = self.forward(pixel_values, curr_ids)
                next_token_logits = out["logits"][:, -1, :] / max(temperature, 1e-4)

                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                curr_ids = torch.cat([curr_ids, next_token], dim=1)

                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break

            return curr_ids
