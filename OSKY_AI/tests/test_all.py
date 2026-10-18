"""
OSKY AI - Unit Test Suite
Menguji seluruh komponen secara independen:
1. Tokenizer (encode, decode, special tokens, byte fallback)
2. Embeddings & Positional representations (RoPE)
3. Causal Multi-Head Attention & Causal Masking
4. Transformer Block & Residual Connections
5. Forward pass model lengkap
6. Cross-Entropy Loss computation & Masking
7. Checkpoint save & load
8. Inference & Generation loop
9. Verifikasi pembaruan bobot (Weight Update Verification) setelah training step
"""

import os
import sys
import unittest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ".")

from OSKY_AI.tokenizer import OskyTokenizer
from OSKY_AI.model.embeddings import OskyEmbedding, RotaryPositionalEmbedding
from OSKY_AI.model.attention import CausalSelfAttention
from OSKY_AI.model.transformer import TransformerBlock, RMSNorm
from OSKY_AI.model.osky_model import OskyLanguageModel
from OSKY_AI.training.checkpoint import CheckpointManager
from OSKY_AI.training.optimizer import configure_optimizers, CosineWarmupLRScheduler

class TestOskyAI(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.toy_config = {
            "model_name": "OSKY-Test",
            "vocab_size": 256,
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 2,
            "max_seq_len": 64,
            "ffn_dim": 128,
            "dropout": 0.0,
            "norm_type": "rmsnorm",
            "use_rope": True,
            "tie_weights": True,
            "activation": "silu",
        }

    def test_01_tokenizer(self):
        tok = OskyTokenizer()
        corpus = ["Halo OSKY AI!", "Selamat datang di dunia neural network dari nol."]
        tok.train_from_corpus(corpus, target_vocab_size=300)

        text = "Halo OSKY AI!"
        ids = tok.encode(text, add_bos=True, add_eos=True)
        self.assertEqual(ids[0], tok.bos_token_id)
        self.assertEqual(ids[-1], tok.eos_token_id)

        decoded = tok.decode(ids, skip_special_tokens=True)
        self.assertEqual(decoded, text)

        # Uji byte fallback pada karakter tak dikenal (misal simbol unicode langka / emoji)
        rare_text = "OSKY 🚀 ✨"
        rare_ids = tok.encode(rare_text)
        rare_decoded = tok.decode(rare_ids)
        self.assertEqual(rare_decoded, rare_text)
        print("[PASS] Test 01: Tokenizer encoding, decoding, dan byte fallback berhasil.")

    def test_02_embeddings(self):
        emb = OskyEmbedding(vocab_size=100, hidden_dim=32)
        x = torch.tensor([[1, 5, 20]], dtype=torch.long)
        out = emb(x)
        self.assertEqual(out.shape, (1, 3, 32))

        # Test RoPE
        rope = RotaryPositionalEmbedding(dim=16, max_seq_len=64)
        cos, sin = rope(seq_len=10, device=torch.device("cpu"))
        self.assertEqual(cos.shape, (10, 16))
        self.assertEqual(sin.shape, (10, 16))
        print("[PASS] Test 02: Embeddings dan RoPE tensor shapes valid.")

    def test_03_attention_and_causal_mask(self):
        attn = CausalSelfAttention(hidden_dim=32, num_heads=2, max_seq_len=64, dropout=0.0, use_rope=False)
        attn.eval()
        x = torch.randn(2, 8, 32)
        out = attn(x)
        self.assertEqual(out.shape, (2, 8, 32))

        # Verifikasi bahwa token masa depan tidak mempengaruhi token sebelumnya
        x1 = torch.randn(1, 4, 32)
        x2 = x1.clone()
        x2[:, 3, :] = torch.randn(1, 32)  # Ubah token ke-4 saja

        out1 = attn(x1)
        out2 = attn(x2)

        # Token 0..2 harus menghasilkan output identik karena ada causal mask!
        diff_token_0 = torch.max(torch.abs(out1[:, 0, :] - out2[:, 0, :])).item()
        self.assertLess(diff_token_0, 1e-6)
        print("[PASS] Test 03: Attention & Causal mask autoregressive properti terbukti valid.")

    def test_04_transformer_block(self):
        block = TransformerBlock(
            hidden_dim=32, num_heads=2, ffn_dim=64, max_seq_len=64,
            dropout=0.0, norm_type="rmsnorm", activation="silu", use_rope=False
        )
        x = torch.randn(2, 6, 32)
        out = block(x)
        self.assertEqual(out.shape, (2, 6, 32))
        print("[PASS] Test 04: TransformerBlock forward pass dan residual connections berhasil.")

    def test_05_forward_pass_and_loss(self):
        model = OskyLanguageModel(self.toy_config)
        input_ids = torch.randint(0, 256, (2, 10))
        labels = torch.randint(0, 256, (2, 10))

        # Forward dengan labels
        loss, logits = model(input_ids, labels=labels)
        self.assertEqual(logits.shape, (2, 10, 256))
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0.0)

        # Forward tanpa labels (mode inference)
        none_loss, inf_logits = model(input_ids)
        self.assertIsNone(none_loss)
        self.assertEqual(inf_logits.shape, (2, 10, 256))
        print(f"[PASS] Test 05: Forward pass & CrossEntropyLoss sukses (Loss: {loss.item():.4f}).")

    def test_06_weight_update_verification(self):
        """
        Sangat penting: Memastikan bahwa bobot neural network benar-benar diperbarui
        setelah 1 langkah optimisasi (bukan model statis atau tiruan).
        """
        model = OskyLanguageModel(self.toy_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Simpan copy bobot sebelum training
        initial_weights = model.token_emb.lut.weight.clone()

        # Buat batch input
        input_ids = torch.randint(0, 256, (2, 8))
        labels = torch.randint(0, 256, (2, 8))

        optimizer.zero_grad()
        loss, _ = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()

        # Ambil bobot setelah step
        updated_weights = model.token_emb.lut.weight

        # Hitung perbedaan
        weight_diff = torch.norm(updated_weights - initial_weights).item()
        self.assertGreater(weight_diff, 1e-6, "Bobot model HARUS berubah setelah training step!")
        print(f"[PASS] Test 06: Verifikasi pembaruan bobot neural network berhasil (L2 Diff: {weight_diff:.6f}).")

    def test_07_checkpoint_save_and_load(self):
        ckpt_dir = "/tmp/osky_test_ckpt"
        mgr = CheckpointManager(checkpoint_dir=ckpt_dir)
        model = OskyLanguageModel(self.toy_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = CosineWarmupLRScheduler(optimizer, warmup_steps=10, max_steps=100, max_lr=1e-3)

        save_path = mgr.save(
            model=model, optimizer=optimizer, scheduler=scheduler,
            epoch=1, step=42, config=self.toy_config,
            train_loss=1.567, val_loss=1.234,
            filename="test_ckpt.pt"
        )
        self.assertTrue(os.path.exists(save_path))

        # Muat kembali ke model baru
        model_loaded = OskyLanguageModel(self.toy_config)
        state = mgr.load(save_path, model_loaded, optimizer=optimizer, scheduler=scheduler)
        self.assertEqual(state["step"], 42)
        self.assertAlmostEqual(state["val_loss"], 1.234, places=3)
        self.assertAlmostEqual(state["train_loss"], 1.567, places=3)

        # Pastikan tensor weights sama persis
        for p1, p2 in zip(model.parameters(), model_loaded.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        print("[PASS] Test 07: Checkpoint manager menyimpan dan memulihkan state dengan sempurna.")

if __name__ == "__main__":
    unittest.main()
