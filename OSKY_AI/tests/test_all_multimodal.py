"""
OSKY AI - Comprehensive Unit & Integration Test Suite (Multimodal, Vision, Image Gen, Editing, Web, Retrieval)
Menguji secara menyeluruh seluruh subsistem yang dibangun dari nol (from scratch):
1. Patch Embedding & Vision Transformer Encoder
2. Handwriting / OCR CRNN Recognition Network
3. Multimodal Projector & Fusion Model
4. Image Generation (Diffusion Model & Conv-VAE)
5. Image Editing & Layout Analysis
6. Web Research & HTML Parser & Citations
7. Local Document Retrieval (TF-IDF Vector Index)
8. Continual Learning & Model Versioning Manager
"""

import os
import sys
import unittest
import torch
import numpy as np
from PIL import Image

# Tambahkan root direktori ke sys.path agar OSKY_AI dapat di-import sebagai paket
current_dir = os.path.dirname(os.path.abspath(__file__))
osky_ai_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(osky_ai_dir)
for p in [root_dir, osky_ai_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)


from OSKY_AI.vision import PatchEmbedding, VisionTransformerBlock, OskyVisionTransformer, preprocess_image
from OSKY_AI.handwriting import HandwritingCharVocabulary, OskyHandwritingRecognizer
from OSKY_AI.multimodal import MultimodalProjector, MultimodalFusion, OskyMultimodalModel
from OSKY_AI.image_generation import OskyConvVAE, OskyDiffusionUNet, OskyImageGenerator, LocalTextConditioner
from OSKY_AI.image_editing import analyze_image_layout, create_center_mask, OskyInpaintingNet, OskyEditingPipeline
from OSKY_AI.web import clean_html_text, extract_article_content, SourceCollection, OskyWebResearcher
from OSKY_AI.retrieval import chunk_text, load_local_document, OskyLocalVectorIndex, retrieve_and_assemble_context
from OSKY_AI.versioning import ModelVersionManager
from OSKY_AI.hardware import get_hardware_info, HARDWARE_PROFILES

class TestOskyMultimodalSuite(unittest.TestCase):

    def test_01_vision_patch_embedding_and_encoder(self):
        """Uji Patch Embedding dan Vision Transformer Encoder"""
        batch_size = 2
        img_size = 64
        patch_size = 8
        embed_dim = 128
        num_patches = (img_size // patch_size) ** 2  # 64 patches

        patch_layer = PatchEmbedding(image_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        dummy_img = torch.randn(batch_size, 3, img_size, img_size)
        
        # Forward pass patch embedding
        tokens = patch_layer(dummy_img)
        self.assertEqual(tokens.shape, (batch_size, num_patches + 1, embed_dim), "Shape patch embedding harus [B, N+1, D]")

        # Vision Transformer forward pass
        vision_model = OskyVisionTransformer({
            "vision_image_size": img_size,
            "vision_patch_size": patch_size,
            "vision_embed_dim": embed_dim,
            "vision_layers": 2,
            "vision_heads": 2
        })
        out = vision_model(dummy_img)
        self.assertIn("patch_tokens", out)
        self.assertIn("cls_token", out)
        self.assertEqual(out["patch_tokens"].shape, (batch_size, num_patches, embed_dim))
        self.assertEqual(out["cls_token"].shape, (batch_size, embed_dim))
        print("[PASS] Test 01: Vision Patch Embedding & Encoder sukses.")

    def test_02_handwriting_recognition(self):
        """Uji model CRNN tulisan tangan dan decode greedy"""
        vocab = HandwritingCharVocabulary()
        encoded = vocab.encode("OSKY 2026")
        decoded = vocab.decode(encoded)
        self.assertEqual(decoded, "OSKY 2026", "Encoding/decoding karakter harus konsisten")

        model = OskyHandwritingRecognizer(vocab_size=vocab.vocab_size)
        dummy_img = torch.randn(2, 1, 32, 128)
        log_probs = model(dummy_img)
        # Sequence output pada sumbu lebar W: [2, 32, vocab_size]
        self.assertEqual(log_probs.shape[0], 2)
        self.assertEqual(log_probs.shape[2], vocab.vocab_size)

        pred_str = model.recognize(dummy_img[0], vocab)
        self.assertIsInstance(pred_str, str)
        print("[PASS] Test 02: Handwriting/OCR Recognizer sukses.")

    def test_03_multimodal_fusion_and_loss(self):
        """Uji Multimodal Projector dan gabungan Vision + Language Model"""
        vis_dim = 128
        llm_dim = 256
        projector = MultimodalProjector(vision_dim=vis_dim, llm_dim=llm_dim)
        vis_feats = torch.randn(2, 64, vis_dim)
        proj_out = projector(vis_feats)
        self.assertEqual(proj_out.shape, (2, 64, llm_dim))

        # Model Terpadu Multimodal
        mm_model = OskyMultimodalModel(
            vision_config={"vision_image_size": 32, "vision_patch_size": 8, "vision_embed_dim": 64, "vision_layers": 1, "vision_heads": 2},
            language_config={"vocab_size": 256, "hidden_dim": 128, "num_layers": 2, "num_heads": 2, "context_length": 128}
        )
        dummy_pix = torch.randn(2, 3, 32, 32)
        dummy_tokens = torch.randint(0, 256, (2, 16))
        dummy_labels = dummy_tokens.clone()

        out = mm_model(dummy_pix, dummy_tokens, labels=dummy_labels)
        self.assertIn("logits", out)
        self.assertIn("loss", out)
        self.assertIsNotNone(out["loss"])
        self.assertTrue(out["loss"].item() > 0)
        print(f"[PASS] Test 03: Multimodal Fusion & VQA Loss sukses (Loss: {out['loss'].item():.4f}).")

    def test_04_image_generation_vae_and_diffusion(self):
        """Uji Image Generation: Conv-VAE dan Diffusion Score Model"""
        # 1. Conv VAE
        vae = OskyConvVAE(in_channels=3, latent_dim=32, image_size=32, base_channels=16)
        x_in = torch.rand(2, 3, 32, 32)
        vae_out = vae(x_in)
        self.assertEqual(vae_out["recon_x"].shape, (2, 3, 32, 32))
        self.assertIn("loss", vae_out)

        # 2. Diffusion Score UNet
        diffusion = OskyDiffusionUNet(in_channels=3, out_channels=3, hidden_dim=16, cond_dim=64, num_timesteps=10)
        cond = torch.randn(2, 64)
        t = torch.tensor([2, 5], dtype=torch.long)
        noise = torch.randn_like(x_in)
        x_noisy, injected = diffusion.q_sample(x_in, t, noise)
        pred_noise = diffusion(x_noisy, t, cond)
        self.assertEqual(pred_noise.shape, (2, 3, 32, 32))

        # 3. Generator Inference
        gen = OskyImageGenerator(image_size=32, cond_dim=64)
        img = gen.generate_from_prompt("lingkaran merah", mode="vae")
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (32, 32))
        print("[PASS] Test 04: Image Generation (VAE & Diffusion UNet) sukses.")

    def test_05_image_editing_and_layout(self):
        """Uji Layout Inspector dan Neural Inpainting / Editing Pipeline"""
        test_img = Image.new("RGB", (64, 64), color=(240, 240, 240))
        # Gambar kotak di kuadran kanan
        arr = np.array(test_img)
        arr[20:44, 30:54] = [200, 50, 50]
        test_img = Image.fromarray(arr)

        layout = analyze_image_layout(test_img)
        self.assertIn("aspect_ratio", layout)
        self.assertIn("symmetry_score", layout)
        self.assertIn("bounding_box", layout)

        pipeline = OskyEditingPipeline(device="cpu")
        edited_img, report = pipeline.execute_instruction(test_img, "Rapikan gambar dan seimbangkan margin")
        self.assertIsInstance(edited_img, Image.Image)
        self.assertIn("editing_plan", report)
        print("[PASS] Test 05: Image Editing & Layout Analysis sukses.")

    def test_06_web_research_and_sources(self):
        """Uji HTML Parser, Ekstraktor Artikel, dan Format Sitasi Web"""
        sample_html = """
        <html>
            <head><title>OSKY AI Documentation</title></head>
            <body>
                <script>console.log("ignore");</script>
                <h1>Arsitektur Mandiri</h1>
                <p>OSKY AI dibangun 100% dari random weights tanpa model luar.</p>
            </body>
        </html>
        """
        parsed = extract_article_content(sample_html)
        self.assertEqual(parsed["title"], "OSKY AI Documentation")
        self.assertIn("OSKY AI dibangun 100%", parsed["text"])

        sources = SourceCollection()
        sources.add("https://osky.ai/vision", "OSKY Vision Spec", "Spesifikasi patch embedding.")
        citations = sources.format_all_citations()
        self.assertIn("[1] OSKY Vision Spec", citations)

        researcher = OskyWebResearcher(max_pages=1)
        res = researcher.conduct_research("Robotika ESP32")
        self.assertIn("sources", res)
        self.assertIn("augmented_prompt", res)
        print("[PASS] Test 06: Web Research Controller & Citations sukses.")

    def test_07_local_document_retrieval(self):
        """Uji Document Chunker dan Local Vector Indexing from Scratch"""
        text = "OSKY AI adalah model mandiri. Dibangun dari nol tanpa bobot pretrained. Memiliki sistem vision dan OCR."
        chunks = chunk_text(text, chunk_size=5, overlap=1)
        self.assertTrue(len(chunks) >= 2)

        # Buat indeks lokal
        from OSKY_AI.retrieval.documents import DocumentChunk
        doc_chunks = [
            DocumentChunk("doc1", 0, "OSKY AI dibangun tanpa pretrained weights.", {}),
            DocumentChunk("doc2", 1, "Sistem vision menggunakan patch embedding ViT.", {})
        ]
        index = OskyLocalVectorIndex(vector_dim=64)
        index.build_index(doc_chunks)

        hits = index.search("pretrained weights", top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0][0].doc_id, "doc1")

        context = retrieve_and_assemble_context("vision patch", index)
        self.assertIn("doc2", context["context_str"])
        print("[PASS] Test 07: Local Document Vector Indexing & RAG sukses.")

    def test_08_continual_learning_versioning(self):
        """Uji Version Manager dan Model Registry"""
        vm = ModelVersionManager(registry_file="/tmp/test_version_registry.json")
        summary = vm.get_summary()
        self.assertIn("active_version", summary)

        vm.register_new_model_version(
            version_name="osky_model_v2",
            checkpoint_path="checkpoints/v2.pt",
            dataset_version="dataset_v2",
            eval_metrics={"perplexity": 4.8},
            description="Versi model hasil penambahan data vision."
        )
        self.assertEqual(vm.registry["current_active_version"], "osky_model_v2")
        print("[PASS] Test 08: Continual Learning & Model Versioning Manager sukses.")

if __name__ == "__main__":
    unittest.main()
