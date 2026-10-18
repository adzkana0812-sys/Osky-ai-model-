"""
OSKY AI - Local Vector & Sparse Index (From Scratch)
Membangun indeks representasi vektor untuk pencarian dokumen lokal.
Menggunakan kombinasi Term Frequency - Inverse Document Frequency (TF-IDF)
dan Proyeksi Hashing Subword untuk menghasilkan dense-sparse embeddings lokal.
100% Bebas API eksternal dan tidak memerlukan pretrained weights embedding.
"""

import math
import hashlib
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from OSKY_AI.retrieval.documents import DocumentChunk

class OskyLocalVectorIndex:
    def __init__(self, vector_dim: int = 128):
        self.vector_dim = vector_dim
        self.chunks: List[DocumentChunk] = []
        self.vectors: np.ndarray = np.zeros((0, vector_dim), dtype=np.float32)
        self.idf_dict: Dict[str, float] = {}

    def _hash_word(self, word: str) -> int:
        h = hashlib.md5(word.encode("utf-8")).hexdigest()
        return int(h, 16) % self.vector_dim

    def _embed_text(self, text: str) -> np.ndarray:
        words = text.lower().split()
        vec = np.zeros(self.vector_dim, dtype=np.float32)
        if not words:
            return vec

        for w in words:
            idx = self._hash_word(w)
            weight = self.idf_dict.get(w, 1.0)
            vec[idx] += weight

        # L2 Normalization
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec

    def build_index(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        num_docs = max(1, len(chunks))

        # 1. Hitung frekuensi dokumen untuk IDF
        doc_freq: Dict[str, int] = {}
        for c in chunks:
            unique_words = set(c.text.lower().split())
            for w in unique_words:
                doc_freq[w] = doc_freq.get(w, 0) + 1

        self.idf_dict = {
            w: math.log((num_docs + 1) / (df + 1)) + 1.0
            for w, df in doc_freq.items()
        }

        # 2. Hasilkan vektor untuk setiap chunk
        vec_list = []
        for c in chunks:
            vec_list.append(self._embed_text(c.text))

        if vec_list:
            self.vectors = np.stack(vec_list, axis=0)
        else:
            self.vectors = np.zeros((0, self.vector_dim), dtype=np.float32)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        if len(self.chunks) == 0 or self.vectors.shape[0] == 0:
            return []

        q_vec = self._embed_text(query)
        # Cosine similarity: perkalian titik karena vektor sudah di-L2 normalize
        dense_scores = np.dot(self.vectors, q_vec)

        # Hybrid lexical boost: beri bobot ekstra jika ada exact term overlap
        q_tokens = set(query.lower().split())
        hybrid_scores = np.copy(dense_scores)
        for i, c in enumerate(self.chunks):
            doc_tokens = set(c.text.lower().split())
            overlap = len(q_tokens.intersection(doc_tokens))
            if overlap > 0:
                # Lexical overlap boost (BM25-style precision booster)
                overlap_ratio = overlap / max(1, len(q_tokens))
                hybrid_scores[i] += 0.35 * overlap_ratio

        top_indices = np.argsort(-hybrid_scores)[:top_k]

        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(hybrid_scores[idx])))
        return results
