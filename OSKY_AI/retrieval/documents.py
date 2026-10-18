"""
OSKY AI - Local Document Loader & Chunker
Memuat dan memecah dokumen lokal (TXT, Markdown, JSON, JSONL, HTML)
menjadi potongan teks (chunks) dengan overlap untuk pencarian semantik lokal.
"""

import os
import json
import re
from typing import List, Dict, Any

class DocumentChunk:
    def __init__(self, doc_id: str, chunk_id: int, text: str, metadata: Dict[str, Any]):
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata
        }

def chunk_text(text: str, chunk_size: int = 120, overlap: int = 20) -> List[str]:
    """Memecah teks panjang menjadi chunks dengan overlap kata"""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += (chunk_size - overlap)
    return chunks

def load_local_document(file_path: str) -> List[DocumentChunk]:
    """Membaca berbagai jenis file teks lokal dan menghasilkan DocumentChunk"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File dokumen tidak ditemukan: {file_path}")

    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[-1].lower()

    raw_text = ""
    if ext in [".txt", ".md", ".markdown"]:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()
    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
            raw_text = json.dumps(data, ensure_ascii=False, indent=2)
    elif ext == ".jsonl":
        lines = []
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                lines.append(line.strip())
        raw_text = "\n".join(lines)
    elif ext in [".html", ".htm"]:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
            raw_text = re.sub(r'<[^>]+>', ' ', raw)
    else:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()

    chunks_str = chunk_text(raw_text)
    chunks = []
    for idx, c in enumerate(chunks_str):
        chunks.append(DocumentChunk(
            doc_id=filename,
            chunk_id=idx,
            text=c,
            metadata={"source": file_path, "total_chunks": len(chunks_str)}
        ))
    return chunks
