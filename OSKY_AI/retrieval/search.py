"""
OSKY AI - Local Document Search & Context Assembler
Mengambil chunks teratas dari indeks lokal dan menyusunnya ke dalam format konteks RAG
untuk diproses oleh OSKY Language Model.
"""

from typing import Dict, Any, List
from OSKY_AI.retrieval.index import OskyLocalVectorIndex
from OSKY_AI.retrieval.documents import DocumentChunk

def retrieve_and_assemble_context(
    query: str,
    index: OskyLocalVectorIndex,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Mencari konteks lokal dan menyusun prompt berdasar dokumen internal.
    """
    hits = index.search(query, top_k=top_k)

    passages = []
    citations = []
    for chunk, score in hits:
        passages.append(f"[{chunk.doc_id} #Chunk {chunk.chunk_id} (Score: {score:.2f})]:\n{chunk.text}")
        citations.append(f"- Dokumen: {chunk.doc_id}, Bagian: {chunk.chunk_id}, Kemiripan: {score:.2f}")

    context_str = "\n\n".join(passages)
    citations_str = "\n".join(citations) if citations else "Tidak ada dokumen lokal yang cocok."

    rag_prompt = (
        f"<system>Anda adalah OSKY AI. Gunakan dokumen pengetahuan lokal berikut "
        f"untuk memberikan jawaban yang akurat.</system>\n\n"
        f"[PENGETAHUAN DOKUMEN LOKAL]:\n{context_str}\n\n"
        f"<user>{query}</user>\n"
        f"<assistant>"
    )

    return {
        "query": query,
        "hits_count": len(hits),
        "citations": citations_str,
        "context_str": context_str,
        "rag_prompt": rag_prompt
    }
