"""
OSKY AI - Web Research Controller
Mengkoordinasikan:
1. Analisis Kebutuhan Riset Pengguna
2. Web Search Tool
3. Fetching Konten Halaman Web
4. Pembersihan Teks & Ekstraksi Kutipan Relevan
5. Penyusunan Sitasi Terverifikasi
6. Injeksi Konteks ke OSKY Language Model

PRINSIP WAJIB:
Web adalah alat bantu eksternal (Tool), BUKAN weights dari neural network.
Tidak melakukan pretraining massal otomatis dari internet.
"""

from typing import Dict, Any, Optional
from OSKY_AI.web.search import search_web
from OSKY_AI.web.fetch import fetch_url
from OSKY_AI.web.parser import extract_article_content
from OSKY_AI.web.sources import SourceCollection

class OskyWebResearcher:
    def __init__(self, max_pages: int = 2):
        self.max_pages = max_pages

    def conduct_research(self, query: str) -> Dict[str, Any]:
        """
        Menjalankan riset web terstruktur untuk mendapatkan konteks berbasis fakta.
        """
        sources = SourceCollection()
        search_hits = search_web(query, max_results=self.max_pages)

        passages = []
        for hit in search_hits:
            url = hit["url"]
            title = hit["title"]
            snippet = hit["snippet"]

            # Unduh halaman jika memungkinkan untuk mendapatkan konten lebih lengkap
            html_content, err = fetch_url(url, timeout=3.0)
            if html_content and not err:
                parsed = extract_article_content(html_content)
                body_snippet = parsed["text"][:300]
                content_to_use = body_snippet if len(body_snippet) > 50 else snippet
            else:
                content_to_use = snippet

            sources.add(url=url, title=title, snippet=content_to_use)
            passages.append(f"[{title}]: {content_to_use}")

        context_text = "\n\n".join(passages)
        citations_text = sources.format_all_citations()

        # Format prompt yang siap disuntikkan ke context window OSKY Language Model
        augmented_prompt = (
            f"<system>Anda adalah OSKY AI dengan kapabilitas penelusuran web terverifikasi. "
            f"Gunakan informasi eksternal berikut untuk menjawab pertanyaan pengguna.</system>\n\n"
            f"[INFORMASI SUMBER WEB]:\n{context_text}\n\n"
            f"<user>{query}</user>\n"
            f"<assistant>"
        )

        return {
            "query": query,
            "sources": [s.to_dict() for s in sources.sources],
            "context_text": context_text,
            "citations_text": citations_text,
            "augmented_prompt": augmented_prompt
        }
