"""
OSKY AI - Web Search Provider
Menjalankan pencarian kata kunci ke search provider (DuckDuckGo Lite / HTML)
dengan proteksi SSRF dan penanganan kegagalan jaringan yang transparan.
TIDAK PERNAH membuat hasil pencarian fiktif jika offline atau pencarian gagal.
"""

import urllib.parse
import re
from typing import List, Dict, Any
from OSKY_AI.web.fetch import fetch_url
from OSKY_AI.web.parser import clean_html_text

def search_web(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    Mencari informasi web publik relevan berdasarkan query.
    Mengembalikan daftar item: [{"title": ..., "url": ..., "snippet": ...}]
    Jika koneksi offline atau query tidak menghasilkan data, mengembalikan list kosong [].
    """
    query = query.strip()
    if not query:
        return []

    encoded_query = urllib.parse.quote_plus(query)
    search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

    html_content, err = fetch_url(search_url, timeout=4.0)
    results: List[Dict[str, str]] = []

    if html_content and not err:
        # Parsing sederhana hasil pencarian dari tag class web-result
        matches = re.findall(
            r'<a class="result__url"[^>]*href="([^"]+)"[^>]*>(.*?)</a>[\s\S]*?<a class="result__snippet"[^>]*>(.*?)</a>',
            html_content,
            flags=re.IGNORECASE
        )
        for url_raw, title_raw, snippet_raw in matches[:max_results]:
            # Ekstraksi URL sebenarnya jika dialihkan
            actual_url = url_raw
            if "uddg=" in url_raw:
                parts = url_raw.split("uddg=")
                if len(parts) > 1:
                    actual_url = urllib.parse.unquote(parts[1].split("&")[0])

            results.append({
                "title": clean_html_text(title_raw),
                "url": actual_url,
                "snippet": clean_html_text(snippet_raw)
            })
    elif err:
        print(f"[OSKY Search] Pencarian web untuk '{query}' tidak dapat dilakukan: {err}")

    return results[:max_results]
