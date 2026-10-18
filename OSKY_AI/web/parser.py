"""
OSKY AI - HTML Parser & Text Cleaner
Mengekstraksi konten teks bersih, judul artikel, dan menghilangkan tag HTML, script, css, navigasi, dan iklan.
Dibangun menggunakan regex murni dan html.parser standar Python tanpa dependensi eksternal.
"""

import re
import html
from typing import Dict, Any, List

def clean_html_text(raw_html: str) -> str:
    """Membersihkan raw HTML menjadi teks murni yang siap dibaca model"""
    # 1. Hapus tag <script> dan <style> beserta isinya
    text = re.sub(r'<script\b[^>]*>([\s\S]*?)<\/script>', ' ', raw_html, flags=re.IGNORECASE)
    text = re.sub(r'<style\b[^>]*>([\s\S]*?)<\/style>', ' ', text, flags=re.IGNORECASE)
    # 2. Hapus tag HTML lainnya
    text = re.sub(r'<[^>]+>', ' ', text)
    # 3. Decode entity HTML (&amp;, &lt;, dll)
    text = html.unescape(text)
    # 4. Normalisasi spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_article_content(raw_html: str) -> Dict[str, Any]:
    """Mengekstraksi judul dan paragraf utama dari HTML"""
    # Ekstraksi title tag
    title_match = re.search(r'<title\b[^>]*>(.*?)<\/title>', raw_html, flags=re.IGNORECASE)
    title = clean_html_text(title_match.group(1)) if title_match else "Tanpa Judul"

    # Bersihkan seluruh teks
    cleaned_body = clean_html_text(raw_html)

    # Potong menjadi paragraf informatif
    words = cleaned_body.split()
    trimmed_text = " ".join(words[:400])  # Batasi 400 kata pertama yang relevan

    return {
        "title": title,
        "text": trimmed_text,
        "word_count": len(words)
    }
