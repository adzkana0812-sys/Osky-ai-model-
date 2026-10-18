"""
OSKY AI - Web Sources & Citation Manager
Menyimpan dan mengelola metadata sumber penelusuran web:
- URL
- Title
- Source Domain
- Retrieved Timestamp (UTC timezone-aware)
- Relevant Extracted Snippet
Memformat sitasi terstruktur untuk jawaban akhir OSKY AI.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from datetime import datetime, timezone

@dataclass
class WebSource:
    url: str
    title: str
    source_domain: str
    retrieved_time: str
    snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def format_citation(self, index: int) -> str:
        return f"[{index}] {self.title} — {self.source_domain} ({self.url})"

class SourceCollection:
    def __init__(self):
        self.sources: List[WebSource] = []

    def add(self, url: str, title: str, snippet: str) -> WebSource:
        domain = url.split("//")[-1].split("/")[0]
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        src = WebSource(
            url=url,
            title=title,
            source_domain=domain,
            retrieved_time=now,
            snippet=snippet
        )
        self.sources.append(src)
        return src

    def format_all_citations(self) -> str:
        if not self.sources:
            return "Tidak ada sumber eksternal yang dirujuk."
        lines = ["\nSumber Rujukan Web Terverifikasi:"]
        for i, s in enumerate(self.sources, 1):
            lines.append(s.format_citation(i))
        return "\n".join(lines)
