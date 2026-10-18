"""
OSKY AI - Neural Network Handwriting & OCR Architecture
Mengekspos modul pembacaan tulisan tangan & karakter (CRNN, CTC Loss, Karakter Vocabulary).
"""

from OSKY_AI.handwriting.model import OskyHandwritingRecognizer
from OSKY_AI.handwriting.dataset import HandwritingCharVocabulary, OskyHandwritingDataset

__all__ = [
    "OskyHandwritingRecognizer",
    "HandwritingCharVocabulary",
    "OskyHandwritingDataset"
]
