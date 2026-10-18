"""
OSKY AI - Custom Local Tokenizer
Dibuat dari nol (from scratch) tanpa dependensi library eksternal (seperti Hugging Face tokenizers).
Mendukung character/byte-level tokenization dengan BPE (Byte-Pair Encoding) merge support,
penanganan unknown bytes yang aman, serta special tokens untuk percakapan chat.
"""

import os
import json
import re
import hashlib
from typing import List, Dict, Optional, Tuple, Set, Any

from OSKY_AI.paths import resolve_path

class OskyTokenizer:
    SPECIAL_TOKENS = {
        "<pad>": 0,
        "<unk>": 1,
        "<s>": 2,
        "</s>": 3,
        "<user>": 4,
        "<assistant>": 5,
        "<system>": 6,
    }

    def __init__(self, vocab_path: Optional[str] = None):
        self.special_tokens = dict(self.SPECIAL_TOKENS)
        self.pad_token_id = self.special_tokens["<pad>"]
        self.unk_token_id = self.special_tokens["<unk>"]
        self.bos_token_id = self.special_tokens["<s>"]
        self.eos_token_id = self.special_tokens["</s>"]
        self.user_token_id = self.special_tokens["<user>"]
        self.assistant_token_id = self.special_tokens["<assistant>"]
        self.system_token_id = self.special_tokens["<system>"]

        self.token_to_id: Dict[str, int] = dict(self.special_tokens)
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.special_tokens.items()}
        self.merges: List[Tuple[str, str]] = []

        # Inisialisasi 256 byte dasar sebagai jaminan byte fallback
        # Dengan ini, OSKY AI tidak akan pernah crash pada karakter Unicode apapun!
        for b in range(256):
            b_str = f"<byte_0x{b:02x}>"
            if b_str not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[b_str] = new_id
                self.id_to_token[new_id] = b_str

        if vocab_path:
            resolved_vpath = resolve_path(vocab_path)
            if os.path.exists(resolved_vpath):
                self.load(resolved_vpath)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def get_vocab_hash(self) -> str:
        """
        Menghasilkan hash SHA-256 deterministik dari vocabulary untuk memverifikasi
        kecocokan antara model checkpoint dan tokenizer.
        """
        sorted_pairs = sorted(self.token_to_id.items(), key=lambda x: x[1])
        vocab_str = "|".join(f"{idx}:{token}" for token, idx in sorted_pairs)
        return hashlib.sha256(vocab_str.encode("utf-8")).hexdigest()

    def get_metadata(self) -> Dict[str, Any]:
        """Mengembalikan metadata lengkap tokenizer."""
        return {
            "version": "1.0",
            "vocab_size": self.vocab_size,
            "vocab_hash": self.get_vocab_hash(),
            "special_tokens": self.special_tokens,
            "num_merges": len(self.merges),
        }

    def train_from_corpus(self, texts: List[str], target_vocab_size: int = 1024):
        """
        Melatih tokenizer dari kumpulan teks menggunakan algoritma Byte-Pair Encoding (BPE).
        """
        print(f"[OSKY Tokenizer] Melatih tokenizer dari {len(texts)} dokumen...")
        # Kumpulkan frekuensi karakter umum dari teks
        char_counts: Dict[str, int] = {}
        for text in texts:
            for ch in text:
                char_counts[ch] = char_counts.get(ch, 0) + 1

        # Tambahkan karakter unik ke vocabulary
        sorted_chars = sorted(char_counts.keys(), key=lambda c: -char_counts[c])
        for ch in sorted_chars:
            if len(self.token_to_id) >= target_vocab_size:
                break
            if ch not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[ch] = new_id
                self.id_to_token[new_id] = ch

        # Lakukan BPE merges untuk frekuensi pasangan tertinggi (subword building)
        # Pisahkan teks menjadi representasi list token
        word_freqs: Dict[Tuple[str, ...], int] = {}
        for text in texts:
            words = text.strip().split()
            for word in words:
                chars = tuple(list(word))
                if chars:
                    word_freqs[chars] = word_freqs.get(chars, 0) + 1

        while len(self.token_to_id) < target_vocab_size:
            pairs: Dict[Tuple[str, str], int] = {}
            for word_tuple, freq in word_freqs.items():
                for i in range(len(word_tuple) - 1):
                    pair = (word_tuple[i], word_tuple[i + 1])
                    pairs[pair] = pairs.get(pair, 0) + freq

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < 2:
                # Frekuensi terlalu rendah untuk di-merge
                break

            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[merged_token] = new_id
                self.id_to_token[new_id] = merged_token
                self.merges.append(best_pair)

            # Perbarui word_freqs
            new_word_freqs = {}
            for word_tuple, freq in word_freqs.items():
                new_word = []
                i = 0
                while i < len(word_tuple):
                    if i < len(word_tuple) - 1 and word_tuple[i] == best_pair[0] and word_tuple[i + 1] == best_pair[1]:
                        new_word.append(merged_token)
                        i += 2
                    else:
                        new_word.append(word_tuple[i])
                        i += 1
                new_word_freqs[tuple(new_word)] = freq
            word_freqs = new_word_freqs

        print(f"[OSKY Tokenizer] Selesai training. Total vocab_size: {self.vocab_size}, merges: {len(self.merges)}")

    def _encode_chunk(self, text: str) -> List[int]:
        """Encode sebuah kata atau potongan teks dengan menerapkan merges yang telah dipelajari."""
        # Jika teks persis ada di vocab
        if text in self.token_to_id:
            return [self.token_to_id[text]]

        # Mulai dari per-karakter
        tokens = list(text)
        for pair in self.merges:
            merged = pair[0] + pair[1]
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # Konversi token string ke ID dengan byte-fallback aman
        result_ids = []
        for t in tokens:
            if t in self.token_to_id:
                result_ids.append(self.token_to_id[t])
            else:
                # Byte fallback: encode ke UTF-8 bytes
                utf8_bytes = t.encode("utf-8")
                for b in utf8_bytes:
                    b_str = f"<byte_0x{b:02x}>"
                    result_ids.append(self.token_to_id.get(b_str, self.unk_token_id))
        return result_ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Mengonversi teks menjadi list token integer ID.
        Mendeteksi tag percakapan khusus seperti <user>, <assistant>, dll.
        """
        pattern = r"(<user>|<assistant>|<system>|<s>|</s>|<pad>|<unk>)"
        parts = re.split(pattern, text)

        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_token_id)

        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            else:
                # Teks biasa: proses per karakter/merge
                ids.extend(self._encode_chunk(part))

        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """
        Mengonversi list token integer ID kembali menjadi teks terbaca.
        Menggabungkan byte fallback menjadi string UTF-8 yang valid.
        """
        out_bytes = bytearray()
        result_chars: List[str] = []

        special_ids = set(self.special_tokens.values())

        for token_id in ids:
            if skip_special_tokens and token_id in special_ids:
                continue

            token_str = self.id_to_token.get(token_id, "<unk>")

            # Cek apakah token adalah byte fallback
            if token_str.startswith("<byte_0x") and token_str.endswith(">"):
                try:
                    hex_val = int(token_str[8:-1], 16)
                    out_bytes.append(hex_val)
                    continue
                except ValueError:
                    pass

            # Jika ada akumulasi byte sebelumnya, flush ke string
            if out_bytes:
                result_chars.append(out_bytes.decode("utf-8", errors="replace"))
                out_bytes.clear()

            result_chars.append(token_str)

        if out_bytes:
            result_chars.append(out_bytes.decode("utf-8", errors="replace"))

        return "".join(result_chars)

    def save(self, file_path: str):
        """Menyimpan vocabulary dan konfigurasi tokenizer ke file JSON."""
        resolved = resolve_path(file_path)
        dir_name = os.path.dirname(resolved)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        data = {
            "version": "1.0",
            "vocab_size": self.vocab_size,
            "vocab_hash": self.get_vocab_hash(),
            "special_tokens": self.special_tokens,
            "token_to_id": self.token_to_id,
            "merges": self.merges,
        }
        with open(resolved, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[OSKY Tokenizer] Vocabulary tersimpan di: {resolved} (hash: {self.get_vocab_hash()[:8]})")

    def load(self, file_path: str):
        """Memuat vocabulary dari file JSON."""
        resolved = resolve_path(file_path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"File tokenizer tidak ditemukan: {file_path} (resolved: {resolved})")
        with open(resolved, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.special_tokens = data.get("special_tokens", self.SPECIAL_TOKENS)
        self.token_to_id = data.get("token_to_id", {})
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
        self.merges = [tuple(m) for m in data.get("merges", [])]
        print(f"[OSKY Tokenizer] Berhasil memuat vocabulary ({self.vocab_size} tokens) dari {resolved}")
