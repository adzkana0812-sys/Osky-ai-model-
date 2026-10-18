"""
OSKY AI - Security, Authentication & Rate Limiting
Sistem keamanan mandiri milik OSKY AI server sendiri:
1. Pembuatan API Key lokal (format: osky_live_...)
2. Hashing API key menggunakan SHA-256 dan cryptographic salt acak
3. Verifikasi token aman tanpa menyimpan plaintext
4. In-memory sliding token-bucket rate limiter
5. Peringatan jelas: API key ini hanya untuk otentikasi ke server OSKY AI lokal Anda,
   TIDAK berhubungan dengan API key Google/OpenAI/Anthropic!
"""

import os
import secrets
import hashlib
import time
import json
from typing import Dict, Optional, Tuple, Any

from OSKY_AI.paths import resolve_path

class OskySecurityManager:
    def __init__(self, keys_file: str = "api/keys.json", default_rate_limit: int = 60):
        self.keys_file = resolve_path(keys_file)
        self.default_rate_limit = default_rate_limit  # requests per minute
        self.keys_db: Dict[str, Dict[str, Any]] = {}
        self.request_history: Dict[str, list] = {}  # key_id -> [timestamps]

        self._load_keys()

    def _hash_key(self, raw_key: str, salt: str) -> str:
        """Hash API key dengan SHA-256 dan salt."""
        return hashlib.sha256((salt + raw_key).encode("utf-8")).hexdigest()

    def generate_api_key(self, client_name: str, rate_limit: Optional[int] = None) -> Tuple[str, str]:
        """
        Menghasilkan API key baru dengan format aman: osky_live_<hex32>.
        Mengembalikan (raw_key, key_id). Raw key hanya ditampilkan sekali!
        """
        raw_token = secrets.token_hex(24)
        raw_key = f"osky_live_{raw_token}"
        salt = secrets.token_hex(16)
        key_hash = self._hash_key(raw_key, salt)
        key_id = f"key_{secrets.token_hex(8)}"

        self.keys_db[key_id] = {
            "client_name": client_name,
            "key_hash": key_hash,
            "salt": salt,
            "rate_limit": rate_limit or self.default_rate_limit,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "key_prefix": raw_key[:12] + "...",
        }
        self._save_keys()
        return raw_key, key_id

    def verify_api_key(self, raw_key: str) -> Optional[Dict[str, Any]]:
        """
        Memverifikasi API key yang dikirimkan klien.
        Mengembalikan info klien jika valid, None jika tidak valid.
        """
        if not raw_key or not raw_key.startswith("osky_live_"):
            return None

        for key_id, record in self.keys_db.items():
            salt = record["salt"]
            expected_hash = record["key_hash"]
            computed_hash = self._hash_key(raw_key, salt)

            if secrets.compare_digest(computed_hash, expected_hash):
                # Valid!
                return {
                    "key_id": key_id,
                    "client_name": record["client_name"],
                    "rate_limit": record["rate_limit"],
                }
        return None

    def check_rate_limit(self, key_id: str, limit: int) -> bool:
        """
        Sliding window rate limiter: memeriksa apakah klien melewati batas per menit.
        Mengembalikan True jika diperbolehkan, False jika kena limit.
        """
        now = time.time()
        one_minute_ago = now - 60.0

        if key_id not in self.request_history:
            self.request_history[key_id] = []

        # Bersihkan timestamp yang sudah lebih dari 60 detik
        self.request_history[key_id] = [t for t in self.request_history[key_id] if t > one_minute_ago]

        if len(self.request_history[key_id]) >= limit:
            return False

        self.request_history[key_id].append(now)
        return True

    def _save_keys(self):
        os.makedirs(os.path.dirname(self.keys_file), exist_ok=True)
        with open(self.keys_file, "w", encoding="utf-8") as f:
            json.dump(self.keys_db, f, indent=2)

    def _load_keys(self):
        if os.path.exists(self.keys_file):
            try:
                with open(self.keys_file, "r", encoding="utf-8") as f:
                    self.keys_db = json.load(f)
            except Exception:
                self.keys_db = {}
        else:
            # Buat default key otomatis untuk pengembang lokal jika file belum ada
            pass


if __name__ == "__main__":
    import sys
    sec = OskySecurityManager()
    name = sys.argv[1] if len(sys.argv) > 1 else "developer_admin"
    raw_key, key_id = sec.generate_api_key(name)
    print("\n" + "=" * 60)
    print("  OSKY AI - PEMBUATAN API KEY SERVER LOKAL")
    print("=" * 60)
    print(f"Client Name: {name}")
    print(f"Key ID     : {key_id}")
    print(f"API Key    : {raw_key}")
    print("\nSIMPAN API KEY INI DENGAN AMAN! Kunci ini tidak akan ditampilkan kembali.")
    print("Gunakan header berikut pada request ke server OSKY AI:")
    print(f"Authorization: Bearer {raw_key}")
    print("=" * 60 + "\n")
