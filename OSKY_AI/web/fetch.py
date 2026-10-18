"""
OSKY AI - Secure Web Page Fetcher
Dilengkapi proteksi keamanan tingkat tinggi terhadap SSRF (Server-Side Request Forgery):
1. Memvalidasi skema (hanya http:// dan https://, menolak file://, ftp://, gopher://, dll).
2. Memeriksa resolusi IP DNS dan memblokir:
   - Loopback (127.0.0.0/8, ::1)
   - Private IP (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
   - Link-local (169.254.0.0/16, fe80::/10)
   - Cloud metadata IP (169.254.169.254, metadata.google.internal)
   - Unspecified & Multicast (0.0.0.0/8, 224.0.0.0/4, ::)
3. Mengontrol redirect dan memvalidasi URL tujuan redirect terhadap aturan SSRF yang sama.
4. Membatasi ukuran download maksimal (1 MB) untuk mencegah buffer exhaustion / Out-Of-Memory.
5. Timeout ketat untuk mencegah hanging connection.
"""

import socket
import ipaddress
import urllib.request
import urllib.error
import urllib.parse
from typing import Optional, Tuple

USER_AGENT = "Mozilla/5.0 (compatible; OSKY-AI-Researcher/1.0; +https://osky.ai)"
MAX_RESPONSE_BYTES = 1024 * 1024  # 1 MB max payload
DEFAULT_TIMEOUT = 3.0

def is_ip_allowed(ip_str: str) -> bool:
    """
    Memeriksa apakah alamat IP aman untuk diakses (bukan private, loopback, link-local, dsb).
    """
    try:
        ip = ipaddress.ip_address(ip_str)
        if ip.is_loopback:
            return False
        if ip.is_private:
            return False
        if ip.is_link_local:
            return False
        if ip.is_multicast:
            return False
        if ip.is_unspecified:
            return False
        if ip.is_reserved:
            return False

        # Cek AWS/GCP/Azure link-local metadata address secara eksplisit
        if ip_str == "169.254.169.254":
            return False

        return True
    except ValueError:
        return False

def validate_url_security(url: str) -> Tuple[bool, Optional[str]]:
    """
    Memvalidasi URL terhadap aturan keamanan SSRF dan skema protokol.
    Mengembalikan (is_safe, error_message).
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as e:
        return False, f"Format URL tidak valid: {str(e)}"

    if parsed.scheme.lower() not in ("http", "https"):
        return False, f"Protokol '{parsed.scheme}' tidak diizinkan. Hanya http dan https yang diperbolehkan."

    hostname = parsed.hostname
    if not hostname:
        return False, "URL tidak memiliki hostname yang valid."

    # Blokir hostname metadata eksplisit
    blocked_hostnames = {
        "localhost",
        "metadata.google.internal",
        "instance-data",
        "169.254.169.254"
    }
    if hostname.lower() in blocked_hostnames:
        return False, f"Akses ke host internal/metadata '{hostname}' diblokir demi keamanan."

    # Resolusi DNS untuk memeriksa semua IP yang dihasilkan
    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        return False, f"Gagal menyelesaikan DNS hostname '{hostname}': {str(e)}"

    resolved_ips = {item[4][0] for item in addr_info}
    for ip_str in resolved_ips:
        if not is_ip_allowed(ip_str):
            return False, f"Akses ke IP non-publik '{ip_str}' untuk host '{hostname}' diblokir (SSRF protection)."

    return True, None

class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Handler redirect yang memvalidasi target URL redirect terhadap SSRF."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        is_safe, err = validate_url_security(newurl)
        if not is_safe:
            raise urllib.error.HTTPError(newurl, 403, f"SSRF Blocked Redirect: {err}", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)

def fetch_url(url: str, timeout: float = DEFAULT_TIMEOUT, max_bytes: int = MAX_RESPONSE_BYTES) -> Tuple[Optional[str], Optional[str]]:
    """
    Mengunduh isi halaman web publik secara aman dengan proteksi SSRF.
    Mengembalikan (html_content, error_message).
    """
    # 1. Validasi SSRF awal
    is_safe, err = validate_url_security(url)
    if not is_safe:
        return None, f"Keamanan: {err}"

    # 2. Bangun opener dengan custom SafeRedirectHandler
    opener = urllib.request.build_opener(SafeRedirectHandler())
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/plain"
        }
    )

    try:
        with opener.open(req, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return None, f"Tipe konten bukan teks/html: {content_type}"

            # 3. Baca data bertahap dengan batas maksimum ukuran
            chunks = []
            total_read = 0
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                total_read += len(chunk)
                if total_read > max_bytes:
                    return None, f"Ukuran respons melebihi batas keamanan {max_bytes} bytes."
                chunks.append(chunk)

            raw_data = b"".join(chunks)

            # Decode encoding dengan fallback utf-8 / latin-1
            try:
                html_text = raw_data.decode("utf-8")
            except UnicodeDecodeError:
                html_text = raw_data.decode("latin-1", errors="replace")

            return html_text, None

    except urllib.error.HTTPError as e:
        return None, f"HTTP Error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return None, f"URL Error: {e.reason}"
    except Exception as e:
        return None, f"Kesalahan koneksi: {str(e)}"
