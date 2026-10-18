"""
OSKY AI - Local REST API Server
Menyediakan antarmuka HTTP REST API lokal untuk mengakses model OSKY AI.
Endpoints:
1. GET /health
2. POST /v1/chat/completions
   Request:
   {
     "model": "osky-ai",
     "messages": [{"role": "user", "content": "Halo"}],
     "temperature": 0.7,
     "max_tokens": 100
   }
   Response:
   {
     "model": "osky-ai",
     "response": "Halo! Saya OSKY AI.",
     "usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13}
   }

Keamanan:
- Autentikasi Bearer Token (API key milik server OSKY AI sendiri)
- Rate limiting per key
- Pembatasan ukuran request maksimal (1 MB)
- Proteksi CORS dan sanitasi input
"""

import os
import sys
import json
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from OSKY_AI.paths import resolve_path
from OSKY_AI.inference import OskyInferenceEngine
from OSKY_AI.api.auth import OskySecurityManager

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Server HTTP multi-threaded untuk melayani request bersamaan."""
    daemon_threads = True

class OskyAPIHandler(BaseHTTPRequestHandler):
    engine: Optional[OskyInferenceEngine] = None
    security: Optional[OskySecurityManager] = None
    max_payload_bytes: int = 1024 * 1024  # 1 MB max

    def _set_headers(self, status_code: int = 200, content_type: str = "application/json"):
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_OPTIONS(self):
        """Menangani pre-flight CORS requests."""
        self._set_headers(204)

    def _send_json_error(self, status_code: int, message: str, code: str = "error"):
        self._set_headers(status_code)
        resp = {
            "error": {
                "message": message,
                "type": code,
                "code": status_code,
            }
        }
        self.wfile.write(json.dumps(resp, ensure_ascii=False).encode("utf-8"))

    def _authenticate(self) -> Optional[Dict[str, Any]]:
        auth_header = self.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return None

        token = auth_header[7:].strip()
        client_info = self.security.verify_api_key(token)
        return client_info

    def do_GET(self):
        if self.path == "/health" or self.path == "/":
            self._set_headers(200)
            status_data = {
                "status": "healthy",
                "service": "OSKY AI Local Inference Server",
                "model": "osky-ai",
                "device": str(self.engine.device) if self.engine else "not_loaded",
                "model_parameters": self.engine.model.count_parameters()["human_readable"] if self.engine else "N/A",
                "engine_type": "PyTorch From-Scratch Neural Network",
            }
            self.wfile.write(json.dumps(status_data, indent=2).encode("utf-8"))
        else:
            self._send_json_error(404, f"Endpoint {self.path} tidak ditemukan.", "not_found")

    def do_POST(self):
        # 1. Cek Autentikasi
        client_info = self._authenticate()
        if not client_info:
            self._send_json_error(
                401,
                "Unauthorized: API key server OSKY AI tidak valid atau tidak disertakan pada header Authorization: Bearer osky_live_...",
                "invalid_api_key"
            )
            return

        # 2. Cek Rate Limiting
        if not self.security.check_rate_limit(client_info["key_id"], client_info["rate_limit"]):
            self._send_json_error(
                429,
                f"Rate limit exceeded: Batas {client_info['rate_limit']} request per menit terlampaui.",
                "rate_limit_exceeded"
            )
            return

        # 3. Validasi Content-Length
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > self.max_payload_bytes:
            self._send_json_error(413, "Payload too large: Ukuran request melebihi batas 1 MB.", "payload_too_large")
            return

        # 4. Baca dan parse JSON body
        body_bytes = self.rfile.read(content_length)
        try:
            body = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            self._send_json_error(400, "Bad Request: Format JSON tidak valid.", "invalid_json")
            return

        # 5. Routing endpoint
        if self.path == "/v1/chat/completions":
            self.handle_chat_completions(body)
        else:
            self._send_json_error(404, f"Endpoint {self.path} tidak ditemukan.", "not_found")

    def handle_chat_completions(self, body: Dict[str, Any]):
        messages = body.get("messages", [])
        if not messages or not isinstance(messages, list):
            self._send_json_error(400, "Parameter 'messages' wajib berupa list pesan percakapan.", "invalid_parameter")
            return

        temperature = float(body.get("temperature", 0.7))
        top_p = float(body.get("top_p", 0.9))
        top_k = int(body.get("top_k", 40))
        max_tokens = int(body.get("max_tokens", 100))

        # Format prompt dari riwayat pesan
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(f"<user>{content}")
            elif role == "assistant":
                prompt_parts.append(f"<assistant>{content}</s>")
        prompt_parts.append("<assistant>")
        full_prompt = "".join(prompt_parts)

        # Hitung prompt tokens
        prompt_ids = self.engine.tokenizer.encode(full_prompt, add_bos=True)

        # Jalankan inferensi neural network lokal
        try:
            response_text = self.engine.generate(
                prompt=full_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                is_chat=False,
            )
        except Exception as e:
            self._send_json_error(500, f"Inference engine error: {str(e)}", "internal_error")
            return

        completion_ids = self.engine.tokenizer.encode(response_text)

        response_payload = {
            "model": "osky-ai",
            "response": response_text,
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(completion_ids),
                "total_tokens": len(prompt_ids) + len(completion_ids),
            }
        }

        self._set_headers(200)
        self.wfile.write(json.dumps(response_payload, ensure_ascii=False).encode("utf-8"))

def run_server(
    checkpoint_path: str = "checkpoints/latest.pt",
    tokenizer_path: str = "tokenizer/vocab.json",
    host: str = "0.0.0.0",
    port: int = 8000,
):
    print("\n" + "=" * 60)
    print("  OSKY AI - LOCAL INFERENCE SERVER")
    print("=" * 60)

    # Inisialisasi engine & security
    chk_path = resolve_path(checkpoint_path)
    tok_path = resolve_path(tokenizer_path)
    engine = OskyInferenceEngine(chk_path, tok_path)
    security = OskySecurityManager()

    # Pastikan ada minimal 1 API key
    if not security.keys_db:
        key, kid = security.generate_api_key("default_admin")
        print(f"[API Security] Dibuat API key default:")
        print(f"  Key ID : {kid}")
        print(f"  API Key: {key}")
        print("  Gunakan header: 'Authorization: Bearer " + key + "'\n")

    OskyAPIHandler.engine = engine
    OskyAPIHandler.security = security

    server = ThreadedHTTPServer((host, port), OskyAPIHandler)
    print(f"[OSKY API] Server berjalan di http://{host}:{port}")
    print(f"[OSKY API] Endpoint: POST http://{host}:{port}/v1/chat/completions")
    print(f"[OSKY API] Status   : GET  http://{host}:{port}/health\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[OSKY API] Mematikan server...")
        server.server_close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OSKY AI Local REST API Server")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/vocab.json")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    run_server(args.checkpoint, args.tokenizer, args.host, args.port)
