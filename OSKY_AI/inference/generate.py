"""
OSKY AI - Local Inference Engine
Menjalankan inferensi lokal secara mandiri dari weights OSKY AI sendiri.
Fitur Sampling:
1. Temperature (suhu probabilitas)
2. Top-K filtering
3. Top-P (Nucleus) sampling
4. Repetition Penalty (mencegah perulangan kata yang monoton)
5. Maximum new tokens limit
6. Stop tokens / EOS detection
7. Multi-turn conversation context manager dengan sliding window
TIDAK memanggil Gemini, OpenAI, Claude, atau API eksternal apapun!
"""

import os
import sys
import argparse
import json
import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from OSKY_AI.paths import resolve_path
from OSKY_AI.model import OskyLanguageModel
from OSKY_AI.tokenizer import OskyTokenizer

class OskyInferenceEngine:
    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str = "tokenizer/vocab.json",
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        resolved_tok = resolve_path(tokenizer_path)
        resolved_chk = resolve_path(checkpoint_path)

        # 1. Muat Tokenizer
        if not os.path.exists(resolved_tok):
            raise FileNotFoundError(f"File tokenizer tidak ditemukan di {tokenizer_path} (resolved: {resolved_tok})")
        self.tokenizer = OskyTokenizer(resolved_tok)

        # 2. Muat Checkpoint
        if not os.path.exists(resolved_chk):
            raise FileNotFoundError(f"File checkpoint tidak ditemukan di {checkpoint_path} (resolved: {resolved_chk})")

        print(f"[OSKY Inference] Memuat model dari {resolved_chk} ke {self.device}...")
        checkpoint = torch.load(resolved_chk, map_location=self.device)
        self.config = checkpoint.get("config", {})

        # 3. Inisialisasi Model dengan konfigurasi yang sama
        self.model = OskyLanguageModel(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print(f"[OSKY Inference] Model siap ({self.model.count_parameters()['human_readable']})")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        min_p: float = 0.05,
        repetition_penalty: float = 1.15,
        stop_tokens: Optional[List[str]] = None,
        is_chat: bool = True,
    ) -> str:
        """
        Menghasilkan kelanjutan teks berdasarkan prompt yang diberikan.
        """
        if is_chat and not prompt.startswith("<user>"):
            formatted_prompt = f"<user>{prompt}<assistant>"
        else:
            formatted_prompt = prompt

        input_ids = self.tokenizer.encode(formatted_prompt, add_bos=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        stop_token_ids = {self.tokenizer.eos_token_id}
        if stop_tokens:
            for st in stop_tokens:
                st_ids = self.tokenizer.encode(st)
                if st_ids:
                    stop_token_ids.add(st_ids[0])

        generated_ids: List[int] = []

        for _ in range(max_new_tokens):
            # Jika sequence melebihi max_seq_len, potong bagian awal (sliding context window)
            if input_tensor.shape[1] >= self.model.max_seq_len:
                input_tensor = input_tensor[:, -self.model.max_seq_len + 1:]

            # Forward pass melalui neural network OSKY AI
            _, logits = self.model(input_tensor)

            # Ambil logits token terakhir: [1, vocab_size]
            next_token_logits = logits[0, -1, :].clone()

            # 1. Repetition Penalty
            if repetition_penalty != 1.0:
                for token_id in set(input_ids + generated_ids):
                    if next_token_logits[token_id] > 0:
                        next_token_logits[token_id] /= repetition_penalty
                    else:
                        next_token_logits[token_id] *= repetition_penalty

            # 2. Temperature scaling
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            else:
                # Greedy search jika temperature == 0
                next_token = torch.argmax(next_token_logits).item()
                if next_token in stop_token_ids:
                    break
                generated_ids.append(next_token)
                input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=self.device)], dim=1)
                continue

            # 3. Top-K filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))[0][..., -1, None]
                next_token_logits[indices_to_remove] = float("-inf")

            # 4. Top-P (Nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                # Pertahankan setidaknya 1 token pertama
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float("-inf")

            # 5. Min-P Sampling (Menyaring token yang probabilitasnya di bawah min_p * probabilitas token teratas)
            if min_p > 0.0:
                step_probs = F.softmax(next_token_logits, dim=-1)
                max_prob = torch.max(step_probs).item()
                scaled_min_p = min_p * max_prob
                next_token_logits[step_probs < scaled_min_p] = float("-inf")

            # 6. Softmax untuk menghasilkan probabilitas multinomial
            probs = F.softmax(next_token_logits, dim=-1)

            # Sampling next token
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token in stop_token_ids:
                break

            generated_ids.append(next_token)
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=self.device)], dim=1)

        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return output_text.strip()


class OskyChatSession:
    """
    Mengelola konteks percakapan multi-turn untuk chatbot OSKY AI.
    Menangani sliding context window secara transparan.
    """
    def __init__(self, engine: OskyInferenceEngine, max_context_turns: int = 5):
        self.engine = engine
        self.max_context_turns = max_context_turns
        self.history: List[Dict[str, str]] = []

    def send_message(self, message: str, **kwargs) -> str:
        self.history.append({"role": "user", "content": message})

        # Ambil maksimal N putaran terakhir agar tidak melebihi context window
        recent_history = self.history[-(self.max_context_turns * 2):]

        # Bangun prompt multi-turn
        prompt_parts = []
        for msg in recent_history:
            if msg["role"] == "user":
                prompt_parts.append(f"<user>{msg['content']}")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"<assistant>{msg['content']}</s>")
        prompt_parts.append("<assistant>")
        full_prompt = "".join(prompt_parts)

        response = self.engine.generate(full_prompt, is_chat=False, **kwargs)
        self.history.append({"role": "assistant", "content": response})
        return response

    def reset(self):
        self.history.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OSKY AI Local Inference CLI")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt", help="Path ke model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/vocab.json", help="Path ke file tokenizer")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt teks untuk inferensi tunggal")
    parser.add_argument("--interactive", action="store_true", help="Jalankan sesi percakapan interaktif")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--max_tokens", type=int, default=100, help="Jumlah token maksimum yang digenerate")
    args = parser.parse_args()

    engine = OskyInferenceEngine(args.checkpoint, args.tokenizer)

    if args.interactive:
        session = OskyChatSession(engine)
        print("\n" + "=" * 60)
        print("  OSKY AI - Sesi Chat Interaktif Lokal")
        print("  (Ketik 'exit' atau 'keluar' untuk mengakhiri)")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("User: ")
                if user_input.strip().lower() in ["exit", "keluar", "quit"]:
                    break
                if not user_input.strip():
                    continue
                response = session.send_message(user_input, temperature=args.temp, top_p=args.top_p, max_new_tokens=args.max_tokens)
                print(f"OSKY AI: {response}\n")
            except KeyboardInterrupt:
                break
    else:
        prompt = args.prompt or "Siapa kamu?"
        print(f"User:\n{prompt}\n")
        reply = engine.generate(prompt, temperature=args.temp, top_p=args.top_p, max_new_tokens=args.max_tokens)
        print(f"OSKY AI:\n{reply}\n")
