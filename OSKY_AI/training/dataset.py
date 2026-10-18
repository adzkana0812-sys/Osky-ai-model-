"""
OSKY AI - Dataset Pipeline & Data Preprocessing
Fitur:
1. Mendukung format JSONL dan file teks TXT
2. Format percakapan:
   - {"instruction": "...", "response": "..."}
   - {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
3. Pembersihan (Cleaning), Deduplikasi Hash, dan Filtering
4. Train / Validation split dengan random seed yang dapat direproduksi
5. Masking Loss: Opsional hanya menghitung loss pada token respons asisten (SFT mode)
6. Dynamic Batching, Shuffling, dan Padding aman
"""

import json
import os
import hashlib
import random
import re
from typing import List, Dict, Any, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from OSKY_AI.tokenizer import OskyTokenizer

def clean_text(text: str) -> str:
    """Membersihkan teks dari karakter kontrol tak terlihat dan menormalkan spasi."""
    if not text:
        return ""
    # Hapus karakter kontrol (kecuali newline dan tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Normalkan multiple newlines dan spaces
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class ConversationRecord:
    def __init__(self, prompt: str, response: str):
        self.prompt = prompt
        self.response = response

    def to_formatted_text(self) -> str:
        return f"<user>{self.prompt}<assistant>{self.response}</s>"


class OskyDatasetPipeline:
    def __init__(
        self,
        tokenizer: OskyTokenizer,
        max_seq_len: int = 512,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.seed = seed

    def load_jsonl(self, file_path: str) -> List[ConversationRecord]:
        """Memuat dataset dari format JSONL percakapan."""
        records: List[ConversationRecord] = []
        if not os.path.exists(file_path):
            print(f"[Dataset Pipeline] Warning: File {file_path} tidak ditemukan.")
            return records

        seen_hashes = set()

        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    continue

                prompt = ""
                response = ""

                # Format 1: {"instruction": "...", "response": "..."}
                if "instruction" in obj and "response" in obj:
                    prompt = clean_text(str(obj["instruction"]))
                    response = clean_text(str(obj["response"]))

                # Format 2: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
                elif "messages" in obj and isinstance(obj["messages"], list):
                    user_msgs = []
                    asst_msgs = []
                    for m in obj["messages"]:
                        role = m.get("role")
                        content = clean_text(m.get("content", ""))
                        if role == "user":
                            user_msgs.append(content)
                        elif role == "assistant":
                            asst_msgs.append(content)
                    prompt = "\n".join(user_msgs)
                    response = "\n".join(asst_msgs)

                # Format 3: {"text": "..."} (Unstructured raw corpus)
                elif "text" in obj:
                    text_content = clean_text(str(obj["text"]))
                    prompt = ""
                    response = text_content

                if not response:
                    continue

                # Deduplikasi sederhana menggunakan MD5
                content_hash = hashlib.md5(f"{prompt}:::==={response}".encode("utf-8")).hexdigest()
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)

                records.append(ConversationRecord(prompt, response))

        print(f"[Dataset Pipeline] Berhasil memuat {len(records)} data unik dari {file_path}")
        return records

    def load_raw_txt(self, file_path: str) -> List[ConversationRecord]:
        """Memuat dokumen teks mentah dari file TXT."""
        records = []
        if not os.path.exists(file_path):
            return records

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        paragraphs = content.split("\n\n")
        seen_hashes = set()
        for p in paragraphs:
            cleaned = clean_text(p)
            if len(cleaned) < 10:
                continue
            h = hashlib.md5(cleaned.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            records.append(ConversationRecord(prompt="", response=cleaned))

        print(f"[Dataset Pipeline] Berhasil memuat {len(records)} paragraf teks dari {file_path}")
        return records

    def split_dataset(
        self, records: List[ConversationRecord], val_ratio: float = 0.1
    ) -> Tuple[List[ConversationRecord], List[ConversationRecord]]:
        """Membagi dataset menjadi train dan validation split yang dapat direproduksi."""
        rng = random.Random(self.seed)
        shuffled = list(records)
        rng.shuffle(shuffled)

        val_size = int(len(shuffled) * val_ratio)
        if val_size == 0 and len(shuffled) > 1:
            val_size = 1

        val_data = shuffled[:val_size]
        train_data = shuffled[val_size:]
        print(f"[Dataset Pipeline] Split selesai: {len(train_data)} train, {len(val_data)} validation")
        return train_data, val_data


class OskyTorchDataset(Dataset):
    """
    PyTorch Dataset untuk OSKY AI.
    Menghasilkan:
    - input_ids: token input [x_0, x_1, ..., x_{T-1}]
    - labels: target token untuk autoregressive prediction [x_1, x_2, ..., x_T]
    - mask_prompt: jika True, token prompt user di-mask dengan -100 agar loss difokuskan pada jawaban model
    """
    def __init__(
        self,
        records: List[ConversationRecord],
        tokenizer: OskyTokenizer,
        max_seq_len: int = 512,
        mask_prompt: bool = True
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mask_prompt = mask_prompt
        self.examples: List[Tuple[List[int], List[int]]] = []

        self._process_records()

    def _process_records(self):
        for rec in self.records:
            if rec.prompt:
                # Mode Percakapan: <user> prompt <assistant> response </s>
                prompt_text = f"<user>{rec.prompt}<assistant>"
                response_text = f"{rec.response}</s>"

                prompt_ids = self.tokenizer.encode(prompt_text, add_bos=True)
                response_ids = self.tokenizer.encode(response_text, add_eos=False)

                full_ids = prompt_ids + response_ids
                if len(full_ids) > self.max_seq_len + 1:
                    full_ids = full_ids[: self.max_seq_len + 1]

                input_ids = full_ids[:-1]
                target_ids = full_ids[1:]

                if self.mask_prompt:
                    # Ganti token target yang berada di bagian prompt dengan -100
                    # Panjang prompt di input_ids adalah len(prompt_ids) - 1
                    prompt_len = min(len(prompt_ids) - 1, len(target_ids))
                    masked_targets = [-100] * prompt_len + target_ids[prompt_len:]
                else:
                    masked_targets = target_ids

                if len(input_ids) >= 2:
                    self.examples.append((input_ids, masked_targets))
            else:
                # Mode Raw Text (Pretraining)
                full_ids = self.tokenizer.encode(rec.response, add_bos=True, add_eos=True)
                if len(full_ids) > self.max_seq_len + 1:
                    full_ids = full_ids[: self.max_seq_len + 1]

                input_ids = full_ids[:-1]
                target_ids = full_ids[1:]
                if len(input_ids) >= 2:
                    self.examples.append((input_ids, target_ids))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids, labels = self.examples[idx]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_fn(batch, pad_token_id: int = 0):
    """
    Collate function untuk DataLoader yang mem-pad sequence secara dinamis sesuai panjang terpanjang dalam batch.
    """
    input_tensors, label_tensors = zip(*batch)
    max_len = max(len(x) for x in input_tensors)

    padded_inputs = []
    padded_labels = []
    attention_masks = []

    for inp, lbl in zip(input_tensors, label_tensors):
        pad_len = max_len - len(inp)
        if pad_len > 0:
            padded_inp = torch.cat([inp, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            # Pad labels dengan -100 agar CrossEntropyLoss mengabaikannya
            padded_lbl = torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])
            mask = torch.cat([torch.ones(len(inp), dtype=torch.bool), torch.zeros(pad_len, dtype=torch.bool)])
        else:
            padded_inp = inp
            padded_lbl = lbl
            mask = torch.ones(len(inp), dtype=torch.bool)

        padded_inputs.append(padded_inp)
        padded_labels.append(padded_lbl)
        attention_masks.append(mask)

    return (
        torch.stack(padded_inputs),
        torch.stack(padded_labels),
        torch.stack(attention_masks),
    )
