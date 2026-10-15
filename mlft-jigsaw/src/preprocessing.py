"""
Modul Preprocessing untuk Jigsaw Toxicity (ML-T7-01).

Tanggung Jawab:
- Menerapkan tokenisasi dinamis menggunakan Pretrained Tokenizer (RoBERTa / DistilRoBERTa).
- Mempertahankan teks asli (TIDAK melakukan stemming, stop-word removal, atau pembuangan tanda baca)
  agar representasi kontekstual pre-trained transformer tidak rusak.
- Mengonversi 6 kolom label menjadi tensor multi-hot float32.
- Menyediakan PyTorch Dataset class untuk integrasi DataLoader / Hugging Face Trainer.
"""

from typing import List, Dict, Any, Union
import numpy as np
import pandas as pd
from .config import LABEL_COLS, MAX_LEN

class JigsawDataset:
    """
    Dataset wrapper untuk Jigsaw Toxic Comment Classification.
    Mengubah teks komentar menjadi token input_ids & attention_mask serta
    target multi-hot 6 dimensi float32.
    Mendukung pemotongan standar maupun Head + Tail Truncation.
    """
    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray = None,
        tokenizer: Any = None,
        max_len: int = MAX_LEN,
        head_tail: bool = False
    ):
        self.texts = [str(t) if pd.notna(t) else "" for t in texts]
        self.labels = labels.astype(np.float32) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.head_tail = head_tail

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        
        if self.head_tail and self.tokenizer is not None:
            # Tokenisasi Head + Tail Truncation: 
            # Mengambil separuh awal dan separuh akhir untuk mempertahankan sinyal penutup
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            max_inner = self.max_len - 2
            if len(tokens) > max_inner:
                half = max_inner // 2
                tokens = tokens[:half] + tokens[-half:]
            input_ids = [self.tokenizer.cls_token_id] + tokens + [self.tokenizer.sep_token_id]
            pad_len = max(0, self.max_len - len(input_ids))
            attention_mask = [1] * len(input_ids) + [0] * pad_len
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        else:
            # Tokenisasi dinamis standar
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors=None
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        if self.labels is not None:
            item["labels"] = self.labels[idx]

        return item

