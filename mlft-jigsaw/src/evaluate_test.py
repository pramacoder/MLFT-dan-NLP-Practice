"""
Modul Evaluasi Test Set untuk Jigsaw Toxicity (ML-T10, Tahap II).

PERHATIAN / ANTI-DATA LEAKAGE:
- Script ini HANYA boleh dieksekusi SATU KALI setelah hyperparameter terbaik terpilih
  dan threshold global dibekukan di experiments/final_config.json.
- Mengevaluasi 63.978 baris test set valid yang telah difilter dari label Kaggle -1.
- Menghasilkan laporan metrik final (Macro-F1, Micro-F1, per-label P/R/F1, Hamming loss).
- Menyimpan hasil prediksi untuk analisis error (ML-T11).
"""

import os
import sys

# Windows PyTorch DLL fix
torch_lib = r"C:\laragon\bin\python\python-3.10\lib\site-packages\torch\lib"
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import json
import time
import pandas as pd
import numpy as np
from typing import Dict, Any

from .config import (
    DATA_DIR, EXPERIMENTS_DIR, REPORTS_DIR, LABEL_COLS, NUM_LABELS,
    SEED, set_seed, LIGHT_MODEL_NAME
)
from .data_loader import load_test_data
from .model import create_model
from .evaluation import evaluate_multi_label

def dynamic_test_collate_fn(batch):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    attention_mask = []
    labels = []
    ids = []

    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [1] * pad_len)  # 1 = pad_token_id for RoBERTa
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        labels.append(item["labels"])
        ids.append(item["id"])

    return {
        "id": ids,
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(np.array(labels), dtype=torch.float32)
    }

class DynamicTestDataset:
    def __init__(self, ids, texts, labels, tok, m_len):
        self.ids = ids
        self.texts = texts
        self.labels = labels.astype(np.float32)
        self.tok = tok
        self.m_len = m_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(str(self.texts[idx]), truncation=True, max_length=self.m_len)
        return {
            "id": self.ids[idx],
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": self.labels[idx]
        }

def run_test_evaluation(batch_size: int = 64, max_test_samples: int = None) -> Dict[str, Any]:
    """
    Eksekusi evaluasi test set dengan konfigurasi model dan threshold yang dibekukan.
    Jika max_test_samples diset (misal 5000), mengevaluasi subset test untuk kecepatan komputasi CPU.
    Secara default mengevaluasi seluruh data test valid.
    """
    final_cfg_path = os.path.join(EXPERIMENTS_DIR, "final_config.json")
    if not os.path.exists(final_cfg_path):
        raise FileNotFoundError(f"final_config.json belum ditemukan di {final_cfg_path}. Jalankan hyperparameter exploration dulu.")

    with open(final_cfg_path, "r", encoding="utf-8") as f:
        final_cfg = json.load(f)

    best_exp = final_cfg["best_experiment"]
    best_thresh = final_cfg["frozen_threshold"]
    max_len = final_cfg["max_length"]
    model_name = final_cfg.get("model_name", LIGHT_MODEL_NAME)

    print("\n=======================================================")
    print(f"MEMULAI EVALUASI FINAL TEST SET (SATU KALI SAJA)")
    print(f"Model: {model_name} dari {best_exp}")
    print(f"Max Length: {max_len} | Frozen Global Threshold: {best_thresh}")
    print("=======================================================")

    # 1. Load Data Test (drop unscored Kaggle -1 rows)
    print("Memuat dataset test dan memfilter baris unscored (-1)...")
    df_test = load_test_data(drop_unscored=True)
    total_valid_test = len(df_test)
    print(f"Total baris test valid: {total_valid_test:,} baris.")

    if max_test_samples is not None and max_test_samples < total_valid_test:
        print(f"Menggunakan subset representatif {max_test_samples:,} baris test untuk efisiensi CPU.")
        df_test = df_test.sample(n=max_test_samples, random_state=SEED).reset_index(drop=True)

    # 2. Inisialisasi Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = create_model(model_name=model_name)

    checkpoint_path = os.path.join(EXPERIMENTS_DIR, best_exp, "best_model.pt")
    print(f"Memuat bobot checkpoint terbaik dari {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # 3. DataLoader
    test_ds = DynamicTestDataset(
        ids=df_test["id"].tolist(),
        texts=df_test["comment_text"].tolist(),
        labels=df_test[LABEL_COLS].values,
        tok=tokenizer,
        m_len=max_len
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=dynamic_test_collate_fn)

    # 4. Inferensi Test
    print(f"Memulai inferensi pada {len(test_ds):,} sampel (Batch Size: {batch_size})...")
    all_probs = []
    all_targets = []
    all_ids = []

    t0 = time.time()
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(batch["labels"].cpu().numpy())
            all_ids.extend(batch["id"])

            if (step + 1) % 50 == 0 or (step + 1) == len(test_loader):
                elapsed = time.time() - t0
                samples_done = min((step + 1) * batch_size, len(test_ds))
                print(f"Progress: {samples_done:,}/{len(test_ds):,} sampel ({elapsed:.1f}s)")

    y_test_probs = np.vstack(all_probs)
    y_test_true = np.vstack(all_targets)
    total_time = time.time() - t0

    # 5. Evaluasi Metrik
    print(f"\nMenghitung metrik performa multi-label pada threshold dibekukan {best_thresh}...")
    metrics_at_frozen = evaluate_multi_label(y_test_true, y_test_probs, threshold=best_thresh)
    # Juga hitung pada threshold baseline 0.5 untuk perbandingan
    metrics_at_05 = evaluate_multi_label(y_test_true, y_test_probs, threshold=0.5)

    test_results = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_experiment": best_exp,
        "frozen_threshold": best_thresh,
        "num_test_samples": len(df_test),
        "total_inference_time_sec": round(total_time, 2),
        "metrics_at_frozen_threshold": metrics_at_frozen,
        "metrics_at_default_05": metrics_at_05
    }

    results_file = os.path.join(EXPERIMENTS_DIR, "test_evaluation_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)

    print(f"\n=======================================================")
    print("HASIL EVALUASI FINAL TEST SET")
    print(f"Macro-F1 (Utama): {metrics_at_frozen['Macro_F1']:.4f} (vs 0.5: {metrics_at_05['Macro_F1']:.4f})")
    print(f"Micro-F1:        {metrics_at_frozen['Micro_F1']:.4f}")
    print(f"Macro-Precision: {metrics_at_frozen['Macro_Precision']:.4f}")
    print(f"Macro-Recall:    {metrics_at_frozen['Macro_Recall']:.4f}")
    print(f"Hamming Loss:    {metrics_at_frozen['Hamming_Loss']:.4f}")
    print("-------------------------------------------------------")
    print("Per-Label Performance (F1 / Precision / Recall / Support):")
    for col in LABEL_COLS:
        pl = metrics_at_frozen["Per_Label"][col]
        print(f"  {col:<15} F1: {pl['F1']:.4f} | P: {pl['Precision']:.4f} | R: {pl['Recall']:.4f} | Support: {pl['Support']:,}")
    print("=======================================================")
    print(f"Hasil lengkap disimpan ke: {results_file}")

    # 6. Simpan Prediksi Detail untuk Error Analysis
    pred_df = pd.DataFrame({
        "id": all_ids,
        "comment_text": df_test["comment_text"].values
    })
    for i, col in enumerate(LABEL_COLS):
        pred_df[f"true_{col}"] = y_test_true[:, i].astype(int)
        pred_df[f"prob_{col}"] = y_test_probs[:, i].astype(float)
        pred_df[f"pred_{col}"] = (y_test_probs[:, i] >= best_thresh).astype(int)

    preds_file = os.path.join(EXPERIMENTS_DIR, "test_predictions.csv")
    pred_df.to_csv(preds_file, index=False)
    print(f"Detail prediksi disimpan ke: {preds_file}")

    return test_results

if __name__ == "__main__":
    run_test_evaluation()
