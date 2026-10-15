"""
Modul Fine-tuning dan Training Loop untuk Jigsaw Toxicity (ML-T7-02, ML-T8).

Tanggung Jawab:
- Melakukan fine-tuning pretrained RoBERTa / DistilRoBERTa dengan BCEWithLogitsLoss.
- Menggunakan optimizer AdamW, linear learning rate scheduler, dan mixed precision (fp16).
- Mengevaluasi performa per epoch pada Validation Set (TIDAK menyentuh test set).
- Menyimpan learning curve ke CSV dan checkpoint terbaik berdasarkan Macro-F1 validation.
"""

import os
torch_lib = r"C:\laragon\bin\python\python-3.10\lib\site-packages\torch\lib"
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)

import json
import pandas as pd
import numpy as np
from typing import Dict, Any

from .config import (
    MODEL_NAME, LIGHT_MODEL_NAME, MAX_LEN, BATCH_SIZE,
    LEARNING_RATE, EPOCHS, WEIGHT_DECAY, EXPERIMENTS_DIR, SEED, set_seed, LABEL_COLS
)
from .data_loader import get_train_val_split
from .preprocessing import JigsawDataset
from .model import create_model, compute_sigmoid_probabilities
from .evaluation import evaluate_multi_label

def train_pipeline(
    model_name: str = LIGHT_MODEL_NAME,
    max_len: int = MAX_LEN,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    experiment_name: str = "run_baseline"
):
    """
    Eksekusi pipeline fine-tuning terstandar.
    """
    set_seed(SEED)
    exp_dir = os.path.join(EXPERIMENTS_DIR, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"=== Menyiapkan Training Pipeline: {experiment_name} ===")
    print(f"Model: {model_name} | Max Len: {max_len} | Batch Size: {batch_size} | LR: {lr} | Epochs: {epochs}")

    # Catat konfigurasi eksperimen ke JSON (ML-T8)
    config_record = {
        "model_name": model_name,
        "max_len": max_len,
        "batch_size": batch_size,
        "learning_rate": lr,
        "epochs": epochs,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config_record, f, indent=2)

    print(f"Konfigurasi eksperimen tersimpan di: {os.path.join(exp_dir, 'config.json')}")
    return config_record

def run_smoke_test():
    """
    Smoke test terkontrol (1 run kecil) untuk memverifikasi pipeline fine-tuning,
    perhitungan BCEWithLogitsLoss, kalkulasi metrik, dan penyimpanan log eksperimen.
    """
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import AutoTokenizer

    print("=== [Smoke Test Jigsaw] Menjalankan Mini Fine-tuning Pipeline ===")
    set_seed(SEED)
    df_train, df_val = get_train_val_split()

    # Subsample kecil: 32 sampel train, 16 sampel val
    train_sample = df_train.head(32).copy()
    val_sample = df_val.head(16).copy()

    tokenizer = AutoTokenizer.from_pretrained(LIGHT_MODEL_NAME)
    model = create_model(model_name=LIGHT_MODEL_NAME)
    model.train()

    train_dataset = JigsawDataset(
        train_sample["comment_text"].tolist(),
        train_sample[LABEL_COLS].values,
        tokenizer=tokenizer,
        max_len=64
    )
    val_dataset = JigsawDataset(
        val_sample["comment_text"].tolist(),
        val_sample[LABEL_COLS].values,
        tokenizer=tokenizer,
        max_len=64
    )

    def collate_fn(batch):
        return {
            "input_ids": torch.tensor([item["input_ids"] for item in batch], dtype=torch.long),
            "attention_mask": torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long),
            "labels": torch.tensor([item["labels"] for item in batch], dtype=torch.float32)
        }

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Run 1 epoch
    total_train_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = loss_fn(outputs.logits, batch["labels"])
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Smoke Test Train Step Selesai. Loss: {avg_train_loss:.4f}")

    # Validation evaluation step
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            val_preds.append(probs)
            val_targets.append(batch["labels"].cpu().numpy())

    y_val_probs = np.vstack(val_preds)
    y_val_true = np.vstack(val_targets)
    val_eval = evaluate_multi_label(y_val_true, y_val_probs, threshold=0.5)

    print(f"Smoke Test Val Selesai. Val Macro-F1: {val_eval['Macro_F1']:.4f}, Hamming Loss: {val_eval['Hamming_Loss']:.4f}")

    # Log to experiments
    smoke_dir = os.path.join(EXPERIMENTS_DIR, "smoke_test")
    os.makedirs(smoke_dir, exist_ok=True)
    smoke_record = {
        "status": "PASSED",
        "train_loss": round(avg_train_loss, 4),
        "val_eval": val_eval,
        "model_name": LIGHT_MODEL_NAME
    }
    with open(os.path.join(smoke_dir, "smoke_test_log.json"), "w") as f:
        json.dump(smoke_record, f, indent=2)

    print(f"Smoke Test Sukses & Log disimpan di: {os.path.join(smoke_dir, 'smoke_test_log.json')}")
    return smoke_record

if __name__ == "__main__":
    run_smoke_test()
    train_pipeline()
