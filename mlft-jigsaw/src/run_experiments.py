import os
import sys

# Windows PyTorch DLL fix
torch_lib = r"C:\laragon\bin\python\python-3.10\lib\site-packages\torch\lib"
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from transformers import AutoTokenizer

from .config import (
    DATA_DIR, EXPERIMENTS_DIR, REPORTS_DIR, LABEL_COLS, NUM_LABELS,
    SEED, set_seed, LIGHT_MODEL_NAME
)
from .data_loader import load_train_data, load_test_data, get_train_val_split
from .preprocessing import JigsawDataset
from .model import create_model, compute_sigmoid_probabilities
from .evaluation import evaluate_multi_label, find_best_threshold_on_validation

def prepare_stratified_subsample(
    train_size: int = 1000,
    val_size: int = 300,
    seed: int = SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Menghasilkan subsample terstratifikasi multi-label yang terdokumentasi (ML-T4-04).
    Mempertahankan proporsi seluruh 6 label sesuai distribusi data penuh.
    """
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    df_train_full, df_val_full = get_train_val_split()

    # Stratified sample train
    N_tr = len(df_train_full)
    msss_tr = MultilabelStratifiedShuffleSplit(n_splits=1, train_size=train_size, test_size=N_tr - train_size, random_state=seed)
    X_tr = df_train_full["comment_text"].values
    Y_tr = df_train_full[LABEL_COLS].values
    tr_idx, _ = next(msss_tr.split(X_tr, Y_tr))
    df_train_sub = df_train_full.iloc[tr_idx].reset_index(drop=True)

    # Stratified sample val
    N_vl = len(df_val_full)
    msss_vl = MultilabelStratifiedShuffleSplit(n_splits=1, train_size=val_size, test_size=N_vl - val_size, random_state=seed)
    X_vl = df_val_full["comment_text"].values
    Y_vl = df_val_full[LABEL_COLS].values
    vl_idx, _ = next(msss_vl.split(X_vl, Y_vl))
    df_val_sub = df_val_full.iloc[vl_idx].reset_index(drop=True)

    meta = {
        "method": "MultilabelStratifiedShuffleSplit",
        "seed": seed,
        "train_size": train_size,
        "val_size": val_size,
        "train_distribution": {col: round(float(df_train_sub[col].mean() * 100), 3) for col in LABEL_COLS},
        "val_distribution": {col: round(float(df_val_sub[col].mean() * 100), 3) for col in LABEL_COLS}
    }

    subsample_path = os.path.join(DATA_DIR, "subsample_split.json")
    with open(subsample_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Subsample terstratifikasi berhasil dibentuk (Train: {train_size}, Val: {val_size}).")
    print(f"Distribusi label train: {meta['train_distribution']}")
    return df_train_sub, df_val_sub, meta

def dynamic_collate_fn(batch):
    """
    Dynamic padding ke panjang maksimum dalam batch untuk menghemat komputasi CPU.
    """
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [1] * pad_len)  # 1 = pad_token_id for RoBERTa
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        labels.append(item["labels"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(np.array(labels), dtype=torch.float32)
    }

def train_one_experiment(
    exp_name: str,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    tokenizer,
    max_len: int = 128,
    lr: float = 3e-5,
    epochs: int = 2,
    batch_size: int = 16,
    use_pos_weight: bool = False
) -> Dict[str, Any]:
    """
    Menjalankan 1 eksperimen fine-tuning terkontrol dan mencatat learning curve.
    """
    set_seed(SEED)
    exp_dir = os.path.join(EXPERIMENTS_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"Memulai Eksperimen: {exp_name}")
    print(f"Params: max_len={max_len}, lr={lr}, epochs={epochs}, batch={batch_size}, pos_weight={use_pos_weight}")
    print(f"=======================================================")

    # Inisialisasi model
    model = create_model(model_name=LIGHT_MODEL_NAME)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Class weighting pos_weight
    if use_pos_weight:
        pos_counts = df_train[LABEL_COLS].sum().values
        neg_counts = len(df_train) - pos_counts
        # Diredam dengan akar kuadrat agar stabil pada label ekstrem threat
        weights = np.sqrt(neg_counts / np.maximum(pos_counts, 1))
        pos_weight_tensor = torch.tensor(weights, dtype=torch.float32)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        print(f"Menggunakan pos_weight teredam: {dict(zip(LABEL_COLS, np.round(weights, 2)))}")
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    # Siapkan dataset tanpa padding statis (padding dinamis di DataLoader)
    class DynamicPaddingDataset:
        def __init__(self, texts, labels, tok, m_len):
            self.texts = texts
            self.labels = labels.astype(np.float32)
            self.tok = tok
            self.m_len = m_len
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            enc = self.tok(str(self.texts[idx]), truncation=True, max_length=self.m_len)
            return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": self.labels[idx]}

    train_ds = DynamicPaddingDataset(df_train["comment_text"].tolist(), df_train[LABEL_COLS].values, tokenizer, max_len)
    val_ds = DynamicPaddingDataset(df_val["comment_text"].tolist(), df_val[LABEL_COLS].values, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=dynamic_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=dynamic_collate_fn)

    history = []
    best_macro_f1 = -1.0
    best_model_state = None
    best_val_eval = None
    val_probs_best = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = loss_fn(outputs.logits, batch["labels"])
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Evaluasi Validation
        model.eval()
        total_val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                v_loss = loss_fn(outputs.logits, batch["labels"])
                total_val_loss += v_loss.item()
                probs = torch.sigmoid(outputs.logits).cpu().numpy()
                val_preds.append(probs)
                val_targets.append(batch["labels"].cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        y_val_probs = np.vstack(val_preds)
        y_val_true = np.vstack(val_targets)

        # Evaluasi threshold baseline 0.5
        eval_metrics = evaluate_multi_label(y_val_true, y_val_probs, threshold=0.5)
        macro_f1 = eval_metrics["Macro_F1"]
        epoch_time = time.time() - t0

        log_entry = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_macro_f1": round(macro_f1, 4),
            "val_micro_f1": round(eval_metrics["Micro_F1"], 4),
            "val_hamming_loss": round(eval_metrics["Hamming_Loss"], 4),
            "epoch_time_sec": round(epoch_time, 1)
        }
        history.append(log_entry)

        print(f"Epoch {epoch}/{epochs} ({epoch_time:.1f}s) | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Macro-F1: {macro_f1:.4f} | Micro-F1: {eval_metrics['Micro_F1']:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_val_eval = eval_metrics
            best_model_state = model.state_dict()
            val_probs_best = y_val_probs

    # Simpan log ke CSV
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(exp_dir, "history.csv"), index=False)

    # Simpan bobot checkpoint terbaik
    checkpoint_path = os.path.join(exp_dir, "best_model.pt")
    torch.save(best_model_state, checkpoint_path)

    # Diagnosis overfitting / normal
    train_loss_diff = history[-1]["train_loss"] - history[0]["train_loss"]
    val_loss_diff = history[-1]["val_loss"] - history[0]["val_loss"]
    if val_loss_diff > 0.05 and train_loss_diff < -0.05:
        diagnosis = "Overfitting terdeteksi (val loss naik sementara train loss turun)"
    elif train_loss_diff < 0 and val_loss_diff <= 0.02:
        diagnosis = "Normal convergence (keduanya stabil menurun)"
    else:
        diagnosis = "Underfitting / proses konvergensi awal stabil"

    exp_summary = {
        "exp_name": exp_name,
        "max_len": max_len,
        "learning_rate": lr,
        "epochs": epochs,
        "use_pos_weight": use_pos_weight,
        "best_macro_f1": round(best_macro_f1, 4),
        "best_val_eval": best_val_eval,
        "diagnosis": diagnosis,
        "history": history
    }
    with open(os.path.join(exp_dir, "summary.json"), "w") as f:
        json.dump(exp_summary, f, indent=2)

    return exp_summary, val_probs_best, y_val_true

def run_all_hyperparameter_experiments():
    """
    Menjalankan rangkaian 4 eksperimen hyperparameter terencana (ML-T8):
    1. Baseline: max_len=128, lr=3e-5, Standard BCE
    2. Panjang Token: max_len=256 (Menguji penanganan truncation label langka)
    3. Learning Rate: lr=5e-5
    4. Class Weighting: pos_weight teredam
    """
    df_train_sub, df_val_sub, meta = prepare_stratified_subsample(train_size=1000, val_size=300)
    tokenizer = AutoTokenizer.from_pretrained(LIGHT_MODEL_NAME)

    experiments_to_run = [
        {"name": "exp1_baseline", "max_len": 128, "lr": 3e-5, "pos_weight": False},
        {"name": "exp2_maxlen_256", "max_len": 256, "lr": 3e-5, "pos_weight": False},
        {"name": "exp3_lr_5e5", "max_len": 128, "lr": 5e-5, "pos_weight": False},
        {"name": "exp4_pos_weight", "max_len": 128, "lr": 3e-5, "pos_weight": True},
    ]

    all_summaries = []
    best_overall_macro = -1.0
    best_overall_exp = None
    best_val_probs = None
    y_val_true_saved = None

    for cfg in experiments_to_run:
        summary, val_probs, y_val_true = train_one_experiment(
            exp_name=cfg["name"],
            df_train=df_train_sub,
            df_val=df_val_sub,
            tokenizer=tokenizer,
            max_len=cfg["max_len"],
            lr=cfg["lr"],
            epochs=2,
            batch_size=16,
            use_pos_weight=cfg["pos_weight"]
        )
        all_summaries.append(summary)

        if summary["best_macro_f1"] > best_overall_macro:
            best_overall_macro = summary["best_macro_f1"]
            best_overall_exp = summary
            best_val_probs = val_probs
            y_val_true_saved = y_val_true

    # Simpan perbandingan ke laporan figures
    figures_dir = os.path.join(REPORTS_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Plot learning curve perbandingan
    plt.figure(figsize=(10, 5))
    for s in all_summaries:
        epochs = [h["epoch"] for h in s["history"]]
        val_losses = [h["val_loss"] for h in s["history"]]
        plt.plot(epochs, val_losses, marker="o", label=f"{s['exp_name']} (Macro-F1: {s['best_macro_f1']})")
    plt.title("Perbandingan Kurva Validation Loss antar Eksperimen Hyperparameter")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss (BCE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "learning_curves_hyperparameters.png"), dpi=200)
    plt.close()

    print(f"\n=======================================================")
    print(f"Eksperimen Terbaik Terpilih: {best_overall_exp['exp_name']} (Macro-F1: {best_overall_macro:.4f})")
    print(f"Diagnosis: {best_overall_exp['diagnosis']}")
    print(f"=======================================================")

    # =======================================================
    # ML-T9: Threshold Sweep pada Validation Split
    # =======================================================
    best_thresh, thresh_eval = find_best_threshold_on_validation(
        y_val_true_saved,
        best_val_probs,
        candidate_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]
    )

    # Bekukan keputusan ke final_config.json (ML-T9-03)
    final_config = {
        "best_experiment": best_overall_exp["exp_name"],
        "model_name": LIGHT_MODEL_NAME,
        "max_length": best_overall_exp["max_len"],
        "learning_rate": best_overall_exp["learning_rate"],
        "epochs": best_overall_exp["epochs"],
        "use_pos_weight": best_overall_exp["use_pos_weight"],
        "frozen_threshold": best_thresh,
        "val_macro_f1_with_threshold": thresh_eval["Macro_F1"],
        "val_metrics_at_best_threshold": thresh_eval,
        "seed": SEED,
        "status": "FROZEN_FOR_TEST_EVALUATION"
    }

    final_cfg_path = os.path.join(EXPERIMENTS_DIR, "final_config.json")
    with open(final_cfg_path, "w", encoding="utf-8") as f:
        json.dump(final_config, f, indent=2)
    print(f"Keputusan final DIBEKUKAN ke: {final_cfg_path}")

    # Simpan ringkasan seluruh eksperimen
    with open(os.path.join(EXPERIMENTS_DIR, "hyperparameter_exploration_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    return final_config

if __name__ == "__main__":
    run_all_hyperparameter_experiments()
