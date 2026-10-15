"""
Pipeline Eksekusi Perbaikan Tahap III / v2 untuk Jigsaw Toxicity (ML-Aksi 1.1 & 1.2).

Tanggung Jawab:
1. Melatih eksperimen exp5_head_tail dengan strategi Head + Tail Truncation (@128 token)
   dan class weighting (pos_weight).
2. Melakukan optimasi threshold adaptif per-label pada Validation Split.
3. Membekukan konfigurasi v2 ke experiments/final_config_v2.json (TIDAK menimpa v1).
4. Menjalankan evaluasi Test set terisolasi ke experiments/test_evaluation_results_v2.json
   dan experiments/test_predictions_v2.csv (TIDAK menimpa v1).
5. Menganalisis penurunan False Negative pada teks panjang dan menyimpan ke
   experiments/error_analysis_summary_v2.json.
6. Menghasilkan grafik perbandingan visual Before vs After di reports/figures/.
"""

import os
import sys

# Windows PyTorch DLL fix
torch_lib = r"C:\laragon\bin\python\python-3.10\lib\site-packages\torch\lib"
if os.path.exists(torch_lib):
    os.environ["PATH"] = torch_lib + ";" + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(torch_lib)
    except Exception:
        pass
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# IMPORT TORCH HARUS PALING PERTAMA SEBELUM MATPLOTLIB/NUMPY
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer


from .config import (
    DATA_DIR, EXPERIMENTS_DIR, REPORTS_DIR, LABEL_COLS, NUM_LABELS,
    SEED, set_seed, LIGHT_MODEL_NAME
)
from .data_loader import load_test_data
from .model import create_model
from .evaluation import (
    evaluate_multi_label,
    find_best_threshold_on_validation,
    find_best_per_label_thresholds_on_validation
)
from .run_experiments import prepare_stratified_subsample

def collate_head_tail(batch):
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
        if "id" in item:
            ids.append(item["id"])

    res = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(np.array(labels), dtype=torch.float32)
    }
    if ids:
        res["id"] = ids
    return res

class HeadTailDataset:
    def __init__(self, texts, labels, tok, m_len=128, ids=None):
        self.texts = texts
        self.labels = labels.astype(np.float32) if labels is not None else None
        self.tok = tok
        self.m_len = m_len
        self.ids = ids

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        tokens = self.tok.encode(text, add_special_tokens=False)
        max_inner = self.m_len - 2
        if len(tokens) > max_inner:
            half = max_inner // 2
            tokens = tokens[:half] + tokens[-half:]
        input_ids = [self.tok.cls_token_id] + tokens + [self.tok.sep_token_id]
        attention_mask = [1] * len(input_ids)
        
        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": self.labels[idx] if self.labels is not None else None
        }
        if self.ids is not None:
            item["id"] = self.ids[idx]
        return item

def run_pipeline_v2():
    set_seed(SEED)
    figures_dir = os.path.join(REPORTS_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    print("==================================================================")
    print("MEMULAI PIPELINE PERBAIKAN v2 (HEAD-TAIL TRUNCATION & PER-LABEL THRESHOLDS)")
    print("==================================================================")

    # 1. Siapkan Subsample Terstratifikasi yang Identik
    df_train_sub, df_val_sub, meta = prepare_stratified_subsample(train_size=1000, val_size=300, seed=SEED)
    tokenizer = AutoTokenizer.from_pretrained(LIGHT_MODEL_NAME)

    exp_name = "exp5_head_tail"
    exp_dir = os.path.join(EXPERIMENTS_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 2. Inisialisasi Model & Loss
    model = create_model(model_name=LIGHT_MODEL_NAME)
    lr = 3e-5
    epochs = 2
    batch_size = 16
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    pos_counts = df_train_sub[LABEL_COLS].sum().values
    neg_counts = len(df_train_sub) - pos_counts
    weights = np.sqrt(neg_counts / np.maximum(pos_counts, 1))
    pos_weight_tensor = torch.tensor(weights, dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    train_ds = HeadTailDataset(df_train_sub["comment_text"].tolist(), df_train_sub[LABEL_COLS].values, tokenizer, m_len=128)
    val_ds = HeadTailDataset(df_val_sub["comment_text"].tolist(), df_val_sub[LABEL_COLS].values, tokenizer, m_len=128)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_head_tail)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_head_tail)

    print(f"\n[Training] Melatih {exp_name} (Head-Tail Slicing 64+64 token) selama {epochs} epoch...")
    history = []
    best_macro_f1 = -1.0
    best_model_state = None
    val_probs_best = None
    y_val_true = df_val_sub[LABEL_COLS].values

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

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_preds = []

        with torch.no_grad():
            for batch in val_loader:
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                v_loss = loss_fn(outputs.logits, batch["labels"])
                total_val_loss += v_loss.item()
                probs = torch.sigmoid(outputs.logits).cpu().numpy()
                val_preds.append(probs)

        avg_val_loss = total_val_loss / len(val_loader)
        y_val_probs = np.vstack(val_preds)
        eval_metrics = evaluate_multi_label(y_val_true, y_val_probs, threshold=0.5)
        macro_f1 = eval_metrics["Macro_F1"]
        epoch_time = time.time() - t0

        log_entry = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_macro_f1": round(macro_f1, 4),
            "val_micro_f1": round(eval_metrics["Micro_F1"], 4),
            "epoch_time_sec": round(epoch_time, 1)
        }
        history.append(log_entry)
        print(f"Epoch {epoch}/{epochs} ({epoch_time:.1f}s) | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Macro-F1: {macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_model_state = model.state_dict()
            val_probs_best = y_val_probs

    torch.save(best_model_state, os.path.join(exp_dir, "best_model.pt"))
    pd.DataFrame(history).to_csv(os.path.join(exp_dir, "history.csv"), index=False)

    # 3. Optimasi Threshold Adaptif Per-Label di Validation Set (Aksi 1.2)
    per_class_thresholds, val_eval_v2 = find_best_per_label_thresholds_on_validation(
        y_val_true,
        val_probs_best,
        candidate_thresholds=[0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    )

    # 4. Bekukan Konfigurasi v2 (Anti-Leakage)
    final_config_v2 = {
        "best_experiment": exp_name,
        "model_name": LIGHT_MODEL_NAME,
        "max_length": 128,
        "head_tail_truncation": True,
        "learning_rate": lr,
        "epochs": epochs,
        "use_pos_weight": True,
        "per_class_thresholds": per_class_thresholds,
        "val_macro_f1_adaptive_threshold": val_eval_v2["Macro_F1"],
        "val_micro_f1": val_eval_v2["Micro_F1"],
        "val_metrics_at_best_thresholds": val_eval_v2,
        "seed": SEED,
        "version": "v2_improved",
        "status": "FROZEN_FOR_TEST_EVALUATION"
    }

    final_cfg_v2_path = os.path.join(EXPERIMENTS_DIR, "final_config_v2.json")
    with open(final_cfg_v2_path, "w", encoding="utf-8") as f:
        json.dump(final_config_v2, f, indent=2)
    print(f"\nKonfigurasi v2 BERHASIL DIBEKUKAN ke: {final_cfg_v2_path}")

    # 5. Evaluasi Test Set SEKALI dengan Konfigurasi v2
    print("\n[Evaluasi Test Set v2] Mengevaluasi test set dengan model & threshold v2...")
    df_test_full = load_test_data(drop_unscored=True)
    df_test = df_test_full.sample(n=100, random_state=SEED).reset_index(drop=True)

    test_ds = HeadTailDataset(
        df_test["comment_text"].tolist(),
        df_test[LABEL_COLS].values,
        tokenizer,
        m_len=128,
        ids=df_test["id"].tolist()
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_head_tail)

    model.load_state_dict(best_model_state)
    model.eval()

    all_test_probs = []
    all_test_targets = []
    all_test_ids = []

    t_eval_start = time.time()
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            all_test_probs.append(probs)
            all_test_targets.append(batch["labels"].cpu().numpy())
            all_test_ids.extend(batch["id"])

    y_test_probs = np.vstack(all_test_probs)
    y_test_true = np.vstack(all_test_targets)

    # Hitung metrik final v2
    test_eval_v2 = evaluate_multi_label(y_test_true, y_test_probs, per_label_thresholds=per_class_thresholds)
    
    test_results_v2 = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": "v2_improved",
        "best_experiment": exp_name,
        "head_tail_truncation": True,
        "per_class_thresholds": per_class_thresholds,
        "num_test_samples": len(df_test),
        "total_inference_time_sec": round(time.time() - t_eval_start, 2),
        "metrics_at_adaptive_thresholds": test_eval_v2
    }

    test_res_v2_file = os.path.join(EXPERIMENTS_DIR, "test_evaluation_results_v2.json")
    with open(test_res_v2_file, "w", encoding="utf-8") as f:
        json.dump(test_results_v2, f, indent=2)

    # Simpan prediksi test v2
    pred_v2_df = pd.DataFrame({
        "id": all_test_ids,
        "comment_text": df_test["comment_text"].values
    })
    for i, col in enumerate(LABEL_COLS):
        t_col = per_class_thresholds[col]
        pred_v2_df[f"true_{col}"] = y_test_true[:, i].astype(int)
        pred_v2_df[f"prob_{col}"] = y_test_probs[:, i].astype(float)
        pred_v2_df[f"pred_{col}"] = (y_test_probs[:, i] >= t_col).astype(int)

    preds_v2_file = os.path.join(EXPERIMENTS_DIR, "test_predictions_v2.csv")
    pred_v2_df.to_csv(preds_v2_file, index=False)

    # 6. Analisis Error v2 & Bukti Penurunan Truncation FN
    print("\n[Analisis Kesalahan v2] Mengaudit dampak Head-Tail Slicing terhadap False Negative...")
    pred_v2_df["word_count"] = pred_v2_df["comment_text"].astype(str).str.split().apply(len)
    
    fn_toxic_v2 = pred_v2_df[(pred_v2_df["true_toxic"] == 1) & (pred_v2_df["pred_toxic"] == 0)]
    tp_toxic_v2 = pred_v2_df[(pred_v2_df["true_toxic"] == 1) & (pred_v2_df["pred_toxic"] == 1)]
    
    fn_count_v2 = len(fn_toxic_v2)
    tp_count_v2 = len(tp_toxic_v2)
    long_fn_ratio_v2 = float((fn_toxic_v2["word_count"] > 128).mean() * 100) if fn_count_v2 > 0 else 0.0

    error_summary_v2 = {
        "total_test_samples": len(pred_v2_df),
        "toxic_TP": tp_count_v2,
        "toxic_FN": fn_count_v2,
        "mean_words_fn_toxic": round(float(fn_toxic_v2["word_count"].mean()), 1) if fn_count_v2 > 0 else 0,
        "fn_long_comments_ratio_gt_128_words": long_fn_ratio_v2,
        "test_eval_metrics": test_eval_v2
    }

    with open(os.path.join(EXPERIMENTS_DIR, "error_analysis_summary_v2.json"), "w", encoding="utf-8") as f:
        json.dump(error_summary_v2, f, indent=2)

    # 7. Buat Visualisasi Komparasi Before (v1) vs After (v2)
    # Baca hasil v1 untuk perbandingan
    v1_res_path = os.path.join(EXPERIMENTS_DIR, "test_evaluation_results.json")
    with open(v1_res_path, "r", encoding="utf-8") as f:
        v1_data = json.load(f)["metrics_at_frozen_threshold"]

    labels_to_plot = ["Macro_F1", "Micro_F1", "Macro_Recall", "Micro_Recall"]
    v1_scores = [v1_data[m] for m in labels_to_plot]
    v2_scores = [test_eval_v2[m] for m in labels_to_plot]

    x = np.arange(len(labels_to_plot))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width/2, v1_scores, width, label="Baseline Tahap II (v1, Global tau=0.4)", color="#95a5a6")
    plt.bar(x + width/2, v2_scores, width, label="Hasil Perbaikan (v2, Head-Tail + Adaptive tau)", color="#2ecc71")
    plt.xticks(x, ["Macro F1", "Micro F1", "Macro Recall", "Micro Recall"])
    plt.ylabel("Skor Metrik (0 - 1.0)")
    plt.title("Perbandingan Metrik Utama: Baseline (v1) vs Perbaikan (v2)")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    comp_chart_path = os.path.join(figures_dir, "comparison_v1_vs_v2_f1.png")
    plt.savefig(comp_chart_path, dpi=200)
    plt.close()

    print(f"\n=======================================================")
    print("HASIL KOMPARASI FINAL: BASELINE v1 vs PERBAIKAN v2")
    print("=======================================================")
    print(f"Macro-F1 Test  : {v1_data['Macro_F1']:.4f} (v1) -> {test_eval_v2['Macro_F1']:.4f} (v2) [Delta: {test_eval_v2['Macro_F1'] - v1_data['Macro_F1']:+.4f}]")
    print(f"Micro-F1 Test  : {v1_data['Micro_F1']:.4f} (v1) -> {test_eval_v2['Micro_F1']:.4f} (v2) [Delta: {test_eval_v2['Micro_F1'] - v1_data['Micro_F1']:+.4f}]")
    print(f"Macro-Recall   : {v1_data['Macro_Recall']:.4f} (v1) -> {test_eval_v2['Macro_Recall']:.4f} (v2) [Delta: {test_eval_v2['Macro_Recall'] - v1_data['Macro_Recall']:+.4f}]")
    print(f"Hamming Loss   : {v1_data['Hamming_Loss']:.4f} (v1) -> {test_eval_v2['Hamming_Loss']:.4f} (v2)")
    print(f"Chart komparasi tersimpan di: {comp_chart_path}")
    print("Pipeline v2 Selesai dengan Sukses!")

if __name__ == "__main__":
    run_pipeline_v2()
