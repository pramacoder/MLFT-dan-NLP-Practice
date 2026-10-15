"""
Modul Evaluasi dan Pemilihan Threshold untuk Jigsaw Toxicity (ML-T5-05, ML-T9, ML-T10).

Tanggung Jawab:
- Menghitung metrik Multi-label: Macro-F1 (metrik utama), Micro-F1, per-label P/R/F1, dan Hamming Loss.
- Melakukan threshold sweep pada Validation Set {0.3, 0.4, 0.5, 0.6, 0.7} untuk memilih threshold optimal
  sebelum melihat data test.
- Menghasilkan laporan evaluasi test (hanya dijalankan SATU KALI pada evaluasi akhir).
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
from .config import LABEL_COLS

def evaluate_multi_label(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5,
    per_label_thresholds: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Evaluasi multi-label dengan threshold tertentu.
    y_true: matriks biner (N, 6)
    y_probs: matriks probabilitas sigmoid (N, 6)
    """
    if per_label_thresholds is not None:
        y_pred = np.zeros_like(y_probs, dtype=int)
        for i, col in enumerate(LABEL_COLS):
            t = per_label_thresholds.get(col, threshold)
            y_pred[:, i] = (y_probs[:, i] >= t).astype(int)
    else:
        y_pred = (y_probs >= threshold).astype(int)

    # Global Metrics
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    micro_p = precision_score(y_true, y_pred, average="micro", zero_division=0)
    micro_r = recall_score(y_true, y_pred, average="micro", zero_division=0)
    h_loss = hamming_loss(y_true, y_pred)

    # Per-label Metrics
    per_label = {}
    for i, col in enumerate(LABEL_COLS):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]
        per_label[col] = {
            "Precision": float(precision_score(col_true, col_pred, zero_division=0)),
            "Recall": float(recall_score(col_true, col_pred, zero_division=0)),
            "F1": float(f1_score(col_true, col_pred, zero_division=0)),
            "Support": int(col_true.sum())
        }

    return {
        "Macro_F1": float(macro_f1),
        "Micro_F1": float(micro_f1),
        "Macro_Precision": float(macro_p),
        "Macro_Recall": float(macro_r),
        "Micro_Precision": float(micro_p),
        "Micro_Recall": float(micro_r),
        "Hamming_Loss": float(h_loss),
        "Per_Label": per_label,
        "Threshold": threshold if per_label_thresholds is None else per_label_thresholds
    }

def find_best_threshold_on_validation(
    y_true_val: np.ndarray,
    y_probs_val: np.ndarray,
    candidate_thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7]
) -> Tuple[float, Dict[str, Any]]:
    """
    Mencari threshold global terbaik berdasarkan Macro-F1 tertinggi pada validation split (ML-T9-02).
    """
    best_t = 0.5
    best_res = None
    best_macro = -1.0

    print("--- Validation Threshold Sweep ---")
    for t in candidate_thresholds:
        res = evaluate_multi_label(y_true_val, y_probs_val, threshold=t)
        macro = res["Macro_F1"]
        print(f"Threshold {t:.2f} -> Macro-F1: {macro:.4f} | Micro-F1: {res['Micro_F1']:.4f} | Hamming Loss: {res['Hamming_Loss']:.4f}")
        if macro > best_macro:
            best_macro = macro
            best_t = t
            best_res = res

    print(f"\nOptimal Global Threshold terpilih: {best_t:.2f} (Macro-F1: {best_macro:.4f})")
    return best_t, best_res

def find_best_per_label_thresholds_on_validation(
    y_true_val: np.ndarray,
    y_probs_val: np.ndarray,
    candidate_thresholds: List[float] = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Mencari threshold adaptif optimal secara independen per label (Aksi 1.2).
    Mengoptimalkan F1 score per label pada validation split untuk memaksimalkan Macro-F1.
    """
    best_thresholds = {}
    print("\n--- Per-Class Validation Threshold Sweep ---")
    for i, col in enumerate(LABEL_COLS):
        col_true = y_true_val[:, i]
        col_probs = y_probs_val[:, i]

        best_t = 0.40
        best_f1 = -1.0

        if col_true.sum() == 0:
            best_thresholds[col] = 0.40
            print(f"Label '{col:<13}': Support=0 -> Default Threshold: 0.40")
            continue

        for t in candidate_thresholds:
            col_pred = (col_probs >= t).astype(int)
            f1 = f1_score(col_true, col_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        best_thresholds[col] = round(float(best_t), 2)
        print(f"Label '{col:<13}': Best Threshold = {best_t:.2f} (Val F1: {best_f1:.4f}, Positif: {col_true.sum()})")

    eval_res = evaluate_multi_label(y_true_val, y_probs_val, per_label_thresholds=best_thresholds)
    print(f"Macro-F1 Validation (Adaptive Per-Class): {eval_res['Macro_F1']:.4f} (Micro-F1: {eval_res['Micro_F1']:.4f})")
    return best_thresholds, eval_res

