"""
Modul Analisis Kesalahan (Error & Bias Analysis) untuk Jigsaw Toxicity (ML-T11, Tahap II).

Tanggung Jawab:
- Menganalisis hasil prediksi test set (test_predictions.csv).
- Mengidentifikasi 4 kategori kesalahan utama:
  1. False Positives pada teks netral (over-flagging karena kata sensitif/slang).
  2. False Negatives pada toksisitas halus / sarkasme / implicit toxicity.
  3. Konflasi antar-label (misal: insult vs toxic, atau severe_toxic vs toxic).
  4. Analisis bias terhadap istilah identitas (gay, lesbian, muslim, jew, black, dsb.).
  5. Pengaruh panjang token terhadap False Negatives (efek truncation).
- Menyimpan ringkasan statistik dan sampel kualitatif untuk laporan.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from .config import EXPERIMENTS_DIR, REPORTS_DIR, LABEL_COLS

def run_error_analysis() -> Dict[str, Any]:
    preds_file = os.path.join(EXPERIMENTS_DIR, "test_predictions.csv")
    if not os.path.exists(preds_file):
        raise FileNotFoundError(f"test_predictions.csv tidak ditemukan di {preds_file}. Jalankan evaluate_test.py terlebih dahulu.")

    print(f"Membaca file prediksi: {preds_file}...")
    df = pd.read_csv(preds_file)
    n_total = len(df)
    print(f"Total sampel dianalisis: {n_total:,}")

    figures_dir = os.path.join(REPORTS_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Matriks Kesalahan per Label (TP, FP, FN, TN)
    label_error_stats = {}
    for col in LABEL_COLS:
        y_true = df[f"true_{col}"].values
        y_pred = df[f"pred_{col}"].values
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        label_error_stats[col] = {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "False_Positive_Rate": round(float(fp / (fp + tn + 1e-9) * 100), 3),
            "False_Negative_Rate": round(float(fn / (fn + tp + 1e-9) * 100), 3)
        }

    # 2. Analisis Bias Istilah Identitas (Identity Bias Analysis)
    identity_keywords = ["gay", "lesbian", "homosexual", "muslim", "jewish", "islam", "black", "woman", "transgender"]
    bias_results = {}
    
    # Hitung base rate FP untuk teks tanpa identity terms
    df["has_identity"] = df["comment_text"].astype(str).str.lower().apply(
        lambda t: any(kw in t.split() or f" {kw} " in f" {t} " for kw in identity_keywords)
    )

    non_id_clean = df[(~df["has_identity"]) & (df["true_toxic"] == 0)]
    base_fp_rate = float(non_id_clean["pred_toxic"].mean() * 100) if len(non_id_clean) > 0 else 0.0

    print(f"\n--- Analisis Bias Istilah Identitas ---")
    print(f"Base FP Rate pada komentar non-identitas yang bersih: {base_fp_rate:.2f}%")

    for kw in identity_keywords:
        mask_kw = df["comment_text"].astype(str).str.lower().apply(lambda t: kw in t.lower().split() or f" {kw} " in f" {t.lower()} ")
        sub_df = df[mask_kw]
        total_kw = len(sub_df)
        if total_kw > 0:
            clean_kw = sub_df[sub_df["true_toxic"] == 0]
            fp_kw = clean_kw[clean_kw["pred_toxic"] == 1]
            fp_rate_kw = float(len(fp_kw) / len(clean_kw) * 100) if len(clean_kw) > 0 else 0.0
            
            bias_results[kw] = {
                "total_occurrences": total_kw,
                "clean_count": len(clean_kw),
                "fp_count": len(fp_kw),
                "fp_rate_percent": round(fp_rate_kw, 2),
                "disproportionate_bias_ratio": round(fp_rate_kw / (base_fp_rate + 1e-5), 2)
            }
            print(f"Keyword '{kw:<12}': total={total_kw:<4} | clean={len(clean_kw):<4} | FP={len(fp_kw):<3} ({fp_rate_kw:.2f}%)")

    # 3. Analisis Pengaruh Panjang Teks terhadap FN (Truncation Effect)
    df["word_count"] = df["comment_text"].astype(str).str.split().apply(len)
    fn_toxic = df[(df["true_toxic"] == 1) & (df["pred_toxic"] == 0)]
    tp_toxic = df[(df["true_toxic"] == 1) & (df["pred_toxic"] == 1)]
    
    length_analysis = {
        "mean_words_all": round(float(df["word_count"].mean()), 1),
        "mean_words_tp_toxic": round(float(tp_toxic["word_count"].mean()), 1) if len(tp_toxic) > 0 else 0,
        "mean_words_fn_toxic": round(float(fn_toxic["word_count"].mean()), 1) if len(fn_toxic) > 0 else 0,
        "fn_long_comments_ratio_gt_128_words": round(float((fn_toxic["word_count"] > 128).mean() * 100), 2) if len(fn_toxic) > 0 else 0
    }

    # 4. Pengambilan Sampel Kualitatif Representatif
    def get_sample_records(subset, n=3):
        samples = []
        for _, row in subset.head(n).iterrows():
            samples.append({
                "id": str(row["id"]),
                "text": str(row["comment_text"])[:250] + ("..." if len(str(row["comment_text"])) > 250 else ""),
                "true_labels": [c for c in LABEL_COLS if row[f"true_{c}"] == 1],
                "pred_labels": [c for c in LABEL_COLS if row[f"pred_{c}"] == 1],
                "probs": {c: round(float(row[f"prob_{c}"]), 3) for c in LABEL_COLS}
            })
        return samples

    # FP: True clean (all 0), Pred has toxic
    clean_all = (df[[f"true_{c}" for c in LABEL_COLS]] == 0).all(axis=1)
    fp_toxic_all = df[clean_all & (df["pred_toxic"] == 1)]
    fp_samples = get_sample_records(fp_toxic_all, n=5)

    # FN: True toxic == 1, Pred toxic == 0
    fn_samples = get_sample_records(fn_toxic, n=5)

    # Label conflation: True insult == 1, Pred insult == 0 but Pred toxic == 1
    conflation_insult = df[(df["true_insult"] == 1) & (df["pred_insult"] == 0) & (df["pred_toxic"] == 1)]
    conflation_samples = get_sample_records(conflation_insult, n=3)

    summary_error = {
        "total_test_samples": n_total,
        "label_error_stats": label_error_stats,
        "base_fp_rate_clean_non_identity": round(base_fp_rate, 2),
        "identity_bias_analysis": bias_results,
        "text_length_and_truncation_analysis": length_analysis,
        "qualitative_cases": {
            "false_positives_benign_overflagged": fp_samples,
            "false_negatives_subtle_missed": fn_samples,
            "label_conflation_insult_as_general_toxic": conflation_samples
        }
    }

    out_file = os.path.join(EXPERIMENTS_DIR, "error_analysis_summary.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_error, f, indent=2)
    print(f"\nAnalisis kesalahan disimpan ke: {out_file}")

    # Plot Visualisasi False Positive Rate vs False Negative Rate
    labels = LABEL_COLS
    fprs = [label_error_stats[c]["False_Positive_Rate"] for c in labels]
    fnrs = [label_error_stats[c]["False_Negative_Rate"] for c in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - width/2, fprs, width, label="False Positive Rate (%)", color="#e74c3c")
    rects2 = ax.bar(x + width/2, fnrs, width, label="False Negative Rate (%)", color="#3498db")

    ax.set_ylabel("Persentase (%)")
    ax.set_title("Perbandingan Tingkat Kesalahan (FPR vs FNR) per Label pada Test Set")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend()
    plt.tight_layout()
    chart_path = os.path.join(figures_dir, "error_rates_per_label.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"Chart FPR/FNR disimpan ke: {chart_path}")

    return summary_error

if __name__ == "__main__":
    run_error_analysis()
