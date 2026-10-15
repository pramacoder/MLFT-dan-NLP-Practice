import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import Dict, List, Tuple

from .config import DATA_DIR, REPORTS_DIR, LABEL_LIST, MAX_LEN
from .data_loader import load_split
from .annotation import extract_spans_from_bio

def run_eda(save_dir: str = None) -> Dict:
    """
    Performs comprehensive Exploratory Data Analysis (EDA) on IndoNLU TermA (NLP-T3):
    - NLP-T3-01: Length distributions & truncation rate
    - NLP-T3-02: Label frequencies & O vs non-O ratio
    - NLP-T3-03: Span statistics & single vs multi-token analysis
    - NLP-T3-04: Negation and intensifiers analysis from Train set
    - NLP-T3-05 & T3-06: Hard cases and modeling implications
    """
    if save_dir is None:
        save_dir = os.path.join(REPORTS_DIR, "figures")
    os.makedirs(save_dir, exist_ok=True)

    df_train = load_split("train")
    df_valid = load_split("valid")
    df_test = load_split("test")

    eda_results = {}

    # ==========================================
    # NLP-T3-01: Review Lengths & Truncation Rate
    # ==========================================
    train_lengths = df_train["tokens"].apply(len)
    val_lengths = df_valid["tokens"].apply(len)
    test_lengths = df_test["tokens"].apply(len)

    trunc_train_128 = (train_lengths > MAX_LEN).mean() * 100
    trunc_val_128 = (val_lengths > MAX_LEN).mean() * 100
    trunc_test_128 = (test_lengths > MAX_LEN).mean() * 100

    eda_results["length_stats"] = {
        "train": {
            "mean": float(train_lengths.mean()),
            "median": float(train_lengths.median()),
            "max": int(train_lengths.max()),
            "min": int(train_lengths.min()),
            "truncation_pct_128": float(trunc_train_128)
        },
        "valid": {
            "mean": float(val_lengths.mean()),
            "median": float(val_lengths.median()),
            "max": int(val_lengths.max()),
            "truncation_pct_128": float(trunc_val_128)
        },
        "test": {
            "mean": float(test_lengths.mean()),
            "median": float(test_lengths.median()),
            "max": int(test_lengths.max()),
            "truncation_pct_128": float(trunc_test_128)
        }
    }

    # Plot length distribution
    plt.figure(figsize=(9, 4))
    plt.hist(train_lengths, bins=40, color="#2b5c8f", edgecolor="white", alpha=0.85)
    plt.axvline(MAX_LEN, color="red", linestyle="--", linewidth=2, label=f"Max Len ({MAX_LEN})")
    plt.title("Distribusi Panjang Ulasan (Token) - Training Set")
    plt.xlabel("Jumlah Token per Ulasan")
    plt.ylabel("Frekuensi")
    plt.legend()
    plt.tight_layout()
    fig1_path = os.path.join(save_dir, "distribusi_panjang_ulasan.png")
    plt.savefig(fig1_path, dpi=200)
    plt.close()

    # ==========================================
    # NLP-T3-02: BIO Label Frequency & O-ratio
    # ==========================================
    train_all_labels = [lbl for labels in df_train["seq_label"] for lbl in labels]
    label_counts = Counter(train_all_labels)
    total_tokens = len(train_all_labels)
    o_count = label_counts.get("O", 0)
    non_o_count = total_tokens - o_count
    ratio_o = (o_count / total_tokens) * 100
    ratio_non_o = (non_o_count / total_tokens) * 100

    eda_results["label_frequencies"] = dict(label_counts)
    eda_results["imbalance"] = {
        "total_tokens": total_tokens,
        "O_tokens": o_count,
        "non_O_tokens": non_o_count,
        "O_percentage": round(ratio_o, 2),
        "non_O_percentage": round(ratio_non_o, 2),
        "O_to_non_O_ratio": round(o_count / non_o_count, 2)
    }

    # Plot label distribution
    plt.figure(figsize=(8, 4))
    ordered_labels = ["O", "B-ASPECT", "I-ASPECT", "B-SENTIMENT", "I-SENTIMENT"]
    counts = [label_counts.get(l, 0) for l in ordered_labels]
    colors = ["#7f8c8d", "#2980b9", "#5dade2", "#e67e22", "#f39c12"]
    bars = plt.bar(ordered_labels, counts, color=colors)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 300, f"{yval:,}\n({yval/total_tokens*100:.1f}%)", ha="center", va="bottom", fontsize=8)
    plt.title("Frekuensi Label BIO pada Training Set")
    plt.ylabel("Jumlah Token")
    plt.ylim(0, max(counts) * 1.18)
    plt.tight_layout()
    fig2_path = os.path.join(save_dir, "frekuensi_label_bio.png")
    plt.savefig(fig2_path, dpi=200)
    plt.close()

    # ==========================================
    # NLP-T3-03: Span Counts & Length Analysis
    # ==========================================
    aspect_lengths = []
    sentiment_lengths = []
    spans_per_sentence = []

    for idx, row in df_train.iterrows():
        spans = extract_spans_from_bio(row["tokens"], row["seq_label"])
        n_asp = len(spans["ASPECT"])
        n_snt = len(spans["SENTIMENT"])
        spans_per_sentence.append(n_asp + n_snt)
        
        for s in spans["ASPECT"]:
            aspect_lengths.append(len(s.split()))
        for s in spans["SENTIMENT"]:
            sentiment_lengths.append(len(s.split()))

    aspect_len_cnt = Counter(aspect_lengths)
    sentiment_len_cnt = Counter(sentiment_lengths)

    eda_results["span_statistics"] = {
        "total_aspect_spans": len(aspect_lengths),
        "total_sentiment_spans": len(sentiment_lengths),
        "avg_spans_per_review": round(np.mean(spans_per_sentence), 2),
        "aspect_single_token_pct": round((aspect_len_cnt[1] / len(aspect_lengths)) * 100, 2),
        "aspect_multi_token_pct": round((sum(v for k, v in aspect_len_cnt.items() if k > 1) / len(aspect_lengths)) * 100, 2),
        "sentiment_single_token_pct": round((sentiment_len_cnt[1] / len(sentiment_lengths)) * 100, 2),
        "sentiment_multi_token_pct": round((sum(v for k, v in sentiment_len_cnt.items() if k > 1) / len(sentiment_lengths)) * 100, 2),
    }

    # Plot Single vs Multi-token Spans
    plt.figure(figsize=(7, 4))
    categories = ["ASPECT", "SENTIMENT"]
    single_pcts = [eda_results["span_statistics"]["aspect_single_token_pct"], eda_results["span_statistics"]["sentiment_single_token_pct"]]
    multi_pcts = [eda_results["span_statistics"]["aspect_multi_token_pct"], eda_results["span_statistics"]["sentiment_multi_token_pct"]]
    
    x = np.arange(len(categories))
    width = 0.35
    plt.bar(x - width/2, single_pcts, width, label="Single Token", color="#3498db")
    plt.bar(x + width/2, multi_pcts, width, label="Multi Token", color="#e74c3c")
    plt.xticks(x, categories)
    plt.ylabel("Persentase (%)")
    plt.title("Perbandingan Panjang Span (Single vs Multi-Token)")
    plt.legend()
    plt.ylim(0, 100)
    plt.tight_layout()
    fig3_path = os.path.join(save_dir, "panjang_span_tipe.png")
    plt.savefig(fig3_path, dpi=200)
    plt.close()

    # ==========================================
    # NLP-T3-04: Negations & Intensifiers (Train Set Only)
    # ==========================================
    negations = ["tidak", "bukan", "kurang", "belum", "tanpa", "tak"]
    intensifiers = ["sangat", "banget", "sekali", "terlalu", "cukup", "lumayan", "agak", "bener"]

    word_counter = Counter([t.lower() for tokens in df_train["tokens"] for t in tokens])
    neg_counts = {w: word_counter.get(w, 0) for w in negations}
    int_counts = {w: word_counter.get(w, 0) for w in intensifiers}

    eda_results["lexical_markers"] = {
        "negations": neg_counts,
        "total_negations": sum(neg_counts.values()),
        "intensifiers": int_counts,
        "total_intensifiers": sum(int_counts.values())
    }

    # ==========================================
    # NLP-T3-06: Modeling Implications Summary
    # ==========================================
    implications = {
        "imbalance_implication": (
            f"Label 'O' mendominasi dengan {ratio_o:.1f}% dari total token (rasio O:non-O = {o_count/non_o_count:.2f}:1). "
            "Akurasi token-level akan menipu; metrik wajib dievaluasi dengan exact-match span Precision, Recall, dan F1 "
            "dengan mengabaikan label 'O'."
        ),
        "truncation_implication": (
            f"Dengan MAX_LEN={MAX_LEN}, ulasan yang terpotong adalah {trunc_train_128:.2f}% (train) dan {trunc_val_128:.2f}% (val). "
            f"Panjang median hanya {train_lengths.median():.0f} token. "
            f"Batas {MAX_LEN} aman dan optimal untuk efisiensi komputasi Transformer dan CRF."
        ),
        "span_length_implication": (
            f"Sebanyak {eda_results['span_statistics']['aspect_multi_token_pct']}% aspek dan "
            f"{eda_results['span_statistics']['sentiment_multi_token_pct']}% sentimen terdiri dari multi-token. "
            "Ini mengindikasikan risiko tinggi 'boundary error' (kesalahan menentukan batas awal/akhir span), "
            "sehingga transisi BIO (seperti B-X diikuti I-X) dan fitur konteks lokal n-gram sangat krusial."
        ),
        "lexical_markers_implication": (
            f"Ditemukan {sum(neg_counts.values())} kemunculan kata negasi (terbanyak: '{max(neg_counts, key=neg_counts.get)}') "
            f"dan {sum(int_counts.values())} kata intensifier (terbanyak: '{max(int_counts, key=int_counts.get)}'). "
            "Pada CRF, penambahan fitur leksikal biner (misal `prev_is_negation` atau `curr_is_intensifier`) "
            "sangat berpotensi meningkatkan F1 sentimen."
        )
    }
    eda_results["implications"] = implications

    # Save summary json
    summary_path = os.path.join(save_dir, "eda_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(eda_results, f, indent=2, ensure_ascii=False)

    print(f"EDA successfully completed! Figures and summary saved to {save_dir}")
    return eda_results

if __name__ == "__main__":
    results = run_eda()
