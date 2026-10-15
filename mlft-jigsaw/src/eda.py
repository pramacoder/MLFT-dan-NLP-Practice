import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from tokenizers import Tokenizer

from .config import DATA_DIR, LABEL_COLS, SEED, MODEL_NAME, LIGHT_MODEL_NAME
from .data_loader import load_train_data, load_test_data, get_train_val_split

def run_jigsaw_eda(save_dir: str = None) -> Dict:
    """
    Performs comprehensive Exploratory Data Analysis for Jigsaw Toxicity (ML-T2 & ML-T3):
    - ML-T2: Data validation (missing values, duplicates, out-of-range labels, test -1 filter)
    - ML-T3-01: Label frequencies, clean vs toxic proportions
    - ML-T3-02: Label cardinality & density
    - ML-T3-03: Co-occurrence matrix & correlation
    - ML-T3-04: Rare labels identification & imbalance analysis
    - ML-T3-05: Subword length & truncation rates at 128/256
    - ML-T3-06: Potential contextual bias & non-offensive identity examples
    - ML-T4: Stratified split verification
    """
    if save_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(base_dir, "reports", "figures")
    os.makedirs(save_dir, exist_ok=True)

    print("=== [1/5] Loading Data & Running ML-T2 Validations ===")
    df_train_full = load_train_data()
    df_test_valid = load_test_data(drop_unscored=True)
    df_test_all = load_test_data(drop_unscored=False)
    df_train, df_val = get_train_val_split()

    # ML-T2-02: Validation checks
    missing_texts = int(df_train_full["comment_text"].isna().sum())
    empty_texts = int((df_train_full["comment_text"].str.strip() == "").sum())
    duplicate_texts = int(df_train_full.duplicated(subset=["comment_text"]).sum())
    
    # Check label values {0, 1}
    out_of_range_labels = {}
    for col in LABEL_COLS:
        invalid_vals = df_train_full[~df_train_full[col].isin([0, 1])]
        out_of_range_labels[col] = len(invalid_vals)

    # ML-T2-03: Test -1 checks
    unscored_test_count = len(df_test_all) - len(df_test_valid)
    unscored_test_pct = (unscored_test_count / len(df_test_all)) * 100

    validation_summary = {
        "missing_texts": missing_texts,
        "empty_texts": empty_texts,
        "duplicate_texts": duplicate_texts,
        "out_of_range_labels": out_of_range_labels,
        "test_total_rows": len(df_test_all),
        "test_unscored_minus_one_rows": unscored_test_count,
        "test_unscored_minus_one_pct": round(unscored_test_pct, 2),
        "test_valid_scored_rows": len(df_test_valid)
    }

    # ==========================================
    # ML-T3-01 & ML-T3-02: Frequencies & Cardinality
    # ==========================================
    print("=== [2/5] Analyzing Label Distributions & Cardinality ===")
    n_train = len(df_train_full)
    label_counts = df_train_full[LABEL_COLS].sum().to_dict()
    label_props = {col: round((cnt / n_train) * 100, 3) for col, cnt in label_counts.items()}

    active_labels_per_row = df_train_full[LABEL_COLS].sum(axis=1)
    clean_count = int((active_labels_per_row == 0).sum())
    toxic_count = int((active_labels_per_row > 0).sum())
    
    label_cardinality_all = float(active_labels_per_row.mean())
    label_cardinality_toxic_only = float(active_labels_per_row[active_labels_per_row > 0].mean())

    # Plot 1: 6 Labels Distribution
    plt.figure(figsize=(9, 4.5))
    x_positions = np.arange(len(LABEL_COLS))
    counts = [label_counts[col] for col in LABEL_COLS]
    bars = plt.bar(x_positions, counts, color="#2980b9", edgecolor="black", alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        pct = (h / n_train) * 100
        plt.text(bar.get_x() + bar.get_width()/2.0, h + 300, f"{h:,}\n({pct:.2f}%)", ha="center", va="bottom", fontsize=8.5)
    plt.xticks(x_positions, LABEL_COLS, rotation=15)
    plt.title("Frekuensi & Persentase 6 Label Toksisitas (Train Set: 159.571 Ulasan)")
    plt.ylabel("Jumlah Ulasan Positif")
    plt.ylim(0, max(counts) * 1.18)
    plt.tight_layout()
    fig1_path = os.path.join(save_dir, "distribusi_6_label.png")
    plt.savefig(fig1_path, dpi=200)
    plt.close()

    # ==========================================
    # ML-T3-03: Co-occurrence & Correlation Heatmap
    # ==========================================
    print("=== [3/5] Computing Co-occurrence Matrix ===")
    cooc_matrix = df_train_full[LABEL_COLS].T.dot(df_train_full[LABEL_COLS])
    
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cooc_matrix, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.title("Matriks Co-occurrence 6 Label Toksisitas (Train)")
    plt.tight_layout()
    fig2_path = os.path.join(save_dir, "heatmap_cooccurrence.png")
    plt.savefig(fig2_path, dpi=200)
    plt.close()

    # ==========================================
    # ML-T3-05: Subword Length & Truncation Rates
    # ==========================================
    print("=== [4/5] Computing Subword Lengths (DistilRoBERTa) ===")
    try:
        tok = Tokenizer.from_pretrained("distilroberta-base")
    except Exception:
        tok = Tokenizer.from_pretrained("roberta-base")

    # Sample 5,000 comments for fast and statistically representative token length estimation
    sample_texts = df_train_full["comment_text"].sample(5000, random_state=SEED).fillna("").tolist()
    subword_lengths = pd.Series([len(tok.encode(t).ids) for t in sample_texts])

    trunc_128 = float((subword_lengths > 128).mean() * 100)
    trunc_256 = float((subword_lengths > 256).mean() * 100)
    trunc_512 = float((subword_lengths > 512).mean() * 100)

    plt.figure(figsize=(9, 4.5))
    plt.hist(subword_lengths, bins=50, range=(0, 600), color="#16a085", edgecolor="white", alpha=0.85)
    plt.axvline(128, color="red", linestyle="--", linewidth=2, label=f"Max Len 128 (Trunc: {trunc_128:.1f}%)")
    plt.axvline(256, color="orange", linestyle="--", linewidth=2, label=f"Max Len 256 (Trunc: {trunc_256:.1f}%)")
    plt.title("Distribusi Panjang Subword Token (RoBERTa Tokenizer)")
    plt.xlabel("Jumlah Subword Token")
    plt.ylabel("Frekuensi")
    plt.legend()
    plt.tight_layout()
    fig3_path = os.path.join(save_dir, "distribusi_panjang_subword.png")
    plt.savefig(fig3_path, dpi=200)
    plt.close()

    # ==========================================
    # ML-T4: Split Stratification Check
    # ==========================================
    print("=== [5/5] Verifying Multi-label Stratification ===")
    train_dist = (df_train[LABEL_COLS].sum() / len(df_train) * 100).to_dict()
    val_dist = (df_val[LABEL_COLS].sum() / len(df_val) * 100).to_dict()

    strat_comparison = {}
    for col in LABEL_COLS:
        strat_comparison[col] = {
            "train_count": int(df_train[col].sum()),
            "train_pct": round(train_dist[col], 3),
            "val_count": int(df_val[col].sum()),
            "val_pct": round(val_dist[col], 3),
            "abs_diff_pct": round(abs(train_dist[col] - val_dist[col]), 4)
        }

    # ==========================================
    # ML-T3-06: Potential Contextual Bias / False Positive Examples (Non-Offensive)
    # ==========================================
    bias_discussion_cases = [
        {
            "category": "Identitas Agama / Etnis",
            "example": "I am a proud Muslim and I love reading Wikipedia articles about world history.",
            "risk": "Model yang overfitting pada kata 'Muslim' (yang sering muncul bersamaan dengan ujaran kebencian di data latih) dapat keliru memprediksi label identity_hate meskipun kalimat bernada positif/netral."
        },
        {
            "category": "Orientasi Seksual",
            "example": "Many gay and lesbian youth struggle with mental health and deserve our support.",
            "risk": "Kata 'gay' dan 'lesbian' rentan memicu false positive pada label insult atau identity_hate jika model hanya mendeteksi kata kunci terisolasi tanpa memahami konteks afirmatif."
        },
        {
            "category": "Slang / Bahasa Informal Positif",
            "example": "That guitar solo was totally wicked and sick! Absolute killer performance.",
            "risk": "Kata 'sick', 'wicked', dan 'killer' sering bermakna negatif dalam kamus harfiah, namun bernada pujian tinggi dalam bahasa gaul, berisiko memicu false positive toxic."
        }
    ]

    eda_summary = {
        "validations": validation_summary,
        "imbalance_and_cardinality": {
            "total_reviews": n_train,
            "clean_reviews_count": clean_count,
            "clean_reviews_pct": round((clean_count / n_train) * 100, 2),
            "toxic_reviews_count": toxic_count,
            "toxic_reviews_pct": round((toxic_count / n_train) * 100, 2),
            "label_cardinality_all": round(label_cardinality_all, 4),
            "label_cardinality_toxic_only": round(label_cardinality_toxic_only, 2),
            "label_frequencies": label_counts,
            "label_percentages": label_props,
            "rarest_label": min(label_props, key=label_props.get),
            "rarest_label_pct": min(label_props.values()),
            "most_common_label": max(label_props, key=label_props.get),
            "most_common_label_pct": max(label_props.values())
        },
        "length_and_truncation": {
            "median_tokens": float(subword_lengths.median()),
            "mean_tokens": round(float(subword_lengths.mean()), 2),
            "truncation_pct_128": round(trunc_128, 2),
            "truncation_pct_256": round(trunc_256, 2),
            "truncation_pct_512": round(trunc_512, 2)
        },
        "stratification": strat_comparison,
        "bias_cases": bias_discussion_cases,
        "ml_decisions": {
            "loss_function": "BCEWithLogitsLoss karena multi-label bukan multi-class saling lepas.",
            "evaluation_metric": "Macro-F1 ditetapkan sebelum melihat test, untuk memastikan label langka (threat, severe_toxic) tidak tertutup oleh akurasi tinggi komentar bersih (89,8%).",
            "max_length": "128 (atau 256 jika memori cukup) dipilih karena 86,4% ulasan berada di bawah 128 token, dan menghemat memori GPU secara drastis.",
            "test_filtering": "89.186 baris berlabel -1 pada test_labels.csv wajib difilter keluar sebelum evaluasi akhir.",
            "thresholding": "Threshold global dan per-label akan di-tune hanya pada validation split dengan kriteria Macro-F1 tertinggi."
        }
    }

    summary_file = os.path.join(save_dir, "eda_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2, ensure_ascii=False)

    print(f"Jigsaw EDA successfully completed! Figures & summary saved to {save_dir}")
    return eda_summary

if __name__ == "__main__":
    run_jigsaw_eda()
