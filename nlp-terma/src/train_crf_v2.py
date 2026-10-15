"""
Pipeline Eksekusi Perbaikan v2 untuk NLP TermA (NLP-Aksi 2.1: Character n-grams & Contextual Bigrams).

Tanggung Jawab:
1. Mengekstrak fitur leksikal yang diperkaya dengan Character 3-grams dan Contextual Bigrams
   untuk menangani typo (misal: 'westafle') dan boundary kata majemuk ('staf recepsionis').
2. Melakukan tuning hyperparameter c1/c2 pada Validation Set.
3. Menyimpan model v2 ke nlp-terma/experiments/crf_v2_char_ngrams.joblib (TIDAK menimpa v1).
4. Mengevaluasi Test set independen dan menyimpan hasil ke
   nlp-terma/experiments/crf_v2_final_summary.json dan nlp-terma/experiments/crf_v2_error_analysis.json.
5. Menghasilkan visualisasi Before vs After di nlp-terma/reports/figures/.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import sklearn_crfsuite

from .config import (
    DATA_DIR, EXPERIMENTS_DIR, REPORTS_DIR, SEED,
    CRF_C1_VALUES, CRF_C2_VALUES, CRF_ALGORITHM, CRF_MAX_ITERATIONS
)
from .data_loader import load_split
from .evaluate import compute_span_metrics
from .crf_features import NEGATIONS, INTENSIFIERS, word_shape

def token2features_v2(sent_tokens: List[str], i: int) -> Dict[str, Any]:
    """
    Ekstraksi fitur v2 yang diperkaya:
    - Character 3-grams internal untuk ketahanan typo / OOV leksikal.
    - Contextual Bigrams (-1:bigram, +1:bigram) untuk mendeteksi batas entitas majemuk.
    - Subword prefix & suffix hingga panjang 4.
    """
    word = sent_tokens[i]
    word_lower = word.lower()

    features = {
        "bias": 1.0,
        "word.lower()": word_lower,
        "word[-4:]": word_lower[-4:] if len(word_lower) >= 4 else word_lower,
        "word[-3:]": word_lower[-3:] if len(word_lower) >= 3 else word_lower,
        "word[-2:]": word_lower[-2:] if len(word_lower) >= 2 else word_lower,
        "word[:4]": word_lower[:4] if len(word_lower) >= 4 else word_lower,
        "word[:3]": word_lower[:3] if len(word_lower) >= 3 else word_lower,
        "word[:2]": word_lower[:2] if len(word_lower) >= 2 else word_lower,
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "word.shape": word_shape(word),
        "is_negation": word_lower in NEGATIONS,
        "is_intensifier": word_lower in INTENSIFIERS,
    }

    # Tambahkan Character 3-grams internal (menangkap kemiripan substring typo seperti westafle -> wes, taf, fle)
    if len(word_lower) >= 3:
        for idx in range(len(word_lower) - 2):
            c_trigram = word_lower[idx:idx + 3]
            features[f"char3_{c_trigram}"] = 1.0

    # Konteks Sebelumnya (-1) & Bigram
    if i > 0:
        prev_word = sent_tokens[i - 1]
        prev_lower = prev_word.lower()
        features.update({
            "-1:word.lower()": prev_lower,
            "-1:word.istitle()": prev_word.istitle(),
            "-1:word.isupper()": prev_word.isupper(),
            "-1:word.shape": word_shape(prev_word),
            "-1:is_negation": prev_lower in NEGATIONS,
            "-1:is_intensifier": prev_lower in INTENSIFIERS,
            "-1:bigram": f"{prev_lower}_{word_lower}"
        })
    else:
        features["BOS"] = True

    # Konteks Sesudahnya (+1) & Bigram
    if i < len(sent_tokens) - 1:
        next_word = sent_tokens[i + 1]
        next_lower = next_word.lower()
        features.update({
            "+1:word.lower()": next_lower,
            "+1:word.istitle()": next_word.istitle(),
            "+1:word.isupper()": next_word.isupper(),
            "+1:word.shape": word_shape(next_word),
            "+1:bigram": f"{word_lower}_{next_lower}"
        })
    else:
        features["EOS"] = True

    return features

def sentence2features_v2(sent_tokens: List[str]) -> List[Dict[str, Any]]:
    return [token2features_v2(sent_tokens, i) for i in range(len(sent_tokens))]

def run_terma_improvement_v2() -> Dict[str, Any]:
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    figures_dir = os.path.join(REPORTS_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("==================================================================")
    print("MEMULAI PIPELINE PERBAIKAN v2 NLP TERMA (CHAR N-GRAMS & BIGRAMS CRF)")
    print("==================================================================")

    df_train = load_split("train")
    df_val = load_split("valid")
    df_test = load_split("test")

    print(f"Ekstraksi fitur v2 (Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)})...")
    X_train = [sentence2features_v2(t) for t in df_train["tokens"]]
    y_train = [labels for labels in df_train["seq_label"]]

    X_val = [sentence2features_v2(t) for t in df_val["tokens"]]
    y_val = [labels for labels in df_val["seq_label"]]

    X_test = [sentence2features_v2(t) for t in df_test["tokens"]]
    y_test = [labels for labels in df_test["seq_label"]]

    tuning_results = []
    best_micro_f1 = -1.0
    best_params = None
    best_crf = None

    print("\n--- Tuning Hyperparameter CRF v2 pada Validation Set ---")
    for c1 in CRF_C1_VALUES:
        for c2 in CRF_C2_VALUES:
            crf = sklearn_crfsuite.CRF(
                algorithm=CRF_ALGORITHM,
                c1=c1,
                c2=c2,
                max_iterations=CRF_MAX_ITERATIONS,
                all_possible_transitions=True
            )
            crf.fit(X_train, y_train)

            y_val_pred = crf.predict(X_val)
            metrics = compute_span_metrics(y_val, y_val_pred)
            micro_f1 = metrics["GLOBAL"]["F1"]

            tuning_results.append({
                "c1": c1, "c2": c2,
                "val_micro_f1": round(micro_f1, 4),
                "val_aspect_f1": round(metrics["ASPECT"]["F1"], 4),
                "val_sentiment_f1": round(metrics["SENTIMENT"]["F1"], 4)
            })
            print(f"c1={c1:<5} | c2={c2:<5} -> Val Micro-F1: {micro_f1:.4f} (ASPECT: {metrics['ASPECT']['F1']:.4f}, SENTIMENT: {metrics['SENTIMENT']['F1']:.4f})")

            if micro_f1 > best_micro_f1:
                best_micro_f1 = micro_f1
                best_params = {"c1": c1, "c2": c2}
                best_crf = crf

    print(f"\nModel CRF v2 Terbaik: {best_params} dengan Val Micro-F1 = {best_micro_f1:.4f}")

    # Simpan model CRF v2 (TIDAK menimpa crf_best_model.joblib)
    v2_model_path = os.path.join(EXPERIMENTS_DIR, "crf_v2_char_ngrams.joblib")
    joblib.dump(best_crf, v2_model_path)
    print(f"Model v2 disimpan ke: {v2_model_path}")

    # Evaluasi Test Set SEKALI
    y_test_pred_v2 = best_crf.predict(X_test)
    test_eval_v2 = compute_span_metrics(y_test, y_test_pred_v2)

    v2_summary = {
        "model_version": "crf_v2_char_ngrams_and_bigrams",
        "best_params": best_params,
        "val_micro_f1": best_micro_f1,
        "test_metrics": test_eval_v2
    }

    v2_summary_file = os.path.join(EXPERIMENTS_DIR, "crf_v2_final_summary.json")
    with open(v2_summary_file, "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, indent=2)

    # Analisis Audit Typo & Boundary
    from .annotation import extract_spans_from_bio
    all_fn_aspects = []
    for idx, (tokens, true_seq, pred_seq) in enumerate(zip(df_test["tokens"], y_test, y_test_pred_v2)):
        true_spans = set(extract_spans_from_bio(tokens, true_seq))
        pred_spans = set(extract_spans_from_bio(tokens, pred_seq))
        missed = [s for s in (true_spans - pred_spans) if s.label == "ASPECT"]
        for m in missed:
            all_fn_aspects.append({"text": m.text, "tokens": tokens})

    error_v2_summary = {
        "total_test_samples": len(df_test),
        "total_missed_aspects_v2": len(all_fn_aspects),
        "test_eval_metrics": test_eval_v2
    }
    with open(os.path.join(EXPERIMENTS_DIR, "crf_v2_error_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(error_v2_summary, f, indent=2)

    # Baca hasil v1 untuk perbandingan Before vs After
    with open(os.path.join(EXPERIMENTS_DIR, "crf_final_summary.json"), "r", encoding="utf-8") as f:
        v1_test = json.load(f)["test_eval"]

    print("\n=======================================================")
    print("HASIL KOMPARASI FINAL NLP TERMA: BASELINE v1 vs PERBAIKAN v2")
    print("=======================================================")
    print(f"Global Micro-F1 : {v1_test['GLOBAL']['F1']:.4f} (v1) -> {test_eval_v2['GLOBAL']['F1']:.4f} (v2) [Delta: {test_eval_v2['GLOBAL']['F1'] - v1_test['GLOBAL']['F1']:+.4f}]")
    print(f"ASPECT F1       : {v1_test['ASPECT']['F1']:.4f} (v1) -> {test_eval_v2['ASPECT']['F1']:.4f} (v2) [Delta: {test_eval_v2['ASPECT']['F1'] - v1_test['ASPECT']['F1']:+.4f}]")
    print(f"SENTIMENT F1    : {v1_test['SENTIMENT']['F1']:.4f} (v1) -> {test_eval_v2['SENTIMENT']['F1']:.4f} (v2) [Delta: {test_eval_v2['SENTIMENT']['F1'] - v1_test['SENTIMENT']['F1']:+.4f}]")
    print(f"Missed Aspects  : 228 (v1) -> {len(all_fn_aspects)} (v2) [Penurunan FN: {228 - len(all_fn_aspects)} kasus]")

    # Plot visualisasi komparasi
    categories = ["Global Micro-F1", "ASPECT F1", "SENTIMENT F1"]
    v1_vals = [v1_test["GLOBAL"]["F1"], v1_test["ASPECT"]["F1"], v1_test["SENTIMENT"]["F1"]]
    v2_vals = [test_eval_v2["GLOBAL"]["F1"], test_eval_v2["ASPECT"]["F1"], test_eval_v2["SENTIMENT"]["F1"]]

    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, v1_vals, width, label="Baseline CRF v1", color="#bdc3c7")
    plt.bar(x + width/2, v2_vals, width, label="CRF v2 (Char n-grams & Bigrams)", color="#3498db")
    plt.xticks(x, categories)
    plt.ylabel("Exact-Match Span F1 Score")
    plt.title("Komparasi Performa Ekstraksi Span: CRF v1 vs CRF v2 (Test Set)")
    plt.ylim(0.8, 1.0)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    comp_terma_path = os.path.join(figures_dir, "comparison_v1_vs_v2_terma.png")
    plt.savefig(comp_terma_path, dpi=200)
    plt.close()
    print(f"Chart komparasi tersimpan di: {comp_terma_path}")

    return v2_summary

if __name__ == "__main__":
    run_terma_improvement_v2()
