import os
import json
import joblib
import pandas as pd
from typing import Dict, List, Any, Tuple
import sklearn_crfsuite

from .config import (
    DATA_DIR, EXPERIMENTS_DIR, SEED,
    CRF_C1_VALUES, CRF_C2_VALUES, CRF_ALGORITHM, CRF_MAX_ITERATIONS
)
from .data_loader import load_split
from .crf_features import sentence2features, sentence2labels
from .evaluate import compute_span_metrics

def prepare_crf_data(df: pd.DataFrame) -> Tuple[List[List[Dict[str, Any]]], List[List[str]]]:
    X = [sentence2features(tokens) for tokens in df["tokens"]]
    y = [sentence2labels(labels) for labels in df["seq_label"]]
    return X, y

def train_and_tune_crf() -> Dict[str, Any]:
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    print("=== [NLP-T8] Memuat Data Train, Validation, dan Test untuk CRF ===")
    df_train = load_split("train")
    df_val = load_split("valid")
    df_test = load_split("test")

    print(f"Ekstraksi fitur kalimat (Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)})...")
    X_train, y_train = prepare_crf_data(df_train)
    X_val, y_val = prepare_crf_data(df_val)
    X_test, y_test = prepare_crf_data(df_test)

    tuning_results = []
    best_micro_f1 = -1.0
    best_params = None
    best_crf = None

    print(f"\n--- Grid Search Hyperparameter CRF pada Validation Set ---")
    print(f"Ruang parameter: c1 in {CRF_C1_VALUES}, c2 in {CRF_C2_VALUES}")

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

            # Prediksi validation
            y_val_pred = crf.predict(X_val)
            metrics = compute_span_metrics(y_val, y_val_pred)
            micro_f1 = metrics["GLOBAL"]["F1"]

            record = {
                "c1": c1,
                "c2": c2,
                "algorithm": CRF_ALGORITHM,
                "val_micro_f1": round(micro_f1, 4),
                "val_aspect_f1": round(metrics["ASPECT"]["F1"], 4),
                "val_sentiment_f1": round(metrics["SENTIMENT"]["F1"], 4),
                "val_metrics": metrics
            }
            tuning_results.append(record)

            print(f"c1={c1:<5} | c2={c2:<5} -> Val Micro-F1: {micro_f1:.4f} (ASPECT: {metrics['ASPECT']['F1']:.4f}, SENTIMENT: {metrics['SENTIMENT']['F1']:.4f})")

            if micro_f1 > best_micro_f1:
                best_micro_f1 = micro_f1
                best_params = {"c1": c1, "c2": c2}
                best_crf = crf

    print(f"\nKonfigurasi CRF Terbaik: {best_params} dengan Val Micro-F1 = {best_micro_f1:.4f}")

    # Simpan hasil tuning ke JSON (NLP-T8-01 DoD)
    tuning_log_path = os.path.join(EXPERIMENTS_DIR, "crf_tuning_results.json")
    with open(tuning_log_path, "w", encoding="utf-8") as f:
        json.dump(tuning_results, f, indent=2)

    # Simpan model terbaik
    model_path = os.path.join(EXPERIMENTS_DIR, "crf_best_model.joblib")
    joblib.dump(best_crf, model_path)
    print(f"Model CRF terbaik disimpan ke: {model_path}")

    # Prediksi validation & test dengan model terbaik (NLP-T8-02)
    y_val_pred_best = best_crf.predict(X_val)
    y_test_pred_best = best_crf.predict(X_test)

    val_eval_final = compute_span_metrics(y_val, y_val_pred_best)
    test_eval_final = compute_span_metrics(y_test, y_test_pred_best)

    print("\n=== Evaluasi Final CRF pada Validation & Test ===")
    print(f"Validation Micro-F1 : {val_eval_final['GLOBAL']['F1']:.4f} (P: {val_eval_final['GLOBAL']['Precision']:.4f}, R: {val_eval_final['GLOBAL']['Recall']:.4f})")
    print(f"Test Micro-F1       : {test_eval_final['GLOBAL']['F1']:.4f} (P: {test_eval_final['GLOBAL']['Precision']:.4f}, R: {test_eval_final['GLOBAL']['Recall']:.4f})")
    print(f"  - ASPECT    (Test): P={test_eval_final['ASPECT']['Precision']:.4f}, R={test_eval_final['ASPECT']['Recall']:.4f}, F1={test_eval_final['ASPECT']['F1']:.4f}")
    print(f"  - SENTIMENT (Test): P={test_eval_final['SENTIMENT']['Precision']:.4f}, R={test_eval_final['SENTIMENT']['Recall']:.4f}, F1={test_eval_final['SENTIMENT']['F1']:.4f}")

    # Simpan prediksi ber-ID
    val_pred_records = [
        {"id": row["id"], "tokens": list(row["tokens"]), "y_true": list(true_lbl), "y_pred": list(pred_lbl)}
        for (_, row), true_lbl, pred_lbl in zip(df_val.iterrows(), y_val, y_val_pred_best)
    ]
    test_pred_records = [
        {"id": row["id"], "tokens": list(row["tokens"]), "y_true": list(true_lbl), "y_pred": list(pred_lbl)}
        for (_, row), true_lbl, pred_lbl in zip(df_test.iterrows(), y_test, y_test_pred_best)
    ]

    with open(os.path.join(EXPERIMENTS_DIR, "crf_val_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(val_pred_records, f, indent=2, ensure_ascii=False)
    with open(os.path.join(EXPERIMENTS_DIR, "crf_test_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(test_pred_records, f, indent=2, ensure_ascii=False)

    summary = {
        "best_params": best_params,
        "validation_eval": val_eval_final,
        "test_eval": test_eval_final
    }
    with open(os.path.join(EXPERIMENTS_DIR, "crf_final_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary

if __name__ == "__main__":
    train_and_tune_crf()
