import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
from .config import DATA_DIR, SEED, LABEL_LIST
from .data_loader import load_split, validate_dataset

def extract_spans_from_bio(tokens: List[str], labels: List[str]) -> Dict[str, List[str]]:
    """
    Extracts surface text of ASPECT and SENTIMENT spans from token and BIO label lists.
    Handles standard BIO as well as malformed sequences.
    """
    spans = {"ASPECT": [], "SENTIMENT": []}
    
    current_entity = None
    current_tokens = []

    for token, label in zip(tokens, labels):
        if label.startswith("B-"):
            if current_entity:
                spans[current_entity].append(" ".join(current_tokens))
                current_tokens = []
            current_entity = label.split("-")[1]
            current_tokens.append(token)
        elif label.startswith("I-"):
            entity = label.split("-")[1]
            if current_entity == entity:
                current_tokens.append(token)
            else:
                # Malformed: I- tag without matching B- or switching type
                if current_entity:
                    spans[current_entity].append(" ".join(current_tokens))
                current_entity = entity
                current_tokens = [token]
        else: # 'O'
            if current_entity:
                spans[current_entity].append(" ".join(current_tokens))
                current_entity = None
                current_tokens = []

    if current_entity:
        spans[current_entity].append(" ".join(current_tokens))

    return spans

def generate_validation_sample(df: pd.DataFrame, n_samples: int = 25, seed: int = SEED) -> pd.DataFrame:
    """
    Selects representative samples for manual annotation validation (NLP-T2-04):
    - Includes multi-token aspect spans
    - Includes multi-token sentiment spans
    - Includes sentences with multiple aspect-sentiment pairs
    - Includes instances with illegal transitions if present
    """
    np.random.seed(seed)
    
    # Analyze each sentence
    features = []
    for idx, row in df.iterrows():
        tokens = list(row["tokens"])
        labels = list(row["seq_label"])
        spans = extract_spans_from_bio(tokens, labels)
        
        has_multi_aspect = any(len(s.split()) > 1 for s in spans["ASPECT"])
        has_multi_sentiment = any(len(s.split()) > 1 for s in spans["SENTIMENT"])
        num_aspects = len(spans["ASPECT"])
        num_sentiments = len(spans["SENTIMENT"])
        total_spans = num_aspects + num_sentiments

        # check illegal transitions
        has_illegal = False
        prev = "O"
        for lbl in labels:
            if lbl.startswith("I-"):
                ent = lbl.split("-")[1]
                if prev not in [f"B-{ent}", f"I-{ent}"]:
                    has_illegal = True
                    break
            prev = lbl

        features.append({
            "idx": idx,
            "id": row["id"],
            "tokens": tokens,
            "labels": labels,
            "spans_aspect": spans["ASPECT"],
            "spans_sentiment": spans["SENTIMENT"],
            "has_multi_aspect": has_multi_aspect,
            "has_multi_sentiment": has_multi_sentiment,
            "num_aspects": num_aspects,
            "num_sentiments": num_sentiments,
            "total_spans": total_spans,
            "has_illegal": has_illegal
        })

    feat_df = pd.DataFrame(features)

    # Stratified selection to ensure rich variety:
    selected_indices = set()

    # 1. Sentences with illegal transitions (great for boundary/formatting ambiguity)
    illegal_cands = feat_df[feat_df["has_illegal"]]["idx"].tolist()
    for idx in illegal_cands[:3]:
        selected_indices.add(idx)

    # 2. Sentences with both multi-token aspect and multi-token sentiment
    multi_both = feat_df[feat_df["has_multi_aspect"] & feat_df["has_multi_sentiment"]]["idx"].tolist()
    np.random.shuffle(multi_both)
    for idx in multi_both[:8]:
        selected_indices.add(idx)

    # 3. Sentences with multiple aspects (>= 2 aspects)
    multi_asp = feat_df[feat_df["num_aspects"] >= 2]["idx"].tolist()
    np.random.shuffle(multi_asp)
    for idx in multi_asp[:8]:
        selected_indices.add(idx)

    # 4. Fill remainder with random samples across the set
    all_cands = feat_df["idx"].tolist()
    np.random.shuffle(all_cands)
    for idx in all_cands:
        if len(selected_indices) >= n_samples:
            break
        selected_indices.add(idx)

    # Build final sample dataframe
    sample_rows = []
    for idx in sorted(list(selected_indices))[:n_samples]:
        item = feat_df.loc[feat_df["idx"] == idx].iloc[0]
        
        # Analyze potential ambiguity notes
        status = "ok"
        notes = []
        
        asp_str = "; ".join(item["spans_aspect"]) if item["spans_aspect"] else "(tidak ada)"
        snt_str = "; ".join(item["spans_sentiment"]) if item["spans_sentiment"] else "(tidak ada)"
        
        # Heuristic hints for reviewer
        tok_text = " ".join(item["tokens"]).lower()
        if item["has_illegal"]:
            status = "ambigu"
            notes.append("Terdapat transisi label ilegal (I-X tanpa B-X pendahulu)")
        if any(w in tok_text for w in ["sangat", "kurang", "banget", "terlalu", "cukup", "agak"]):
            notes.append("Memuat kata penguat/modifikator (cek batas span SENTIMENT)")
        if any(w in tok_text for w in ["kamar mandi", "air panas", "kolam renang", "tempat parkir", "resepsionis"]):
            notes.append("Memuat aspek frasa majemuk (cek kelengkapan batas ASPECT)")
        if item["num_aspects"] != item["num_sentiments"]:
            notes.append(f"Jumlah aspek ({item['num_aspects']}) != sentimen ({item['num_sentiments']}) (kemungkinan aspek implisit atau sentimen jamak)")

        sample_rows.append({
            "id": item["id"],
            "tokens": " ".join(item["tokens"]),
            "label": " ".join(item["labels"]),
            "span_ASPECT": asp_str,
            "span_SENTIMENT": snt_str,
            "status": status,
            "catatan": " | ".join(notes) if notes else "Anotasi konsisten & jelas"
        })

    return pd.DataFrame(sample_rows)

if __name__ == "__main__":
    df_train = load_split("train")
    sample_df = generate_validation_sample(df_train, n_samples=25, seed=SEED)
    output_path = os.path.join(DATA_DIR, "sampel_validasi.csv")
    sample_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Sampel validasi {len(sample_df)} contoh berhasil disimpan di: {output_path}")
