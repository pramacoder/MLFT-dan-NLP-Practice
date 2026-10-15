import os
import sys
import json
import pandas as pd
import numpy as np
from collections import Counter
from tokenizers import Tokenizer

# -------------------------------------------------------------
# 1. TermA: Feature Dictionaries & Lexical Marker Distribution
# -------------------------------------------------------------
sys.path.append(os.path.abspath("nlp-terma"))
from src.crf_features import sentence2features
from src.data_loader import load_split as load_terma_split

test_sent = ["kamar", "mandi", "sangat", "bersih", "."]
feats = sentence2features(test_sent)

print("=== REAL CRF FEATURE DICTS ===")
print("Token 0 ('kamar'):")
print(json.dumps(feats[0], indent=2))
print("\nToken 2 ('sangat'):")
print(json.dumps(feats[2], indent=2))

df_train_terma = load_terma_split("train")

lexical_words = ["sangat", "banget", "tidak", "kurang"]
word_label_stats = {w: Counter() for w in lexical_words}

for idx, row in df_train_terma.iterrows():
    tokens = [t.lower() for t in row["tokens"]]
    labels = list(row["seq_label"])
    for t, l in zip(tokens, labels):
        if t in word_label_stats:
            word_label_stats[t][l] += 1

print("\n=== TERMA LEXICAL MARKERS IN TRAIN SET ===")
marker_summary = {}
for w, cnt in word_label_stats.items():
    total = sum(cnt.values())
    snt_count = cnt["B-SENTIMENT"] + cnt["I-SENTIMENT"]
    o_count = cnt["O"]
    asp_count = cnt["B-ASPECT"] + cnt["I-ASPECT"]
    snt_pct = (snt_count / total * 100) if total > 0 else 0
    o_pct = (o_count / total * 100) if total > 0 else 0
    asp_pct = (asp_count / total * 100) if total > 0 else 0
    marker_summary[w] = {
        "total": total,
        "sentiment_count": snt_count,
        "sentiment_pct": round(snt_pct, 2),
        "O_count": o_count,
        "O_pct": round(o_pct, 2),
        "aspect_count": asp_count,
        "aspect_pct": round(asp_pct, 2),
        "breakdown": dict(cnt)
    }
    print(f"'{w}': Total={total}, SENTIMENT={snt_count} ({snt_pct:.2f}%), O={o_count} ({o_pct:.2f}%), ASPECT={asp_count} ({asp_pct:.2f}%)")

# -------------------------------------------------------------
# 2. Jigsaw: Truncation per label & 'gay' keyword analysis
# -------------------------------------------------------------
sys.path.insert(0, os.path.abspath("mlft-jigsaw"))
for mod in list(sys.modules.keys()):
    if mod == "src" or mod.startswith("src."):
        del sys.modules[mod]
from src.data_loader import load_train_data as load_jigsaw_train
from src.config import LABEL_COLS

df_train_jigsaw = load_jigsaw_train()
tok = Tokenizer.from_pretrained("distilroberta-base")

# Calculate lengths for all comments or sample
print("\n=== JIGSAW TRUNCATION PER LABEL ===")
# To be fast and exact, we can compute token lengths for all toxic comments + random clean sample
toxic_indices = df_train_jigsaw[df_train_jigsaw[LABEL_COLS].sum(axis=1) > 0].index
clean_sample_indices = df_train_jigsaw[df_train_jigsaw[LABEL_COLS].sum(axis=1) == 0].sample(20000, random_state=42).index
eval_indices = toxic_indices.union(clean_sample_indices)
sub_df = df_train_jigsaw.loc[eval_indices].copy()

sub_df["tokens_len"] = [len(tok.encode(str(t)).ids) for t in sub_df["comment_text"]]

label_truncation = {}
for col in LABEL_COLS:
    col_df = sub_df[sub_df[col] == 1]
    trunc_128 = (col_df["tokens_len"] > 128).mean() * 100
    trunc_256 = (col_df["tokens_len"] > 256).mean() * 100
    median_len = col_df["tokens_len"].median()
    mean_len = col_df["tokens_len"].mean()
    label_truncation[col] = {
        "n_pos": len(col_df),
        "median_len": float(median_len),
        "mean_len": round(float(mean_len), 2),
        "trunc_128_pct": round(float(trunc_128), 2),
        "trunc_256_pct": round(float(trunc_256), 2)
    }
    print(f"[{col}] N={len(col_df)}, Median={median_len}, Mean={mean_len:.1f}, Trunc@128={trunc_128:.2f}%, Trunc@256={trunc_256:.2f}%")

# Clean comments truncation
clean_df = sub_df[sub_df[LABEL_COLS].sum(axis=1) == 0]
clean_trunc_128 = (clean_df["tokens_len"] > 128).mean() * 100
clean_trunc_256 = (clean_df["tokens_len"] > 256).mean() * 100
print(f"[clean_comments] N={len(clean_df)}, Median={clean_df['tokens_len'].median()}, Trunc@128={clean_trunc_128:.2f}%, Trunc@256={clean_trunc_256:.2f}%")

# 'gay' keyword analysis in full train set
print("\n=== JIGSAW 'GAY' KEYWORD ANALYSIS (TRAIN SET) ===")
# regex whole word \bgay\b or substring
gay_mask = df_train_jigsaw["comment_text"].str.contains(r"\bgay\b", case=False, na=False, regex=True)
gay_total = int(gay_mask.sum())
gay_df = df_train_jigsaw[gay_mask]

gay_label_counts = gay_df[LABEL_COLS].sum().to_dict()
gay_toxic_total = int((gay_df[LABEL_COLS].sum(axis=1) > 0).sum())
gay_clean_total = gay_total - gay_toxic_total

gay_stats = {
    "total_occurrences": gay_total,
    "pct_of_all_train": round((gay_total / len(df_train_jigsaw)) * 100, 3),
    "clean_count": gay_clean_total,
    "clean_pct": round((gay_clean_total / gay_total) * 100, 2),
    "toxic_any_count": gay_toxic_total,
    "toxic_any_pct": round((gay_toxic_total / gay_total) * 100, 2),
    "label_breakdown": gay_label_counts,
    "label_percentages": {k: round(v / gay_total * 100, 2) for k, v in gay_label_counts.items()}
}
print(json.dumps(gay_stats, indent=2))

output_data = {
    "terma_marker_stats": marker_summary,
    "jigsaw_label_truncation": label_truncation,
    "jigsaw_gay_stats": gay_stats,
    "feats_token0": feats[0],
    "feats_token2": feats[2]
}
with open("audit_stats.json", "w") as f:
    json.dump(output_data, f, indent=2)
