import os
import requests
import pandas as pd
import json

URLS = {
    "train": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/terma_term-extraction-airy/train_preprocess.txt",
    "valid": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/terma_term-extraction-airy/valid_preprocess.txt",
    "test": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/terma_term-extraction-airy/test_preprocess.txt"
}

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_conll_file(file_content, split_name):
    """
    Parses a tab-separated CoNLL-style file into a list of records.
    Each record: {'id': f'{split_name}_{idx}', 'tokens': [...], 'seq_label': [...], 'text': '...'}
    """
    records = []
    current_tokens = []
    current_labels = []
    sent_idx = 0

    for line in file_content.splitlines():
        line = line.strip()
        if not line:
            if current_tokens:
                records.append({
                    "id": f"{split_name}_{sent_idx}",
                    "tokens": current_tokens,
                    "seq_label": current_labels,
                    "text": " ".join(current_tokens)
                })
                sent_idx += 1
                current_tokens = []
                current_labels = []
        else:
            parts = line.split("\t")
            if len(parts) >= 2:
                token = parts[0].strip()
                label = parts[1].strip()
                current_tokens.append(token)
                current_labels.append(label)
            elif len(parts) == 1:
                current_tokens.append(parts[0].strip())
                current_labels.append("O")

    if current_tokens:
        records.append({
            "id": f"{split_name}_{sent_idx}",
            "tokens": current_tokens,
            "seq_label": current_labels,
            "text": " ".join(current_tokens)
        })
    return records

def download_and_process():
    print("=== Downloading and Processing IndoNLU TermA Dataset ===")
    
    summary = {}
    label_set = set()

    for split, url in URLS.items():
        print(f"\n[1/3] Downloading {split} split from: {url}")
        res = requests.get(url)
        res.raise_for_status()
        raw_text = res.text

        # 1. Save raw txt
        raw_txt_path = os.path.join(DATA_DIR, f"{split}.txt")
        with open(raw_txt_path, "w", encoding="utf-8") as f:
            f.write(raw_text)
        print(f"  -> Raw saved to: {raw_txt_path} ({len(raw_text)} chars)")

        # 2. Parse into structured records
        records = parse_conll_file(raw_text, split)
        df = pd.DataFrame(records)

        # 3. Save as Parquet and JSONL
        parquet_path = os.path.join(DATA_DIR, f"{split}.parquet")
        df.to_parquet(parquet_path, index=False)
        
        jsonl_path = os.path.join(DATA_DIR, f"{split}.jsonl")
        df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
        print(f"  -> Parquet saved: {parquet_path} ({len(df)} rows)")
        print(f"  -> JSONL saved: {jsonl_path}")

        # Collect stats
        total_tokens = sum(len(r["tokens"]) for r in records)
        split_labels = [lbl for r in records for lbl in r["seq_label"]]
        label_set.update(split_labels)
        
        from collections import Counter
        label_counts = Counter(split_labels)

        summary[split] = {
            "sentence_count": len(records),
            "total_tokens": total_tokens,
            "avg_tokens_per_sentence": round(total_tokens / len(records), 2) if records else 0,
            "label_distribution": dict(label_counts)
        }

    # Save summary metadata
    summary_path = os.path.join(DATA_DIR, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Dataset Download & Verification Completed Successfully ===")
    print(f"Classes found: {sorted(list(label_set))}")
    for split, s in summary.items():
        print(f"  * {split.upper()}: {s['sentence_count']} sentences, {s['total_tokens']} tokens (avg: {s['avg_tokens_per_sentence']} tokens/sent)")
        print(f"    Labels: {s['label_distribution']}")

if __name__ == "__main__":
    download_and_process()
