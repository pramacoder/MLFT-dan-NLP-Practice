import os
import requests
import pandas as pd
import json

BASE_URL = "https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge/resolve/main"

FILES = [
    "train.csv",
    "test.csv",
    "test_labels.csv",
    "sample_submission.csv"
]

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def download_file_stream(filename):
    dest_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"File {filename} already exists ({os.path.getsize(dest_path) / (1024*1024):.2f} MB), skipping download.")
        return dest_path

    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename} from {url}...")
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB
        
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r  {filename}: {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({percent:.1f}%)", end="", flush=True)
                    else:
                        print(f"\r  {filename}: {downloaded / (1024*1024):.1f} MB", end="", flush=True)
    print(f"\n  -> Finished downloading {filename} to {dest_path}")
    return dest_path

def inspect_and_summarize():
    print("\n=== Inspecting and Verifying Downloaded Jigsaw Data ===")
    
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    test_labels_path = os.path.join(DATA_DIR, "test_labels.csv")

    label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    # 1. Inspect train.csv
    print("\n[1/3] Inspecting train.csv...")
    df_train = pd.read_csv(train_path)
    print(f"  Train shape: {df_train.shape} (rows, cols)")
    print(f"  Columns: {list(df_train.columns)}")
    train_label_sums = df_train[label_cols].sum().to_dict()
    clean_train_count = int((df_train[label_cols].sum(axis=1) == 0).sum())
    toxic_train_count = int((df_train[label_cols].sum(axis=1) > 0).sum())

    # 2. Inspect test.csv
    print("\n[2/3] Inspecting test.csv...")
    df_test = pd.read_csv(test_path)
    print(f"  Test comments shape: {df_test.shape}")

    # 3. Inspect test_labels.csv and check for -1 labels
    print("\n[3/3] Inspecting test_labels.csv (checking Kaggle -1 un-evaluated rows)...")
    df_test_labels = pd.read_csv(test_labels_path)
    print(f"  Test labels shape: {df_test_labels.shape}")

    # Check for -1 in test_labels
    unscored_mask = (df_test_labels[label_cols] == -1).any(axis=1)
    unscored_count = int(unscored_mask.sum())
    valid_test_count = int((~unscored_mask).sum())
    
    print(f"  Total test rows: {len(df_test_labels)}")
    print(f"  Unscored test rows (value -1, must be discarded in evaluation): {unscored_count} ({unscored_count / len(df_test_labels) * 100:.2f}%)")
    print(f"  Valid scored test rows: {valid_test_count} ({valid_test_count / len(df_test_labels) * 100:.2f}%)")

    valid_test_df = df_test_labels[~unscored_mask]
    test_label_sums = valid_test_df[label_cols].sum().to_dict()

    summary = {
        "train": {
            "total_rows": len(df_train),
            "label_columns": label_cols,
            "clean_comments_count": clean_train_count,
            "toxic_comments_count": toxic_train_count,
            "label_sums": train_label_sums
        },
        "test": {
            "total_rows": len(df_test),
            "test_labels_total_rows": len(df_test_labels),
            "unscored_rows_with_minus_one": unscored_count,
            "valid_scored_rows": valid_test_count,
            "valid_test_label_sums": test_label_sums
        }
    }

    summary_file = os.path.join(DATA_DIR, "dataset_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary successfully written to {summary_file}")

if __name__ == "__main__":
    for f in FILES:
        download_file_stream(f)
    inspect_and_summarize()
