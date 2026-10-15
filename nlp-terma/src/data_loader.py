import os
import pandas as pd
from typing import Dict, List, Tuple
from .config import DATA_DIR, LABEL_LIST

def load_split(split: str = "train") -> pd.DataFrame:
    """
    Loads the official split (train, valid, or test) for IndoNLU TermA.
    Reads from the local parquet file.
    """
    valid_splits = ["train", "valid", "test"]
    if split not in valid_splits:
        raise ValueError(f"split must be one of {valid_splits}, got '{split}'")

    parquet_path = os.path.join(DATA_DIR, f"{split}.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found at {parquet_path}. Run download_data.py first.")

    df = pd.read_parquet(parquet_path)
    return df

def validate_dataset(df: pd.DataFrame) -> Dict:
    """
    Validates the dataset annotations automatically (NLP-T2-03):
    - checks that token count == label count for every sentence
    - checks for illegal transitions (e.g. O -> I-X, B-ASPECT -> I-SENTIMENT)
    - checks for empty sentences
    """
    issues = {
        "mismatched_lengths": [],
        "empty_sentences": [],
        "invalid_labels": [],
        "illegal_transitions": []
    }

    valid_labels_set = set(LABEL_LIST)

    for idx, row in df.iterrows():
        tokens = list(row["tokens"])
        labels = list(row["seq_label"])
        sent_id = row["id"]

        # Check empty
        if len(tokens) == 0:
            issues["empty_sentences"].append(sent_id)
            continue

        # Check length match
        if len(tokens) != len(labels):
            issues["mismatched_lengths"].append({
                "id": sent_id,
                "n_tokens": len(tokens),
                "n_labels": len(labels)
            })

        # Check illegal labels & transitions
        prev_label = "O"
        for i, (tok, lbl) in enumerate(zip(tokens, labels)):
            if lbl not in valid_labels_set:
                issues["invalid_labels"].append({"id": sent_id, "token": tok, "label": lbl})

            # Transition check:
            # e.g., I-ASPECT must follow B-ASPECT or I-ASPECT
            if lbl.startswith("I-"):
                entity_type = lbl.split("-")[1]
                expected_prev = [f"B-{entity_type}", f"I-{entity_type}"]
                if prev_label not in expected_prev:
                    issues["illegal_transitions"].append({
                        "id": sent_id,
                        "token_idx": i,
                        "token": tok,
                        "prev_label": prev_label,
                        "curr_label": lbl
                    })
            prev_label = lbl

    return issues

if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        df = load_split(split)
        issues = validate_dataset(df)
        print(f"[{split.upper()}] Rows: {len(df)}")
        print(f"  Empty sentences: {len(issues['empty_sentences'])}")
        print(f"  Length mismatches: {len(issues['mismatched_lengths'])}")
        print(f"  Invalid labels: {len(issues['invalid_labels'])}")
        print(f"  Illegal transitions: {len(issues['illegal_transitions'])}")
