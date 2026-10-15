import os
import json
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from .config import DATA_DIR, LABEL_COLS, SEED, VAL_SPLIT_RATIO

def load_train_data() -> pd.DataFrame:
    """
    Loads raw train.csv.
    """
    path = os.path.join(DATA_DIR, "train.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"train.csv not found at {path}. Run data/download_data.py first.")
    df = pd.read_csv(path)
    return df

def load_test_data(drop_unscored: bool = True) -> pd.DataFrame:
    """
    Loads test.csv and merges with test_labels.csv by 'id'.
    If drop_unscored is True, discards rows where label values are -1 (Kaggle un-evaluated test rows).
    """
    test_path = os.path.join(DATA_DIR, "test.csv")
    labels_path = os.path.join(DATA_DIR, "test_labels.csv")

    if not os.path.exists(test_path) or not os.path.exists(labels_path):
        raise FileNotFoundError("test.csv or test_labels.csv not found. Run data/download_data.py first.")

    df_test = pd.read_csv(test_path)
    df_labels = pd.read_csv(labels_path)

    merged = pd.merge(df_test, df_labels, on="id")

    if drop_unscored:
        # Rows with -1 in any label column are un-evaluated rows in Kaggle
        unscored_mask = (merged[LABEL_COLS] == -1).any(axis=1)
        valid_merged = merged[~unscored_mask].copy().reset_index(drop=True)
        return valid_merged
    return merged

def get_train_val_split(val_ratio: float = VAL_SPLIT_RATIO, force_recreate: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates or loads a reproducible train/validation split using Multilabel Stratified Shuffle Split (ML-T4-01).
    Saves split indices to data/split_ids.json (ML-T4-02).
    """
    split_file = os.path.join(DATA_DIR, "split_ids.json")
    df_train_full = load_train_data()

    if os.path.exists(split_file) and not force_recreate:
        with open(split_file, "r") as f:
            split_ids = json.load(f)
        train_idx = split_ids["train_indices"]
        val_idx = split_ids["val_indices"]
        df_train = df_train_full.iloc[train_idx].reset_index(drop=True)
        df_val = df_train_full.iloc[val_idx].reset_index(drop=True)
        return df_train, df_val

    # Create stratified multi-label split
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=SEED)
        X = df_train_full["comment_text"].values
        Y = df_train_full[LABEL_COLS].values

        train_idx, val_idx = next(msss.split(X, Y))
    except ImportError:
        # Fallback if iterative-stratification is not installed
        np.random.seed(SEED)
        indices = np.random.permutation(len(df_train_full))
        val_size = int(len(df_train_full) * val_ratio)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

    split_ids = {
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "seed": SEED,
        "val_ratio": val_ratio
    }

    with open(split_file, "w") as f:
        json.dump(split_ids, f, indent=2)

    df_train = df_train_full.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_full.iloc[val_idx].reset_index(drop=True)
    return df_train, df_val

if __name__ == "__main__":
    print("Testing Jigsaw data loader...")
    df_train_full = load_train_data()
    print(f"Loaded train: {len(df_train_full)} rows")

    df_train, df_val = get_train_val_split()
    print(f"Split train: {len(df_train)} rows | val: {len(df_val)} rows")

    df_test_valid = load_test_data(drop_unscored=True)
    print(f"Loaded valid scored test comments: {len(df_test_valid)} rows")
