import os
import random
import numpy as np

# Reproducibility Seed
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

# Label Definitions (6 Dimensions)
LABEL_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]
NUM_LABELS = len(LABEL_COLS)
ID2LABEL = {i: label for i, label in enumerate(LABEL_COLS)}
LABEL2ID = {label: i for i, label in enumerate(LABEL_COLS)}

# Modeling Defaults (as recommended in Plan)
MODEL_NAME = "roberta-base"
LIGHT_MODEL_NAME = "distilroberta-base"
MAX_LEN = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3
WEIGHT_DECAY = 0.01

# Split Settings
VAL_SPLIT_RATIO = 0.10
