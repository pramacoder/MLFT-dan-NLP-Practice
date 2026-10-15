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

# Dataset specifications
LABEL_LIST = ["O", "B-ASPECT", "I-ASPECT", "B-SENTIMENT", "I-SENTIMENT"]
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}

# Model Hyperparameters
MODEL_NAME = "indobenchmark/indobert-base-p1"
FALLBACK_MODEL_NAME = "indolem/indobert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 3
WEIGHT_DECAY = 0.01

# CRF Hyperparameter Search Space (on Validation)
CRF_C1_VALUES = [0.01, 0.1, 1.0]
CRF_C2_VALUES = [0.01, 0.1, 1.0]
CRF_ALGORITHM = "lbfgs"
CRF_MAX_ITERATIONS = 100
