"""
Modul Arsitektur Model untuk Jigsaw Toxicity (ML-T5-01).

Tanggung Jawab:
- Memuat model Pretrained Transformer (roberta-base atau distilroberta-base).
- Mengonfigurasi classification head dengan 6 logit keluaran (linear layer).
- Menggunakan problem_type='multi_label_classification' dengan BCEWithLogitsLoss saat training.
- Menyediakan fungsi inferensi probabilitas dengan fungsi Sigmoid (bukan Softmax)
  karena label toksisitas tidak saling eksklusif (bisa 0, 1, atau banyak label aktif sekaligus).
"""

import os
torch_lib = r"C:\laragon\bin\python\python-3.10\lib\site-packages\torch\lib"
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)

from typing import Dict, Any, Tuple
from .config import MODEL_NAME, LIGHT_MODEL_NAME, NUM_LABELS, LABEL_COLS

def create_model(model_name: str = LIGHT_MODEL_NAME, num_labels: int = NUM_LABELS):
    """
    Inisialisasi arsitektur Pretrained Transformer untuk Multi-label Classification.
    """
    from transformers import AutoModelForSequenceClassification, AutoConfig

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label={i: l for i, l in enumerate(LABEL_COLS)},
        label2id={l: i for i, l in enumerate(LABEL_COLS)}
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config
    )
    return model

def compute_sigmoid_probabilities(logits):
    """
    Menghitung probabilitas per label menggunakan Sigmoid independen pada tiap logit.
    P(Y_i = 1 | X) = 1 / (1 + exp(-logit_i)).
    """
    import numpy as np
    return 1.0 / (1.0 + np.exp(-logits))
