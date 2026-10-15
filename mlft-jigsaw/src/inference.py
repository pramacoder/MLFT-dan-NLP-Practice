"""
Modul Inferensi untuk Jigsaw Toxicity (ML-T12, Tahap II).

Tanggung Jawab:
- Memuat model hasil fine-tuning dan konfigurasi threshold terpilih.
- Menerima raw text komentar, melakukan tokenisasi, dan menghitung probabilitas sigmoid untuk 6 label.
- Mengembalikan prediksi label aktif berdasarkan threshold yang telah dibekukan.
"""

from typing import Dict, List, Any
import numpy as np
from .config import LABEL_COLS, MAX_LEN
from .model import compute_sigmoid_probabilities

class ToxicityPredictor:
    """
    Predictor inference konsisten yang digunakan oleh CLI, script evaluasi, dan GUI Streamlit.
    """
    def __init__(self, model=None, tokenizer=None, threshold: float = 0.5):
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold

    def predict_one(self, text: str) -> Dict[str, Any]:
        """
        Prediksi probabilitas 6 dimensi dan status aktif per label untuk satu komentar.
        """
        if not text or not text.strip():
            return {
                "probabilities": {col: 0.0 for col in LABEL_COLS},
                "active_labels": [],
                "is_toxic": False
            }

        # Jika model nyata belum dimuat (mock mode untuk testing skeleton GUI)
        if self.model is None or self.tokenizer is None:
            # Fallback mock heuristic
            lower = text.lower()
            mock_probs = {}
            for col in LABEL_COLS:
                # simple mock based on key phrases for testing skeleton GUI
                score = 0.85 if col in lower or ("bad" in lower and col == "toxic") else 0.02
                mock_probs[col] = float(score)
            active = [c for c, p in mock_probs.items() if p >= self.threshold]
            return {
                "probabilities": mock_probs,
                "active_labels": active,
                "is_toxic": len(active) > 0,
                "threshold_used": self.threshold
            }

        # Inference riil dengan transformer model
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        outputs = self.model(**encoded)
        logits = outputs.logits.detach().cpu().numpy()[0]
        probs = compute_sigmoid_probabilities(logits)

        prob_dict = {col: float(probs[i]) for i, col in enumerate(LABEL_COLS)}
        active = [col for col, p in prob_dict.items() if p >= self.threshold]

        return {
            "probabilities": prob_dict,
            "active_labels": active,
            "is_toxic": len(active) > 0,
            "threshold_used": self.threshold
        }

def load_trained_predictor(checkpoint_dir: str = None) -> ToxicityPredictor:
    """
    Factory function untuk memuat model terbaik yang telah dibekukan.
    Otomatis membaca final_config.json jika tersedia.
    """
    import os
    import json
    import torch
    from transformers import AutoTokenizer
    from .config import EXPERIMENTS_DIR, LIGHT_MODEL_NAME
    from .model import create_model

    exp_dir = checkpoint_dir or EXPERIMENTS_DIR
    cfg_path = os.path.join(exp_dir, "final_config.json")

    if not os.path.exists(cfg_path):
        # Fallback ke mock mode jika eksperimen belum selesai
        return ToxicityPredictor(model=None, tokenizer=None, threshold=0.5)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    best_exp = cfg.get("best_experiment", "exp1_baseline")
    threshold = cfg.get("frozen_threshold", 0.5)
    model_name = cfg.get("model_name", LIGHT_MODEL_NAME)
    ckpt_path = os.path.join(exp_dir, best_exp, "best_model.pt")

    if not os.path.exists(ckpt_path):
        return ToxicityPredictor(model=None, tokenizer=None, threshold=threshold)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = create_model(model_name=model_name)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return ToxicityPredictor(model=model, tokenizer=tokenizer, threshold=threshold)

