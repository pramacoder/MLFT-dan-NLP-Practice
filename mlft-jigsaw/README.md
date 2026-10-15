# MLFT Jigsaw — Multi-label Toxicity Classification

Proyek ini bertujuan melakukan klasifikasi multi-label teks toksik pada ulasan/komentar Wikipedia dalam 6 dimensi: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, dan `identity_hate` menggunakan arsitektur pretrained Transformer (RoBERTa / DistilRoBERTa).

## Struktur Direktori
```
mlft-jigsaw/
├── data/
│   ├── download_data.py       # Script unduh & verifikasi data Jigsaw
│   ├── train.csv              # Training set lengkap (159,571 komentar)
│   ├── test.csv               # Test set komentar (153,164 komentar)
│   ├── test_labels.csv        # Label test set (-1 untuk un-evaluated rows)
│   ├── split_ids.json         # Indeks split train-validation terstratifikasi
│   └── dataset_summary.json   # Ringkasan data & sebaran 6 label
├── notebooks/
│   └── eda.ipynb              # Notebook EDA khusus multi-label & imbalance
├── src/
│   ├── config.py              # Konfigurasi SEED (42), model, hyperparameter
│   ├── data_loader.py         # Loader data, filter -1 pada test, split stratifikasi
│   ├── preprocessing.py       # Tokenisasi Transformer & target multi-hot float32
│   ├── model.py               # Model Transformer + head 6 output
│   ├── train.py               # Fine-tuning loop dengan BCEWithLogitsLoss
│   ├── evaluation.py          # Metrik Macro/Micro F1, Hamming Loss, per-label
│   └── inference.py           # Pipeline inferensi & thresholding
├── gui/
│   └── app.py                 # Streamlit UI untuk analisis toksisitas komentar
├── experiments/               # Log metrik, kurva training, final_config.json
├── PLAN.md                    # Roadmap lengkap & checklist pengerjaan
└── requirements.txt
```

## Memulai Cepat
1. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
2. Unduh/validasi data:
   ```bash
   python data/download_data.py
   python -m src.data_loader
   ```
