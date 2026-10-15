# NLP TermA — Aspect & Sentiment Extraction (IndoNLU)

Proyek ini bertujuan melakukan sequence labeling BIO untuk ekstraksi span `ASPECT` dan `SENTIMENT` (opini) pada ulasan hotel AiryRooms berbahasa Indonesia menggunakan IndoNLU TermA.

## Struktur Direktori
```
nlp-terma/
├── data/
│   ├── download_data.py       # Script unduh & parser data resmi
│   ├── train.txt / .parquet   # Split training resmi (3,000 ulasan)
│   ├── valid.txt / .parquet   # Split validasi resmi (1,000 ulasan)
│   ├── test.txt / .parquet    # Split test resmi (1,000 ulasan)
│   └── dataset_summary.json   # Ringkasan statistik & sebaran label
├── notebooks/
│   └── eda.ipynb              # Notebook analisis data eksploratif
├── src/
│   ├── config.py              # Konfigurasi seed (42), path, hyperparameter
│   ├── data_loader.py         # Loader split & validator anotasi
│   ├── annotation.py          # Validasi token-label BIO
│   ├── eda.py                 # Modul analisis statistik teks & label
│   ├── crf_features.py        # Ekstraksi fitur CRF (word_shape, konteks ±1)
│   ├── train_crf.py           # Pelatihan CRF & tuning validation
│   ├── transformer_data.py    # Subword tokenization & alignment label
│   ├── train_transformer.py   # Fine-tuning IndoBERT
│   ├── spans.py               # Rekonstruksi span & normalisasi BIO
│   └── evaluate.py            # Evaluasi exact-match span P/R/F1
├── gui/
│   └── app.py                 # Streamlit UI untuk interaksi model
├── experiments/               # Log metrik & konfigurasi eksperimen
├── reports/                   # Laporan Tahap I & II
├── PLAN.md                    # Detail roadmap & checklist pengerjaan
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
