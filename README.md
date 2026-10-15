# MLFT-dan-NLP-Practice

Repositori praktikum dan proyek komprehensif untuk mata kuliah **Machine Learning for Text (MLFT)** dan **Natural Language Processing (NLP)**.

---

## 📌 Ringkasan Proyek

Repositori ini memuat implementasi *end-to-end* (mulai dari formulasi masalah, validasi data, EDA, split anti-leakage, rekayasa fitur, pemodelan, optimasi threshold, analisis kesalahan, hingga antarmuka GUI Streamlit) untuk 2 proyek utama:

### 1. Project NLP — TermA (IndoNLU: Aspect & Sentiment Extraction)
* **Tugas**: Ekstraksi span berbasis urutan (*Sequence Labeling BIO*) untuk entitas `ASPECT` dan `SENTIMENT` pada ulasan hotel AiryRooms Indonesia.
* **Model**: Linear-Chain Conditional Random Field (CRF) dengan fitur leksikal, ortografis, morfologis, negasi, penguat, serta **Character 3-grams** dan **Contextual Bigrams** (v2).
* **Metrik Utama**: **Exact-Match Span Micro-F1** (standar CoNLL / seqeval), mengabaikan label `O`.
* **Hasil Test Set**: Micro Span F1 **0,8763** (SENTIMENT Precision: **90,26%**, F1: **0,8942**).
* **Laporan Lengkap**:
  * [Laporan Tahap I: Validasi Anotasi & Rekayasa Fitur](nlp-terma/reports/laporan_tahap_1.md)
  * [Laporan Tahap II: Pelatihan CRF, Evaluasi Span & Error Analysis](nlp-terma/reports/laporan_tahap_2.md)
  * [Laporan Tahap III: Evaluasi Perbaikan v2 (Char n-grams)](nlp-terma/reports/laporan_tahap_3_perbaikan_v2.md)

### 2. Project MLFT — Jigsaw Toxicity (Multi-label Binary Classification)
* **Tugas**: Klasifikasi multi-label teks komentar Wikipedia ke dalam 6 dimensi toksisitas: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, dan `identity_hate`.
* **Model**: Pretrained Transformer **`distilroberta-base`** dengan *Classification Head* 6 logit independen.
* **Loss Function**: `BCEWithLogitsLoss` dengan pembobotan kelas teredam (*damped pos_weight*) untuk menangani ketimpangan kelas ekstrem (*extreme imbalance*).
* **Inovasi v2**: **Head + Tail Truncation** (64 token awal + 64 token akhir) untuk menyelamatkan sinyal toksik di penutup teks panjang, serta **Adaptive Per-Class Thresholds**.
* **Hasil Test Set**: Micro Recall **80,00%**, Toxic Recall **83,33%** (True Positive: 5/6 kasus), Hamming Loss **0,0233 - 0,0683**.
* **Laporan Lengkap**:
  * [Laporan Tahap I: Formulasi, EDA & Desain Sistem](mlft-jigsaw/reports/laporan_tahap_1.md)
  * [Laporan Tahap II: Pelatihan Transformer, Tuning Threshold & Error Analysis](mlft-jigsaw/reports/laporan_tahap_2.md)
  * [Laporan Tahap III: Evaluasi Perbaikan v2 (Head-Tail Truncation)](mlft-jigsaw/reports/laporan_tahap_3_perbaikan_v2.md)

---

## 📂 Struktur Repositori

```text
.
├── 00_INDEX_DAN_ATURAN_AGENT.md      # Panduan indeks dataset & aturan integritas
├── 01_NLP_TermA_plan.md               # Rencana kerja proyek TermA (Tahap I & II)
├── 02_MLFT_Jigsaw_plan.md             # Rencana kerja proyek Jigsaw (Tahap I & II)
├── 03_IMPROVEMENT_PLAN.md             # Roadmap perbaikan bertarget & tata kelola v2
├── mlft-jigsaw/
│   ├── data/                          # Skrip unduh data & metadata split
│   ├── experiments/                   # Konfigurasi beku (JSON), log metrik, prediksi test
│   ├── gui/                           # Antarmuka interaktif Streamlit
│   ├── notebooks/                     # Notebook EDA interaktif
│   ├── reports/                       # Laporan Tahap 1, 2, dan 3 (v2) beserta visualisasi
│   └── src/                           # Kode modular (data_loader, model, train, eval)
└── nlp-terma/
    ├── data/                          # Dataset split IndoNLU TermA
    ├── experiments/                   # Model CRF joblib, ringkasan evaluasi, error analysis
    ├── gui/                           # Antarmuka interaktif Streamlit
    ├── notebooks/                     # Notebook EDA validasi anotasi
    ├── reports/                       # Laporan Tahap 1, 2, dan 3 (v2) beserta visualisasi
    └── src/                           # Ekstraksi fitur CRF, BIO reconstruction, training
```

---

## 🚀 Cara Menjalankan

### 1. Menjalankan GUI Interaktif (Streamlit)

#### Demo Jigsaw Toxicity Classifier:
```bash
cd mlft-jigsaw/gui
streamlit run app.py
```

#### Demo IndoNLU TermA Span Highlighter:
```bash
cd nlp-terma/gui
streamlit run app.py
```

### 2. Menjalankan Pelatihan & Evaluasi

```bash
# Eksperimen Perbaikan v2 Jigsaw
cd mlft-jigsaw
python -m src.run_improvement_v2

# Eksperimen Perbaikan v2 TermA
cd nlp-terma
python -m src.train_crf_v2
```

---

## 🛡️ Reproducibility & Integritas Metodologi
* **Random Seed**: `42` pada seluruh split data, inisialisasi bobot, dan data loader.
* **Anti-Data Leakage**: Test set bersih hanya dievaluasi SATU KALI setelah hyperparameter dan threshold dibekukan di validation set.
* **Protokol Non-Destruktif**: Seluruh artefak baseline (v1) dipertahankan utuh dan perbaikan didokumentasikan transparan pada laporan v2.
