# 02 — PLAN Machine Learning for Text: P3 Jigsaw Toxicity (Multi-label)

> Baca dulu `00_INDEX_DAN_ATURAN_AGENT.md`. ID langkah: `ML-T{tahap}-{nomor}`.

## 1. Ringkasan Tugas

- **Input X**: satu komentar Wikipedia berbahasa Inggris.
- **Target Y**: vektor biner 6 dimensi: `toxic, severe_toxic, obscene, threat, insult, identity_hate` (multi-hot, boleh 0, 1, atau banyak label aktif).
- **Paradigma**: supervised learning, multi-label binary classification.
- **Model**: satu pretrained English Transformer + 6 logits, **sigmoid** saat inferensi, **BCEWithLogitsLoss** saat training (bukan softmax).
- **Fokus metodologi**: imbalance berat (dampak diukur dan ditafsirkan), pemilihan threshold di validation, analisis error dan bias kontekstual.
- **Bukan tujuan**: membandingkan banyak model, hyperparameter search besar, deployment production.

## 2. Gambaran Fase (12 langkah metodologi dosen)

```
Tahap I : Formulasi -> Validasi data -> EDA -> Persiapan data/split -> Desain eksperimen -> Desain arsitektur/loss/metrik -> Kerangka modular + GUI
Tahap II: Fine-tuning -> Seleksi di validation -> Threshold -> Test (sekali) -> Error/bias analysis -> GUI
```

## 3. Struktur Repo

```
mlft-jigsaw/
├── data/
├── notebooks/eda.ipynb
├── src/
│   ├── config.py          # seed, model, batch, lr, max_len, epoch, path
│   ├── data_loader.py     # muat data, cek schema, buat val split
│   ├── preprocessing.py   # tokenizer, truncation/padding, target multi-hot
│   ├── model.py           # model + head 6 output
│   ├── train.py           # Trainer/loop, optimizer, checkpoint, log
│   ├── evaluation.py      # metrik, per-label, threshold sweep, hamming loss
│   └── inference.py       # muat model final + threshold, prediksi konsisten
├── gui/app.py
├── experiments/           # config + hasil tiap run
├── main.py
└── README.md
```

## 4. Tahap I — Analisis, EDA, Desain Eksperimen

### ML-T1 Formulasi Masalah
- [ ] ML-T1-01 Tulis tabel: Input X, Target Y, unit prediksi (per komentar), paradigma, tipe masalah, output inferensi (probabilitas 6 label + label aktif).
- [ ] ML-T1-02 Jelaskan mengapa sigmoid + BCE (label tidak saling eksklusif).

### ML-T2 Pemahaman dan Validasi Data
- [ ] ML-T2-01 Muat dataset (lihat link di `00_INDEX`). Identifikasi kolom teks (`comment_text`), 6 kolom label, tipe data, jumlah baris per split.
- [ ] ML-T2-02 Cek: teks kosong/missing, duplikasi, nilai label di luar {0,1}.
- [ ] ML-T2-03 **Cek label test**: jika ada nilai `-1`, buang baris tersebut dari evaluasi test dan laporkan jumlahnya.
- [ ] ML-T2-04 Catat apakah validation resmi tersedia (tidak; harus dibuat sendiri dari train).

### ML-T3 EDA Berorientasi Keputusan ML
- [ ] ML-T3-01 Frekuensi dan proporsi 6 label; proporsi komentar tanpa label vs ≥1 label.
- [ ] ML-T3-02 Label cardinality (rata-rata label aktif per komentar).
- [ ] ML-T3-03 Matriks/heatmap co-occurrence 6 label.
- [ ] ML-T3-04 Identifikasi label langka (severe_toxic, threat, identity_hate biasanya paling jarang) dan hipotesis dampaknya (recall rendah, Macro-F1 jauh di bawah Micro-F1).
- [ ] ML-T3-05 Panjang komentar (token subword) dan persentase terpotong pada 128/256, untuk memilih max length.
- [ ] ML-T3-06 Contoh **non-ofensif** yang menjelaskan potensi false positive berbasis kata tanpa konteks. Jangan menampilkan konten ofensif yang tidak perlu.
- **DoD**: tiap analisis punya kesimpulan "keputusan ML yang dipengaruhi".

### ML-T4 Split dan Anti-Leakage
- [ ] ML-T4-01 Bentuk validation dari train (mis. 90/10) dengan seed 42. Gunakan **stratifikasi multi-label** (paket `iterative-stratification`, `MultilabelStratifiedShuffleSplit`) agar label langka ada di validation. Laporkan distribusi label train vs validation.
- [ ] ML-T4-02 Simpan indeks split ke file (`data/split_ids.json`) agar reproducible.
- [ ] ML-T4-03 Tulis aturan: test tidak untuk tuning, threshold, atau preprocessing.
- [ ] ML-T4-04 Jika subsampling karena Colab: catat metode, seed, ukuran, dan distribusi label.

### ML-T5 Desain Model, Loss, Metrik
- [ ] ML-T5-01 Pilih satu model, rekomendasi `roberta-base` (alternatif ringan: `distilroberta-base`). Tulis alasan. Gunakan `AutoModelForSequenceClassification` dengan `problem_type="multi_label_classification"`, `num_labels=6`.
- [ ] ML-T5-02 Jelaskan alur input: raw text, tokenizer, subword, token ID, attention mask, contextual representation.
- [ ] ML-T5-03 Preprocessing: **tidak** ada stemming, stop-word removal, atau pembuangan tanda baca (merusak informasi pretraining). Tangani hanya hal yang terdokumentasi (mis. komentar kosong).
- [ ] ML-T5-04 Rencana hyperparameter kecil: lr {2e-5, 3e-5, 5e-5}, batch {16, 32}, epoch {3, 5}, max_len dari EDA, weight decay 0.01. Eksperimen bertahap, tiap perubahan dicatat alasannya.
- [ ] ML-T5-05 Metrik utama ditetapkan SEBELUM melihat test: **Macro-F1 validation** (dipakai juga untuk memilih threshold). Metrik lain: Micro/Macro P/R/F1, per-label F1, Hamming Loss.
- [ ] ML-T5-06 (Opsional) class weighting (`pos_weight`) sebagai SATU eksperimen terkontrol, diputuskan hanya dari train/validation.

### ML-T6 Laporan Tahap I dan Skeleton
- [ ] ML-T6-01 Buat skeleton modular (file kosong + docstring tanggung jawab) dan rancangan GUI.
- [ ] ML-T6-02 Susun laporan Tahap I dari ML-T1 sampai ML-T5.

## 5. Tahap II — Implementasi, Evaluasi, Interpretasi

### ML-T7 Pipeline dan Fine-tuning
- [x] ML-T7-01 `preprocessing.py`: tokenisasi dinamis, target multi-hot bertipe float32.
- [x] ML-T7-02 `train.py`: fine-tuning dengan BCEWithLogits, AdamW, fp16, evaluasi per epoch di validation; log train loss, val loss, val Macro-F1 (threshold 0,5) ke CSV.
- [x] ML-T7-03 Simpan checkpoint + config per run ke Drive.

### ML-T8 Eksperimen Hyperparameter Terkontrol
- [x] ML-T8-01 Jalankan ruang kecil sesuai ML-T5-04 (tidak semua kombinasi wajib).
- [x] ML-T8-02 Plot learning curve per run; diagnosis: normal / overfitting / underfitting.
- [x] ML-T8-03 Pilih konfigurasi dan checkpoint terbaik dari validation Macro-F1.

### ML-T9 Threshold di Validation
- [x] ML-T9-01 Hitung probabilitas validation dari checkpoint terpilih.
- [x] ML-T9-02 Bandingkan global threshold {0.3, 0.4, 0.5, 0.6, 0.7} (ditetapkan sebelumnya) dengan baseline 0,5; pilih satu berdasarkan validation Macro-F1.
- [x] ML-T9-03 **Bekukan** seluruh keputusan (model, hyperparameter, checkpoint, threshold) ke `experiments/final_config.json`. Setelah ini tidak ada perubahan.

### ML-T10 Evaluasi Test (SEKALI)
- [x] ML-T10-01 Hitung probabilitas test, terapkan threshold final, laporkan: Micro dan Macro P/R/F1, per-label P/R/F1, Hamming Loss.
- [x] ML-T10-02 Jelaskan Micro vs Macro dan kaitannya dengan imbalance; bandingkan per-label F1 dengan frekuensi label dari EDA.

### ML-T11 Error dan Bias Analysis
- [x] ML-T11-01 False negative pada label langka (recall rendah?).
- [x] ML-T11-02 False positive (kata tanpa konteks).
- [x] ML-T11-03 Co-occurrence error (hanya sebagian label benar terdeteksi).
- [x] ML-T11-04 Kasus probabilitas dekat threshold (mis. 0,4 sampai 0,6) yang berubah keputusan.
- [x] ML-T11-05 Bias kontekstual: komentar yang menyebut identitas secara netral tetapi diprediksi toksik (gunakan contoh non-ofensif atau disensor).
- [x] ML-T11-06 Tulis keterbatasan.

### ML-T12 GUI
- [x] ML-T12-01 `gui/app.py` (Streamlit/Gradio): input komentar, tombol Analyze, 6 probabilitas, threshold final, label prediksi, dan peringatan bahwa keluaran adalah prediksi model yang bisa salah. Memakai `inference.py` yang sama dengan eksperimen.

### ML-T13 Laporan Tahap II dan Artefak
- [x] ML-T13-01 Laporan: training, eksperimen hyperparameter, learning curve, seleksi validation, test final, error analysis, keterbatasan.
- [x] ML-T13-02 Saved configuration (model, seed, lr, batch, epoch, max len, threshold) + README.

## 6. Risiko dan Titik Kontrol

| Risiko | Pencegahan |
|---|---|
| Label test `-1` ikut terhitung | ML-T2-03, filter sebelum evaluasi |
| Label langka hilang dari validation | Stratifikasi multi-label (ML-T4-01) |
| Threshold dipilih setelah lihat test | Bekukan di ML-T9-03 |
| Akurasi tampak tinggi tapi menipu | Laporkan Macro-F1, per-label, bukan akurasi |
| Training lama di Colab | distilroberta, max_len 128, fp16, subsampling terdokumentasi |
| Konten ofensif di laporan | Gunakan contoh non-ofensif atau sensor |
