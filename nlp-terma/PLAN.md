# 01 — PLAN Project NLP: Project 2 TermA (Aspect dan Opinion Extraction, Ulasan Hotel)

> Baca dulu `00_INDEX_DAN_ATURAN_AGENT.md`. ID langkah: `NLP-T{tahap}-{nomor}`.

## 1. Ringkasan Tugas

- **Task**: sequence labeling BIO untuk span `ASPECT` dan `SENTIMENT` (ekspresi opini) pada ulasan hotel berbahasa Indonesia. Ini BUKAN klasifikasi sentimen seluruh ulasan.
- **Dataset**: IndoNLU, config `terma` (split resmi train/validation/test, jangan diubah).
- **Anotasi**: tidak membuat anotasi baru. Tugas Anda memvalidasi span dan mencatat contoh batas yang ambigu.
- **Dua model**: (1) satu CRF dengan fitur otomatis, (2) satu pretrained Transformer untuk token classification.
- **Evaluasi**: exact-match span Precision, Recall, F1 (keduanya memakai fungsi yang sama), plus per tipe label.
- **Luaran akhir**: kode modular, laporan Tahap I dan II, GUI (Streamlit/Gradio) dengan pilihan model dan highlight span.

Label: `O`, `B-ASPECT`, `I-ASPECT`, `B-SENTIMENT`, `I-SENTIMENT`. Verifikasi nama label persis dari dataset (jangan diasumsikan).

## 2. Gambaran Fase

```
Tahap I : T1 Setup -> T2 Load+Validasi -> T3 EDA -> T4 Desain fitur CRF -> T5 Desain Transformer -> T6 Desain eksperimen -> T7 Laporan I
Tahap II: T8 CRF -> T9 Transformer -> T10 Span+Evaluasi -> T11 Error analysis -> T12 Perbandingan -> T13 GUI -> T14 Laporan II
```

## 3. Struktur Repo

```
nlp-terma/
├── data/                  # cache/parquet, sampel anotasi tervalidasi
├── notebooks/eda.ipynb
├── src/
│   ├── config.py          # seed, path, model id, hyperparameter
│   ├── data_loader.py     # load split resmi, jaga ID data
│   ├── annotation.py      # validasi token-label BIO/IOB
│   ├── eda.py
│   ├── crf_features.py    # word_shape, token2features, sentence2features
│   ├── train_crf.py
│   ├── transformer_data.py# tokenisasi, alignment, mapping balik
│   ├── train_transformer.py
│   ├── spans.py           # normalisasi BIO + rekonstruksi span (dipakai kedua model)
│   └── evaluate.py        # span P/R/F1 + kategori error
├── gui/app.py
├── experiments/           # config + hasil tiap run (json/csv)
├── reports/
├── main.py
└── README.md
```

## 4. Tahap I — Data dan Desain

### NLP-T1 Setup
- [x] NLP-T1-01 Buat repo, venv, `requirements.txt` (datasets, transformers, torch, sklearn-crfsuite, seqeval, pandas, matplotlib, streamlit atau gradio). Catat versi library.
- [x] NLP-T1-02 `config.py` berisi `SEED=42`, path, `MODEL_NAME`, `MAX_LEN=128`.
- **DoD**: `python -c "import sklearn_crfsuite, transformers"` sukses.

### NLP-T2 Load dan Validasi Anotasi
- [x] NLP-T2-01 Muat config `terma`; pertahankan split resmi. Coba `load_dataset`, fallback ke parquet.
- [x] NLP-T2-02 Catat: jumlah kalimat per split, nama kolom token dan label, `label_list`, panjang token rata-rata.
- [x] NLP-T2-03 Cek otomatis: jumlah token = jumlah label, tidak ada label di luar `label_list`, tidak ada kalimat kosong, deteksi transisi ilegal (`O -> I-X`, `B-ASPECT -> I-SENTIMENT`). Laporkan jumlahnya.
- [x] NLP-T2-04 **Validasi manual minimal 20 contoh** (acak, seed tetap, sertakan span multi-token dan kalimat dengan beberapa span). Simpan ke `data/sampel_validasi.csv` dengan kolom: id, tokens, label, span ASPECT, span SENTIMENT, status (ok/ambigu), catatan.
- [x] NLP-T2-05 Catat contoh **batas ambigu** (mis. apakah "sangat" ikut dalam span SENTIMENT, apakah aspek "kamar mandi" satu span atau dua). Ini inti tugas validasi Anda.
- **DoD**: tabel validasi 20 contoh + daftar contoh ambigu siap masuk laporan. **Review manual Anda wajib** (agent tidak boleh menilai ambiguitas sendirian).

### NLP-T3 EDA Khusus Task
- [x] NLP-T3-01 Panjang ulasan (token) dan persentase terpotong pada max length 128 (subword, bukan kata).
- [x] NLP-T3-02 Frekuensi label BIO dan jumlah span per tipe; rasio `O` terhadap non-`O`.
- [x] NLP-T3-03 Jumlah span per ulasan; panjang span (1 token vs multi-token) per tipe.
- [x] NLP-T3-04 Frekuensi negasi dan intensifier (daftar kecil, didokumentasikan, dibuat dari training set saja).
- [x] NLP-T3-05 Contoh kalimat sulit: aspek implisit, opini tanpa aspek, span berdekatan.
- [x] NLP-T3-06 Tulis implikasi ke pemodelan (mis. imbalance `O`, risiko boundary error).
- **DoD**: grafik tersimpan + 1 paragraf implikasi per analisis.

### NLP-T4 Desain Fitur CRF
- [x] NLP-T4-01 Implementasikan `word_shape`, `token2features`, `sentence2features` dari kerangka dosen (fitur inti: leksikal, afiks, ortografis, word shape, konteks ±1, BOS/EOS).
- [x] NLP-T4-02 Jalankan pada minimal 2 kalimat, tampilkan dictionary fitur, verifikasi BOS di token pertama dan EOS di token terakhir.
- [x] NLP-T4-03 (Opsional, maks 2 kategori khusus) fitur negasi dan intensifier, misalnya `prev_is_negation`. Daftar kata hanya dari training set atau sumber eksternal yang dicatat.
- [x] NLP-T4-04 Tulis spesifikasi fitur (tabel: kelompok, contoh, alasan).
- **DoD**: dua contoh dictionary fitur otomatis siap dimasukkan laporan.

### NLP-T5 Desain Transformer
- [x] NLP-T5-01 Pilih satu model Indonesia ukuran base atau lebih kecil, mis. `indobenchmark/indobert-base-p1` (alternatif: `indolem/indobert-base-uncased`). Tulis alasan pemilihan. Verifikasi model ada dan tokenizer fast tersedia.
- [x] NLP-T5-02 Kebijakan alignment: first-subword berlabel, subword lanjutan dan token khusus `-100`. Demokan pada 3 kata yang terpecah.
- [x] NLP-T5-03 Rencana pemetaan prediksi kembali ke token asli (label subword pertama).

### NLP-T6 Desain Eksperimen
- [x] NLP-T6-01 CRF: ruang tuning kecil pada validation (`c1`, `c2`, `algorithm='lbfgs'`, `max_iterations`), contoh `c1, c2 in {0.01, 0.1, 1}` secukupnya.
- [x] NLP-T6-02 Transformer: lr 2e-5 (opsional 3e-5), batch 8 (turun ke 4 jika OOM), epoch 3, weight decay 0.01, max len 128, pilih checkpoint dari validation F1.
- [x] NLP-T6-03 Tulis aturan anti-leakage: test tidak dipakai untuk fitur, daftar kata, parameter, maupun checkpoint.

### NLP-T7 Laporan Tahap I
- [x] NLP-T7-01 Susun: deskripsi task, dataset, split, label, unit evaluasi, bukti validasi (20 contoh), EDA + implikasi, spesifikasi fitur + 2 dictionary, model + alignment, rancangan eksperimen.

## 5. Tahap II — Model dan Aplikasi

### NLP-T8 CRF
- [x] NLP-T8-01 `train_crf.py`: latih di train, tuning di validation, simpan model (`joblib`/`.crfsuite`) dan config.
- [x] NLP-T8-02 Prediksi validation dan test; simpan prediksi BIO ber-ID.
- **DoD**: F1 validation per konfigurasi tercatat di `experiments/crf_*.json`.

### NLP-T9 Transformer
- [x] NLP-T9-01 `transformer_data.py`: `tokenize_and_align_labels` dengan `word_ids()`, `is_split_into_words=True`.
- [ ] NLP-T9-02 `train_transformer.py`: `AutoModelForTokenClassification` + `Trainer`, `load_best_model_at_end` berdasarkan validation F1 (span-level via seqeval). Test TIDAK dimasukkan ke Trainer.
- [ ] NLP-T9-03 Simpan learning curve (loss dan F1 per epoch), checkpoint terbaik, tokenizer, label map, config ke Drive.
- [ ] NLP-T9-04 Prediksi test; petakan subword ke token asli.

### NLP-T10 Rekonstruksi Span dan Evaluasi
- [x] NLP-T10-01 `spans.py`: satu aturan normalisasi BIO (mis. `I-X` tanpa pendahulu sah diubah menjadi `B-X`), satu fungsi rekonstruksi span, dipakai kedua model.
- [x] NLP-T10-02 `evaluate.py`: exact-match span P/R/F1 keseluruhan dan per tipe (ASPECT, SENTIMENT); uji unit dengan contoh dosen (TP=2, FP=1, FN=1, F1=2/3).
- [x] NLP-T10-03 Laporkan hasil validation dan test terpisah jelas. Test dievaluasi satu kali untuk konfigurasi final.

### NLP-T11 Analisis Kesalahan (kedua model)
- [x] NLP-T11-01 Kategori: missed aspect, missed opinion, ASPECT-SENTIMENT confusion, false span, boundary error, negation/modifier error. Hitung jumlah tiap kategori.
- [x] NLP-T11-02 Ambil contoh representatif per kategori (minimal 3 per model), hubungkan dengan hasil EDA dan ambiguitas anotasi dari NLP-T2-05.

### NLP-T12 Perbandingan
- [ ] NLP-T12-01 Tabel CRF vs Transformer (P/R/F1 global dan per tipe).
- [ ] NLP-T12-02 Interpretasi: jangan berhenti di "Transformer lebih baik". Kaitkan dengan frekuensi label, kompleksitas batas span, fitur lokal vs konteks, tokenisasi/subword, dan ambiguitas anotasi.

### NLP-T13 GUI
- [x] NLP-T13-01 `gui/app.py`: input ulasan, pilihan model (CRF/Transformer), highlight ASPECT dan SENTIMENT, tabel hasil ekstraksi. Tanpa logika training; memakai model tersimpan dan `spans.py` yang sama.

### NLP-T14 Laporan Tahap II dan Reproducibility
- [x] NLP-T14-01 Catat: versi dataset, ukuran split, label map, seed, versi library, fitur dan parameter CRF, model id, lr, batch, epoch, max len, kebijakan alignment, kriteria checkpoint.
- [x] NLP-T14-02 Tulis keterbatasan dan pernyataan penggunaan AI/sumber eksternal.

## 6. Risiko dan Titik Kontrol

| Risiko | Pencegahan |
|---|---|
| Alignment label salah (label bergeser) | Uji pada 3 kalimat, cek jumlah label = jumlah word_ids non-None |
| Metrik CRF dan Transformer tidak sebanding | Pakai `spans.py` + `evaluate.py` yang sama |
| Leakage lewat daftar negasi/fitur | Bangun hanya dari train |
| Colab OOM | Batch 4 + `gradient_accumulation_steps=2`, max len 128 |
| Hasil hilang saat sesi putus | Simpan checkpoint ke Drive tiap epoch |
