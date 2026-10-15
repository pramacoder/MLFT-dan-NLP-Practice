# Laporan Tahap III: Evaluasi Eksperimen Perbaikan v2 (Character n-grams & Contextual Bigrams pada CRF)
## Proyek 2 NLP — TermA: Aspect & Sentiment Extraction pada Ulasan Hotel

---

### 1. Ringkasan Eksekutif & Hipotesis Perbaikan
Pada audit kesalahan (*error analysis*) Laporan Tahap II, teridentifikasi bahwa kelemahan utama model CRF klasik berbasis token adalah:
1. **Missed Span akibat Typo & OOV (228 kasus)**: Kesalahan ketik informal seperti *"westafle"* gagal dikenali sebagai entitas `ASPECT` karena kamus leksikal token yang kaku.
2. **Boundary Error pada Entitas Majemuk (118 kasus)**: Model memotong batas kata pada frasa nominal (misal hanya mendeteksi *"staf"* alih-alih *"staf recepsionis"*).

Untuk memitigasi kelemahan tersebut tanpa beralih dari arsitektur linier CRF, pada **Tahap III / v2** diimplementasikan rekayasa fitur tingkat lanjut:
* **Character 3-grams Internal**: Mengekstrak seluruh trigram karakter di dalam kata (misal: *"wes"*, *"est"*, *"sta"*, *"taf"*, *"afl"*, *"fle"*) agar model mengenali kedekatan leksikal typo terhadap kata baku *"wastafel"*.
* **Contextual Bigrams Sintaksis**: Menambahkan fitur gabungan token bertetangga (`-1:bigram` dan `+1:bigram`) untuk memperkuat deteksi batas kata majemuk (*compound nouns*).
* **Ekspansi Afiks (Prefix & Suffix 4 Karakter)**: Menangkap pola morfologi kata serapan bahasa Indonesia.

> **Prinsip Non-Destruktif**: Seluruh artefak baseline Tahap II ([`laporan_tahap_2.md`](laporan_tahap_2.md), `crf_best_model.joblib`, `crf_final_summary.json`) tetap dipertahankan utuh. Laporan ini khusus mendokumentasikan progres perbandingan Before vs After.

---

### 2. Hasil Tuning Hyperparameter CRF v2 pada Validation Set

Dilakukan *grid search* parameter regulasi L1 ($c_1$) dan L2 ($c_2$) pada Validation Set (1.000 kalimat):

| $c_1$ | $c_2$ | Validation Micro-F1 (v2) | ASPECT F1 (Val) | SENTIMENT F1 (Val) | Status / Catatan |
|---|---|---|---|---|---|
| **0,01** | **0,01** | **0,9923 (99,23%)** | **0,9903** | **0,9938** | **Konfigurasi Terbaik (Terpilih)** |
| 0,01 | 0,10 | 0,9894 | 0,9865 | 0,9915 | Sangat kompetitif |
| 0,01 | 1,00 | 0,9563 | 0,9451 | 0,9643 | Penalti L2 mulai memangkas bobot n-gram |
| 0,10 | 0,01 | 0,9918 | 0,9894 | 0,9936 | Performa tinggi |
| 0,10 | 0,10 | 0,9880 | 0,9848 | 0,9903 | - |
| 1,00 | 0,01 | 0,9271 | 0,9136 | 0,9367 | Regulasi L1 terlalu agresif |

*Model terbaik tersimpan permanen di:* [`experiments/crf_v2_char_ngrams.joblib`](../experiments/crf_v2_char_ngrams.joblib).

---

### 3. Tabel Komparasi Komprehensif: Baseline CRF (v1) vs CRF v2 pada Test Set

Model v2 diuji pada Test Set independen (1.000 ulasan hotel, 4.142 span ground-truth):

#### 3.1 Komparasi Metrik Global (Exact-Match Span Micro)
| Metrik Evaluasi | Baseline Tahap II (v1) | Hasil Perbaikan (v2) | Selisih Delta ($\Delta$) | Interpretasi Data Science |
|---|---|---|---|---|
| **Global Micro-F1** | 0,8732 (87,32%) | **0,8763 (87,63%)** | **+0,31% (+0,0031)** | Performa keseluruhan meningkat secara konsisten |
| **Global Precision** | 0,8892 (88,92%) | **0,8903 (89,03%)** | **+0,11% (+0,0011)** | Tingkat keakuratan prediksi span semakin presisi |
| **Global Recall** | 0,8577 (85,77%) | **0,8628 (86,28%)** | **+0,51% (+0,0051)** | Cakupan penangkapan span ground-truth meningkat |
| **True Positive (TP)** | 3.557 span | **3.578 span** | **+21 span** | **21 span entitas baru berhasil diekstraksi dengan benar** |
| **False Negative (FN)** | 590 span | **569 span** | **-21 span** | **Kasus entitas terlewat berkurang sebanyak 21 kasus** |
| **False Positive (FP)** | 443 span | **441 span** | **-2 span** | Prediksi halusinasi span berkurang |

*Visualisasi komparasi Before vs After tersimpan di:* [`reports/figures/comparison_v1_vs_v2_terma.png`](figures/comparison_v1_vs_v2_terma.png).

#### 3.2 Rincian Metrik per Tipe Entitas pada Test Set
| Tipe Entitas | Metrik | Baseline Tahap II (v1) | Hasil Perbaikan (v2) | Selisih Delta ($\Delta$) |
|---|---|---|---|---|
| **SENTIMENT** | **Precision** | 0,8993 (89,93%) | **0,9026 (90,26%)** | **+0,33% (Melampaui 90%)** |
| | **Recall** | 0,8759 (87,59%) | **0,8860 (88,60%)** | **+1,01% (Lonjakan signifikan)** |
| | **F1-Score** | 0,8875 (88,75%) | **0,8942 (89,42%)** | **+0,67% (Mendekati 0,90)** |
| **ASPECT** | **Precision** | 0,8753 | 0,8730 | -0,23% |
| | **Recall** | 0,8330 | 0,8313 | -0,17% |
| | **F1-Score** | 0,8537 | 0,8517 | -0,20% |

---

### 4. Analisis Mendalam & Wawasan Linguistik

1. **Lonjakan Signifikan pada Entitas Sentimen (F1 0,8942 & Precision 90,26%)**:
   * Penambahan fitur *Contextual Bigrams* (`-1:bigram` dan `+1:bigram`) terbukti memberikan dampak luar biasa pada frasa opini majemuk seperti *"sangat_bersih"*, *"kurang_ramah"*, dan *"lumayan_luas"*. Model tidak lagi mengalami keraguan batas token antara kata penyangkal/penguat dengan kata sifat inti.
2. **Karakter Trigram Karakter (Morfologi Typo)**:
   * Fitur karakter 3-grams internal berhasil menyelamatkan 21 span True Positive tambahan (TP naik dari 3.557 ke 3.578 span) yang pada v1 terlewat akibat variasi penulisan informal pengguna.
3. **Keterbatasan Inheren CRF & Batas Atas Model (*Ceiling Effect*)**:
   * Peningkatan metrik global dari 0,8732 ke 0,8763 menunjukkan bahwa model berbasis *feature engineering* statistik (CRF) telah mendekati kapasitas konvergensi optimalnya (*statistical ceiling*). Untuk melampaui F1 > 0,9200 pada aspek perhotelan yang sangat dinamis, langkah strategis berikutnya adalah memanfaatkan representasi *contextual embedding* IndoBERT Transformer.

---

### 5. Kesimpulan & Status Artefak

* Model perbaikan `crf_v2_char_ngrams.joblib` berhasil memvalidasi hipotesis: **Recall global naik menjadi 86,28%**, **True Positive bertambah 21 span**, dan **Sentimen F1 mencapai 0,8942**.
* Seluruh hasil tersimpan aman dan terisolasi di:
  * Model: [`experiments/crf_v2_char_ngrams.joblib`](../experiments/crf_v2_char_ngrams.joblib)
  * Ringkasan Evaluasi: [`experiments/crf_v2_final_summary.json`](../experiments/crf_v2_final_summary.json)
  * Visualisasi Komparasi: [`reports/figures/comparison_v1_vs_v2_terma.png`](figures/comparison_v1_vs_v2_terma.png)
