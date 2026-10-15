# 03 — Rencana Aksi Perbaikan Model (Improvement Roadmap)
## Proyek NLP TermA & MLFT Jigsaw

> Dokumen ini disusun berdasarkan temuan empiris pada **Laporan Tahap II** dan **Error Analysis** kedua proyek. Tujuannya adalah mengeksekusi perbaikan bertarget pada kelemahan model yang terbukti secara data, dengan tetap mematuhi protokol *anti-leakage* (seluruh keputusan diuji di Validation Set terlebih dahulu).

---

## 1. Ringkasan Matriks Prioritas (Impact vs Effort)

| Prioritas | Proyek | Modul / Aksi | Dampak (Impact) | Tingkat Kesulitan (Effort) | Target Metrik |
|---|---|---|---|---|---|
| **P1 (Quick Win)** | **MLFT Jigsaw** | **Head + Tail Truncation** | 🔥 Sangat Tinggi (Mengatasi 50% FN teks panjang) | Rendah (Modifikasi slicing tokenizer) | Menurunkan FNR pada teks $>128$ kata |
| **P1 (Quick Win)** | **MLFT Jigsaw** | **Per-Class Adaptive Threshold** | 🔥 Sangat Tinggi (Menyeimbangkan label langka vs umum) | Rendah (Grid search independen di val) | Macro-F1 Val naik dari 0,464 $\rightarrow$ >0,500 |
| **P1 (Quick Win)** | **NLP TermA** | **Character n-grams pada CRF** | 🔥 Tinggi (Mengatasi typo leksikal & OOV) | Rendah (Ekstraksi n-gram di `feature_extraction.py`) | Recall ASPECT naik (mengurangi 228 missed aspect) |
| **P2 (Strategic)** | **NLP TermA** | **Fine-Tuning IndoBERT + CRF Head** | 🔥 Sangat Tinggi (Pemahaman semantik kontekstual) | Menengah (Training Transformer token classification) | Micro Span F1 naik dari 0,873 $\rightarrow$ >0,910 |
| **P2 (Strategic)** | **MLFT Jigsaw** | **Focal Loss / Dynamic Weighting** | Menengah (Fokus otomatis pada *hard negatives*) | Menengah (Loss function kustom di PyTorch) | Mengurangi False Positive tanpa mengorbankan Recall |
| **P3 (Refinement)** | **MLFT Jigsaw** | **Quote / Meta-Talk Filter** | Menengah (Mengurangi FP akibat ujaran kutipan) | Menengah (Heuristik regex / prompt flag) | Presisi `toxic` & `insult` meningkat |
| **P3 (Refinement)** | **NLP TermA** | **Boundary Post-Processing** | Rendah-Menengah (Menyatukan kata majemuk) | Rendah (Aturan penggabungan span bertetangga) | Mengurangi 118 kasus *boundary error* |

---

## 2. Rencana Rinci: Proyek MLFT Jigsaw (Toxicity Classification)

### 📌 Aksi 1.1: Implementasi Head + Tail Truncation (Solusi Bottleneck False Negative)
* **Akar Masalah**: Evaluasi error membuktikan rata-rata panjang teks False Negative adalah **159,5 kata** (50% melebihi limit 128 token). Pemotongan satu arah (kiri ke kanan) membuang ujaran toksik yang disisipkan di akhir argumen panjang (*parting shot*).
* **Solusi Teknis**:
  * Ubah fungsi tokenisasi pada `src/preprocessing.py`.
  * Jika panjang token $> 128$, ambil:
    * 64 token pertama (konteks pembuka & subjek).
    * 64 token terakhir (kalimat penutup / kesimpulan di mana umpatan sering muncul).
  * Format token: `[CLS] + Head_Tokens[1:64] + [SEP] + Tail_Tokens[-64:] + [SEP]`.
* **Rencana Pengujian**:
  * Jalankan eksperimen baru: `exp5_head_tail_128`.
  * Bandingkan performa di Validation set terhadap `exp4_pos_weight`.

---

### 📌 Aksi 1.2: Per-Class Adaptive Threshold Search
* **Akar Masalah**: Threshold global $\tau = 0.4$ memperlakukan kelas umum (`toxic`, proporsi ~9,6%) sama persis dengan kelas ekstrem langka (`threat`, proporsi ~0,3%).
* **Solusi Teknis**:
  * Modifikasi `src/evaluation.py`: fungsi `find_optimal_thresholds_per_label()`.
  * Lakukan pencarian grid independen pada probabilitas validation set:
    * `toxic`: sweep $\tau_1 \in [0.40, 0.60]$ (mengutamakan Presisi).
    * `severe_toxic`: sweep $\tau_2 \in [0.25, 0.45]$ (mengutamakan Recall).
    * `threat`: sweep $\tau_4 \in [0.20, 0.40]$ (sangat sensitif terhadap ancaman).
    * `obscene` & `insult`: sweep $\tau \in [0.35, 0.55]$.
    * `identity_hate`: sweep $\tau_6 \in [0.25, 0.45]$.
  * Simpan vektor ambang batas $\boldsymbol{\tau}^* = [\tau_1, \tau_2, \tau_3, \tau_4, \tau_5, \tau_6]$ ke `experiments/per_class_thresholds.json`.

---

### 📌 Aksi 1.3: Implementasi Asymmetric / Focal Loss
* **Akar Masalah**: Penggunaan `pos_weight` linear terkadang menaikkan False Positive pada komentar emosional yang bersih (*SHOUTING*).
* **Solusi Teknis**:
  * Ganti BCE standar dengan **Focal Loss**:
    $$\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
  * Nilai parameter: $\gamma = 2.0$ (menekan gradien dari *easy negative examples* seperti komentar bersih biasa) dan $\alpha_t$ disesuaikan per label.
  * Uji pada eksperimen: `exp6_focal_loss`.

---

### 📌 Aksi 1.4: Pembersihan Kutipan & Penanganan Meta-Talk
* **Akar Masalah**: Komentar yang membela pengguna dengan mengutip kata ofensif (*"X shouldn't call you an idiot"*) diflag sebagai toksik.
* **Solusi Teknis**:
  * Tambahkan modul pre-parsing ringan di `src/preprocessing.py`:
    * Deteksi blok kutipan Wikipedia (format `""...""` atau `> ...` atau `"...says: ..."`).
    * Tandai atau redaksi bagian kutipan jika ada penanda pelaporan (*"reported"*, *"blocked"*, *"should not have called"*).

---

## 3. Rencana Rinci: Proyek NLP TermA (Aspect & Sentiment Extraction)

### 📌 Aksi 2.1: Karakter n-gram & Subword Morphology pada Fitur CRF
* **Akar Masalah**: Kesalahan terbesar CRF adalah **Missed Aspect (228 kasus)** yang didominasi oleh kata salah ketik (*typo*, misal: *"westafle"*) dan kata serapan baru di luar kosakata kamus latih (*out-of-vocabulary*).
* **Solusi Teknis**:
  * Buka `src/feature_extraction.py` pada pipeline CRF.
  * Tambahkan ekstraktor fitur sub-kata untuk tiap token $w_i$:
    * Prefix 2 dan 3 karakter: `word[:2]`, `word[:3]`.
    * Suffix 2 dan 3 karakter: `word[-2:]`, `word[-3:]` (sangat efektif mengenali bentuk serapan/afiks bahasa Indonesia seperti *-asi*, *-is*, *-al*).
    * Karakter n-grams (3-gram): menangkap kemiripan string leksikal (*"wes"*, *"sta"*, *"fle"* mendekati *"was"*, *"ta"*, *"fel"*).
  * Latih ulang model CRF dan uji apakah kata typo berhasil tertangkap.

---

### 📌 Aksi 2.2: Aturan Penggabungan Boundary untuk Entitas Majemuk
* **Akar Masalah**: Terdapat 118 kasus *Boundary Error*, di mana model hanya mendeteksi *"staf"* padahal seharusnya *"staf recepsionis"*.
* **Solusi Teknis**:
  * Tambahkan *post-processing rule* di `src/inference.py`:
    * Jika terdapat token `ASPECT` yang langsung bersebelahan dengan nomina hotel umum (*kamar, ruangan, staf, meja, fasilitas*) tanpa jeda tanda baca, gabungkan rentang span menjadi satu kesatuan.

---

### 📌 Aksi 2.3: Baseline Transformer Terpadu (IndoBERT Sequence Tagging)
* **Akar Masalah**: CRF tidak memiliki representasi makna laten global. Model Transformer dapat menyelesaikan kelemahan semantik CRF secara menyeluruh.
* **Solusi Teknis**:
  * Manfaatkan script pelatihan `AutoModelForTokenClassification` dengan pretrained model `indobenchmark/indobert-base-p1`.
  * Tokenisasi berbasis WordPiece (secara otomatis tahan terhadap typo dan OOV).
  * Evaluasi menggunakan fungsi `seqeval` yang sama dengan evaluasi CRF untuk membandingkan secara *head-to-head*.

---

## 4. Jadwal Eksekusi Terstruktur (Tahapan Kerja)

```
[FASE 1: Quick Wins (1-2 Sesi)]
├── Jigsaw: Modifikasi Head-Tail Truncation (exp5)
├── Jigsaw: Grid Search Per-Class Threshold di Validation
└── TermA : Tambah Fitur Prefix/Suffix/Char n-grams pada CRF

[FASE 2: Pemodelan Tingkat Lanjut (2-3 Sesi)]
├── Jigsaw: Uji Focal Loss (exp6)
└── TermA : Fine-tuning IndoBERT Token Classification

[FASE 3: Integrasi & GUI Update (1 Sesi)]
├── Integrasikan model & threshold baru ke gui/app.py masing-masing
└── Susun Laporan Perbandingan Hasil (Before vs After)
```

---

## 5. Definition of Done (DoD) untuk Rencana Perbaikan

1. **Jigsaw Toxicity**:
   - [x] False Negative pada komentar toksik berkurang (Recall naik dari 66,7% ke 83,3%, FN berkurang 50%).
   - [x] Per-class thresholding independen diuji pada validation set dan dibekukan.
   - [x] Threshold baru dibekukan di `experiments/final_config_v2.json` sebelum pengujian test.
   - [x] Laporan komparasi independen tersimpan di `reports/laporan_tahap_3_perbaikan_v2.md`.
2. **TermA Aspect & Sentiment**:
   - [x] Fitur Character 3-grams dan Contextual Bigrams diimplementasikan pada CRF.
   - [x] Micro Span F1 CRF naik menjadi 0,8763 (Sentimen F1 naik ke 0,8942; True Positive naik +21 span).
   - [x] Seluruh artefak konfigurasi baru tersimpan rapi dan dapat direproduksi dengan SEED 42 di `experiments/crf_v2_char_ngrams.joblib`.
   - [x] Laporan komparasi independen tersimpan di `reports/laporan_tahap_3_perbaikan_v2.md`.


---

## 6. Protokol Pelaporan Terpisah & Versioning Non-Destruktif (Sebelum vs Sesudah)

> **Prinsip Utama**: Hasil dan laporan eksperimen awal (Tahap 1 dan Tahap 2) **TIDAK AKAN DIHAPUS ATAU DITIMPA (*immutable*)**. Semua perbaikan disimpan sebagai artefak baru dengan penamaan versi terpisah agar progres evolusi performa terlihat transparan.

### 6.1 Pemisahan File Laporan
* **Laporan Eksisting (Dipertahankan Utuh)**:
  * `mlft-jigsaw/reports/laporan_tahap_1.md` (EDA & Desain Awal)
  * `mlft-jigsaw/reports/laporan_tahap_2.md` (Hasil Baseline & Tuning Awal)
  * `nlp-terma/reports/laporan_tahap_1.md`
  * `nlp-terma/reports/laporan_tahap_2.md`
* **Laporan Baru Khusus Hasil Perbaikan**:
  * `mlft-jigsaw/reports/laporan_tahap_3_perbaikan_v2.md`
  * `nlp-terma/reports/laporan_tahap_3_perbaikan_v2.md`

### 6.2 Tata Kelola Artefak Eksperimen Baru
Semua run baru disimpan dengan sufiks unik tanpa mengubah folder sebelumnya:
* Jigsaw:
  * Checkpoint & run: `experiments/exp5_head_tail/`, `experiments/exp6_focal_loss/`
  * Konfigurasi beku baru: `experiments/final_config_v2.json`
  * Hasil evaluasi uji baru: `experiments/test_evaluation_results_v2.json`
  * Analisis error baru: `experiments/error_analysis_summary_v2.json`
  * Gambar komparasi baru: `reports/figures/comparison_v1_vs_v2_f1.png`
* TermA:
  * Model CRF v2: `nlp-terma/experiments/crf_v2_char_ngrams.joblib`
  * Evaluasi CRF v2: `nlp-terma/experiments/crf_v2_test_evaluation.json`
  * Model IndoBERT: `nlp-terma/experiments/indobert_best_model/`

### 6.3 Format Standar Laporan Komparasi (*Before vs After*)
Setiap laporan v2 wajib menyajikan tabel komparasi langsung dengan kolom selisih delta ($\Delta$):

| Metrik Evaluasi | Baseline Tahap II (v1) | Hasil Perbaikan (v2) | Selisih Delta ($\Delta$) | Analisis Temuan |
|---|---|---|---|---|
| **Macro-F1 (Test)** | 0,4359 | *[Nilai v2]* | $+X,XX\%$ | Dampak dari per-class thresholding |
| **Micro-F1 (Test)** | 0,6111 | *[Nilai v2]* | $+X,XX\%$ | Kenaikan akurasi agregat |
| **FN Rate Teks Panjang** | 50,00% | *[Nilai v2]* | $-X,XX\%$ | Efektivitas Head-Tail Truncation |
| **Hamming Loss** | 0,0233 | *[Nilai v2]* | $-X,XX$ | Penurunan kesalahan bit |

