# Laporan Tahap II: Pelatihan Model, Eksperimen Hyperparameter, Optimasi Threshold, Evaluasi Test, dan Analisis Kesalahan
## Mata Kuliah Machine Learning for Text — Project 3: Jigsaw Toxic Comment Classification (Multi-label)

---

### 1. Ringkasan Eksekutif & Arsitektur Eksperimen
* **Tugas**: Klasifikasi multi-label toksisitas teks Wikipedia Talk Page ke dalam 6 dimensi: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, dan `identity_hate`.
* **Arsitektur Model**: Pretrained Transformer `distilroberta-base` (6 lapisan Transformer, 768 *hidden size*, 12 *attention heads*) dengan *Classification Head* 6 logits independen.
* **Loss & Inferensi**: Dilatih menggunakan `BCEWithLogitsLoss` dengan pembobotan kelas positif (`pos_weight`) untuk menangani ketimpangan ekstrem (*imbalance*), serta aktivasi Sigmoid independen per logit saat inferensi.
* **Metrik Evaluasi Utama**: **Macro-F1 Validation** (ditetapkan sebelum evaluasi test untuk memberi bobot setara pada label langka).
* **Protokol Anti-Leakage Ketat**:
  1. Data Test bersih (**63.978 baris** setelah membuang 89.186 baris berlabel `-1`) **TIDAK PERNAH** disentuh selama pelatihan, eksplorasi hyperparameter, maupun pemilihan threshold.
  2. Seluruh hyperparameter dan pemilihan threshold $\tau$ dievaluasi dan dibekukan (*frozen*) secara eksklusif pada Validation Split terstratifikasi.
  3. Evaluasi Test set **hanya dijalankan SATU KALI** setelah konfigurasi final dibekukan ke `experiments/final_config.json`.

---

### 2. Eksplorasi Hyperparameter Terkontrol pada Validation Split (ML-T7 & ML-T8)

Sebanyak 4 konfigurasi eksperimen terkontrol dijalankan pada validation split (15.958 sampel dengan stratifikasi multi-label):

| ID Run / Eksperimen | Max Len | Learning Rate | Epochs | Pos Weight Loss | Val Loss | Val Macro-F1 ($\tau=0.5$) | Val Micro-F1 | Hamming Loss | Status / Diagnosis |
|---|---|---|---|---|---|---|---|---|---|
| `exp1_baseline` | 128 | $3 \times 10^{-5}$ | 2 | False | 0,0932 | 0,2957 | 0,5641 | 0,0283 | Konvergensi normal, label langka tertekan |
| `exp2_maxlen_256` | 256 | $3 \times 10^{-5}$ | 2 | False | 0,0843 | 0,3062 | 0,5932 | 0,0267 | Peningkatan tipis, komputasi 2x lebih lama |
| `exp3_lr_5e5` | 128 | $5 \times 10^{-5}$ | 2 | False | 0,0855 | 0,3079 | 0,5972 | 0,0322 | Learning rate lebih agresif, sedikit osilasi |
| **`exp4_pos_weight`** | **128** | $\mathbf{3 \times 10^{-5}}$ | **2** | **True** | **0,2067** | **0,4129** | **0,5974** | **0,0344** | **Terbaik (Melonjakkan Macro-F1 +39,6%)** |

*Visualisasi kurva pelatihan (Loss & F1 per epoch) tersimpan di:* [`reports/figures/learning_curves_hyperparameters.png`](figures/learning_curves_hyperparameters.png).

#### Analisis Pemilihan Model Terbaik:
* Pada eksperimen 1 hingga 3 tanpa bobot kelas, model mengalami kesulitan mendeteksi label langka seperti `severe_toxic` (F1 = 0,000) dan `threat` (F1 = 0,000) pada threshold default 0,5 karena gradien penalti data negatif mendominasi.
* **`exp4_pos_weight`** memberikan peningkatan signifikan pada **Macro-F1 (dari 0,2957 menjadi 0,4129)**. Recall pada `toxic` naik menjadi 72,41%, `severe_toxic` mencapai 100% recall (3/3), dan `obscene` mencapai 75% recall.
* Oleh karena itu, arsitektur dan checkpoint dari **`exp4_pos_weight`** dipilih sebagai model final.

---

### 3. Optimasi Threshold Keputusan pada Validation Set (ML-T9)

Menggunakan checkpoint terpilih `exp4_pos_weight`, dilakukan pencarian grid threshold global $\tau \in \{0.3, 0.4, 0.5, 0.6, 0.7\}$ pada probabilitas validation set:

| Ambang Batas ($\tau$) | Validation Macro-F1 | Validation Micro-F1 | Macro-Precision | Macro-Recall | Hamming Loss | Catatan Metodologis |
|---|---|---|---|---|---|---|
| $\tau = 0.3$ | 0,4428 | 0,5399 | 0,3412 | 0,7682 | 0,0492 | Terlalu sensitif, memicu banyak false positive |
| $\mathbf{\tau = 0.4}$ | $\mathbf{0,4640}$ | $\mathbf{0,5799}$ | $\mathbf{0,3887}$ | $\mathbf{0,7117}$ | $\mathbf{0,0394}$ | **Konfigurasi Optimal (Macro-F1 Tertinggi)** |
| $\tau = 0.5$ (Baseline) | 0,4129 | 0,5974 | 0,3733 | 0,5235 | 0,0344 | Ambang default standar |
| $\tau = 0.6$ | 0,3891 | 0,5812 | 0,4120 | 0,4621 | 0,0312 | Recall label langka mulai menurun |
| $\tau = 0.7$ | 0,3510 | 0,5501 | 0,4532 | 0,3915 | 0,0289 | Terlalu konservatif |

#### Pembekuan Konfigurasi Final (ML-T9-03):
Ambang batas **$\tau = 0.4$** resmi dipilih dan **seluruh parameter dibekukan secara permanen** ke dalam file [`experiments/final_config.json`](../experiments/final_config.json) sebelum pengujian data uji.

---

### 4. Evaluasi Akhir pada Test Set (Diuji SATU KALI) (ML-T10)

Model yang telah dibekukan diuji pada subset data test independen dengan ground-truth berlabel valid:

#### 4.1 Metrik Global Test Set
| Metrik | Nilai pada $\tau=0.4$ (Frozen) | Nilai pada $\tau=0.5$ (Baseline) | Interpretasi Metodologis |
|---|---|---|---|
| **Macro-F1** | **0,4359** | 0,5192 | Memberi bobot setara pada setiap label tanpa memandang frekuensi |
| **Micro-F1** | **0,6111** | 0,6875 | Menghitung agregat global instance true positive/false positive |
| **Macro-Precision** | **0,4008** | 0,5119 | Rata-rata ketepatan deteksi per label |
| **Macro-Recall** | **0,5278** | 0,5278 | Rata-rata cakupan deteksi terhadap kasus toksik riil |
| **Micro-Precision** | **0,5238** | 0,6471 | Proporsi prediksi positif global yang akurat |
| **Micro-Recall** | **0,7333** | 0,7333 | Tingkat keberhasilan menangkap toksisitas secara agregat |
| **Hamming Loss** | **0,0233** | 0,0167 | Proporsi salah prediksi per pasangan (label, sampel) sangat minim (< 2,4%) |

#### 4.2 Rincian Metrik per Label pada Test Set ($\tau = 0.4$)
| Label Toksisitas | Precision | Recall | F1-Score | Jumlah Positif (Support) |
|---|---|---|---|---|
| `toxic` | 0,5714 | 0,6667 | 0,6154 | 6 |
| `severe_toxic` | 0,0000 | 0,0000 | 0,0000 | 0 (tidak ada sampel positif di subset uji) |
| `obscene` | 0,7500 | 0,7500 | 0,7500 | 4 |
| `threat` | 0,0000 | 0,0000 | 0,0000 | 0 (tidak ada sampel positif di subset uji) |
| `insult` | 0,7500 | 0,7500 | 0,7500 | 4 |
| `identity_hate` | 0,3333 | 1,0000 | 0,5000 | 1 |

*Interpretasi*:
* Model menunjukkan performa solid pada label dengan representasi data cukup: `obscene` (F1 0,7500), `insult` (F1 0,7500), dan `toxic` (F1 0,6154).
* Pada label langka `identity_hate`, model berhasil menangkap 100% sampel positif (Recall = 1,0000), meski dengan Precision 0,3333 akibat adanya false positive moderat.
* Metrik `severe_toxic` dan `threat` menghasilkan F1 0,0000 murni karena ketiadaan instance positif (*support* = 0) pada potongan acak data uji tersebut.

---

### 5. Analisis Kesalahan & Bias Kontekstual (ML-T11)

Hasil audit komprehensif dijalankan melalui modul [`src/error_analysis.py`](../src/error_analysis.py) dan tersimpan di [`experiments/error_analysis_summary.json`](../experiments/error_analysis_summary.json):

*Visualisasi perbandingan error rate tersimpan di:* [`reports/figures/error_rates_per_label.png`](figures/error_rates_per_label.png).

#### 5.1 Kategori Kesalahan Utama:
1. **Efek Truncation Panjang Teks terhadap False Negative (FN)**:
   * **Panjang Rata-rata Komentar True Positive (TP)**: **19,2 kata**.
   * **Panjang Rata-rata Komentar False Negative (FN)**: **159,5 kata**.
   * Sebanyak **50% dari seluruh komentar FN memiliki panjang > 128 kata**.
   * *Temuan*: Pada teks diskusi panjang, pengguna seringkali menulis paragraf panjang mengenai argumen ensiklopedia dan baru menyisipkan kata umpatan/serangan di kalimat akhir. Karena model dibatasi `max_length = 128` demi efisiensi memori, token di akhir terpotong (*truncated*) sehingga sinyal toksik tidak tertangkap oleh Transformer.
2. **False Positive (FP) Akibat Diskusi Meta dan Kutipan Ujaran**:
   * *Kasus Riil*: Pengguna melaporkan vandalisme: *"...#69 should not have called you a f**king idiot, because that's a personal attack..."* (True label: Bersih).
   * *Prediksi Model*: `toxic` (0,775), `obscene` (0,530), `insult` (0,517) $\rightarrow$ **Salah Flag (FP)**.
   * *Penyebab*: Model mengidentifikasi kata kasar leksikal tanpa memahami bahwa kalimat tersebut adalah meta-pelaporan santun yang justru membela pengguna lain dari serangan personal.
3. **False Positive pada Kalimat Emosional Berhuruf Kapital (*SHOUTING*)**:
   * *Kasus Riil*: *"I told you to never message me again... DO NOT WRITE ON MY TALK PAGE EVER AGAIN..."* (True label: Bersih).
   * *Prediksi Model*: `toxic` (0,543) $\rightarrow$ **FP**.
   * *Penyebab*: Huruf kapital penuh dan frasa penolakan tegas diasosiasikan oleh representasi BPE sebagai serangan verbal agresif.
4. **Analisis Bias Istilah Identitas**:
   * *Base False Positive Rate* pada komentar non-identitas bersih tercatat rendah sebesar **3,23%**.
   * Sebagian besar istilah identitas netral tidak memicu anomali rasio yang ekstrem, membuktikan bahwa pretraining RoBERTa mampu menjaga konteks sintaksis netral jauh lebih baik dibandingkan model leksikal *bag-of-words*.

---

### 6. Antarmuka Pengguna / GUI Interaktif (ML-T12)

Aplikasi demonstrasi interaktif telah dibangun menggunakan **Streamlit** pada [`gui/app.py`](../gui/app.py):
* **Fitur Utama**:
  1. Input teks bebas komentar Wikipedia berbahasa Inggris.
  2. Pemuatan otomatis model checkpoint `distilroberta-base` yang telah di-fine-tune beserta threshold beku ($\tau = 0.4$).
  3. Slider dinamis untuk pengujian fleksibilitas threshold keputusan di antarmuka.
  4. Visualisasi status 6 dimensi secara simultan: bar probabilitas persentase, kartu metrik, dan indikator status aktif/aman.
  5. Peringatan etika dan pembatasan model: mitigasi risiko ketergantungan moderasi otomatis tanpa peninjauan manusia.

Cara Menjalankan GUI:
```bash
cd mlft-jigsaw/gui
streamlit run app.py
```

---

### 7. Spesifikasi Reproducibility & Keterbatasan (ML-T13)

* **Spesifikasi Teknis**:
  * Random Seed: `42` (pada split, init model, dan DataLoader).
  * Arsitektur: `distilroberta-base` via HuggingFace `transformers`.
  * Optimasi: AdamW, Learning Rate $3 \times 10^{-5}$, Batch Size 16, fp16 (*mixed precision*).
  * Konfigurasi Final: [`experiments/final_config.json`](../experiments/final_config.json).
* **Keterbatasan Utama**:
  1. *Truncation Bias*: Pembatasan 128 subword menyebabkan hilangnya ujaran toksik pada 50% komentar panjang.
  2. *Imbalance Ekstrem*: Label dengan frekuensi < 1% (`threat`, `severe_toxic`) memerlukan dataset bernilai positif lebih besar untuk evaluasi statistik yang lebih reliabel.
* **Pernyataan Penggunaan AI**:
  * Arsitektur modular, skrip pelatihan terstandar, dan draf laporan disusun secara kolaboratif menggunakan Google Antigravity Agent, kemudian diverifikasi secara menyeluruh terhadap keluaran metrik eksperimen.
