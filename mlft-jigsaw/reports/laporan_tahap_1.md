# Laporan Tahap I: Analisis Masalah, EDA Berorientasi Keputusan ML, dan Desain Sistem
## Mata Kuliah Machine Learning for Text — Project 3: Jigsaw Toxic Comment Classification

---

### 1. Formulasi Masalah (ML-T1)

#### 1.1 Tabel Formulasi Tugas
| Komponen | Spesifikasi / Keterangan |
|---|---|
| **Input $X$** | Satu rangkaian teks komentar berbahasa Inggris dari Wikipedia Talk Page (kolom `comment_text`). |
| **Target $Y$** | Vektor biner 6 dimensi: `[toxic, severe_toxic, obscene, threat, insult, identity_hate]` $\in \{0, 1\}^6$. |
| **Unit Prediksi** | Level dokumen komentar tunggal (*per comment*). |
| **Paradigma** | *Supervised Learning*. |
| **Tipe Masalah** | *Multi-label Binary Classification* (setiap komentar dapat memiliki 0 label aktif, 1 label, atau beberapa label aktif sekaligus). |
| **Arsitektur Model** | Pretrained English Transformer (`roberta-base` / `distilroberta-base`) + Classification Head 6 Logits. |
| **Fungsi Aktivasi Inferensi** | **Sigmoid** independen per logit: $\hat{p}_j = \frac{1}{1 + e^{-z_j}}$ untuk $j \in \{1, \dots, 6\}$. |
| **Fungsi Loss Training** | **Binary Cross-Entropy with Logits Loss (`BCEWithLogitsLoss`)**: $\mathcal{L} = -\sum_{j=1}^6 \left[ y_j \log \sigma(z_j) + (1 - y_j) \log (1 - \sigma(z_j)) \right]$. |
| **Keluaran Akhir Inferensi** | 6 nilai probabilitas kontinu $\hat{p}_j \in [0, 1]$ serta daftar label aktif yang melampaui ambang batas keputusan (*threshold* $\tau$). |

#### 1.2 Justifikasi Matematis: Mengapa Sigmoid + BCE (Bukan Softmax + CE)?
* Pada klasifikasi *multi-class* standar, fungsi Softmax memaksakan total probabilitas $\sum_{j} p_j = 1$, yang berarti kelas-kelas bersifat saling lepas (*mutually exclusive*).
* Pada kasus toksisitas teks Jigsaw, sebuah komentar seringkali merupakan ujaran kotor (`obscene`) sekaligus hinaan (`insult`) dan bernada ancaman (`threat`). Label-label ini **tidak saling lepas**.
* Penggunaan fungsi Sigmoid memodelkan 6 tugas klasifikasi biner Bernoulli secara paralel dan independen. Kombinasi dengan `BCEWithLogitsLoss` menggabungkan operasi Sigmoid dan Cross-Entropy ke dalam satu fungsi terstabilisasi secara numerik (*log-sum-exp trick*), mencegah *underflow* gradien.

---

### 2. Pemahaman dan Validasi Data (ML-T2)

#### 2.1 Ringkasan Skema & Karakteristik Data
* **Kolom Input**: `id` (string heksadesimal) dan `comment_text` (string teks bebas).
* **Kolom Target**: 6 kolom biner bertipe integer: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.
* **Ukuran Dataset**:
  * **Train Set**: **159.571 baris** komentar.
  * **Test Set Mentah**: **153.164 baris** komentar.
  * **Test Labels**: **153.164 baris** label.
  * **Validation Set**: **Tidak disediakan secara resmi oleh penyelenggara/Kaggle**. Wajib dibentuk secara independen dari Train set dengan stratifikasi multi-label.

#### 2.2 Hasil Audit Kualitas Data (ML-T2-02)
* **Komentar Kosong / NaN**: **0 baris** (seluruh baris memiliki teks valid).
* **Duplikasi Teks**: **0 baris** duplikasi identik.
* **Nilai Label di Luar $\{0, 1\}$ pada Train**: **0 kasus** (seluruh label valid 0 atau 1).

#### 2.3 Titik Kritis: Penanganan Label Test `-1` (ML-T2-03)
* Dari **153.164 baris** pada `test_labels.csv`, ditemukan **89.186 baris (58,23%)** yang memiliki nilai `-1` pada seluruh kolom label.
* *Latar Belakang*: Pada kompetisi Kaggle aslinya, baris berlabel `-1` adalah data pengacau (*dummy/unscored data*) untuk mencegah *probing submission*.
* *Keputusan Metodologi*: Seluruh **89.186 baris** tak-dinilai ini **wajib dibuang sebelum evaluasi akhir**. Test set evaluasi yang valid berjumlah tepat **63.978 baris**. Modul [`src/data_loader.py`](../src/data_loader.py) telah diuji dan otomatis memfilter baris tersebut.

---

### 3. EDA Berorientasi Keputusan Machine Learning (ML-T3)

Grafik visualisasi tersimpan di folder `reports/figures/`:
* `distribusi_6_label.png`
* `heatmap_cooccurrence.png`
* `distribusi_panjang_subword.png`

#### 3.1 Frekuensi Label & Ketimpangan Ekstrem (ML-T3-01 & ML-T3-04)
Total data latih: 159.571 komentar.
* **Komentar Bersih (Non-toksik)**: **143.346 ulasan (89,83%)**.
* **Komentar Toksik ($\ge 1$ label aktif)**: **16.225 ulasan (10,17%)**.

| Label Target | Frekuensi Positif | Proporsi (%) | Kategori Kelangkaan |
|---|---|---|---|
| `toxic` | 15.294 | 9,584% | Mayoritas Toksik |
| `severe_toxic` | 1.595 | 1,000% | Sangat Langka |
| `obscene` | 8.449 | 5,295% | Menengah |
| `threat` | 478 | **0,300%** | **Paling Langka (Ekstrem)** |
| `insult` | 7.877 | 4,936% | Menengah |
| `identity_hate` | 1.405 | 0,880% | Sangat Langka |

* **Keputusan ML yang Dipengaruhi**:
  1. Akurasi global akan menipu (model yang memprediksi nol untuk semua komentar akan mencapai akurasi ~90%).
  2. Metrik utama evaluasi wajib menggunakan **Macro-F1**, yang menghitung rata-rata F1 per kelas dengan bobot setara, sehingga performa pada label langka (`threat`, `severe_toxic`, `identity_hate`) tidak tertutup oleh dominasi komentar bersih.

#### 3.2 Label Cardinality & Co-occurrence (ML-T3-02 & ML-T3-03)
* **Label Cardinality (Seluruh Data)**: **0,22** label aktif per komentar.
* **Label Cardinality (Hanya Komentar Toksik)**: **2,16** label aktif per komentar.
  * Artinya, jika suatu komentar mengandung toksisitas, rata-rata ia mengaktifkan lebih dari 2 dimensi sekaligus.
* **Analisis Ko-okurensi**:
  * Hampir seluruh kemunculan `severe_toxic` (1.595) beririsan langsung dengan `toxic` (1.595) dan `obscene` (1.517).
  * Label `insult` dan `obscene` memiliki korelasi tumpang tindih tertinggi (6.155 komentar mengandung keduanya sekaligus).
* **Keputusan ML yang Dipengaruhi**:
  * Representasi *shared hidden representations* pada Transformer sangat cocok karena korelasi antar-fitur laten dapat dipelajari bersama (*joint representation learning*).

#### 3.3 Panjang Komentar Subword & Truncation Rate per Label (ML-T3-05)
Berdasarkan tokenisasi RoBERTa:
* **Median Keseluruhan**: 52 subword token (Rata-rata: 93,8 token).
* **Karakteristik Panjang Berdasarkan Kategori Komentar**:
  * Komentar bersih (*clean*) memiliki median lebih panjang (**54 token**) dibanding komentar toksik (**30–36 token**). Namun sejumlah ulasan toksik berbentuk serangan spam berulang yang sangat panjang.

| Kategori / Label | Jumlah Sampel Positif | Median Token | Rata-rata Token | Truncation @128 (%) | Truncation @256 (%) |
|---|---|---|---|---|---|
| `toxic` | 15.294 | 36,0 | 84,3 | 11,98% | 4,80% |
| `severe_toxic` | 1.595 | 30,0 | 146,7 | **14,17%** | **9,78%** |
| `obscene` | 8.449 | 33,0 | 84,1 | 11,30% | 4,85% |
| `threat` | 478 | 35,5 | 96,9 | **10,88%** | **5,23%** |
| `insult` | 7.877 | 34,0 | 81,7 | 10,65% | 4,24% |
| `identity_hate` | 1.405 | 35,0 | 95,5 | 13,17% | 5,77% |
| *Komentar Bersih (Baseline)* | 143.346 | 54,0 | 95,1 | 19,75% | 7,05% |

* **Keputusan ML yang Dipengaruhi**:
  * Truncation sebesar 10–14% pada label toksik (khususnya `severe_toxic` dan `threat`) tergolong material. Oleh karena itu, perbandingan **`max_length in {128, 256}` dimasukkan secara resmi ke dalam ruang eksperimen hyperparameter**, di mana `max_length=128` menjadi baseline hemat memori dan `max_length=256` diuji untuk memitigasi pemotongan sinyal toksik.

#### 3.4 Analisis Empiris Potensi Bias Kontekstual & Ilustrasi Kasus (ML-T3-06)

##### Bukti Empiris Data Latih (Kata Kunci "Gay"):
Pengecekan pada seluruh Training Set (159.571 baris) menunjukkan bahwa kata *"gay"* muncul pada **867 komentar (0,543%)**:
* **515 komentar (59,40%)** berlabel **toksik** (terbanyak: `toxic` 57,55%, `insult` 32,53%, `identity_hate` 30,57%).
* **352 komentar (40,60%)** berlabel **bersih / non-toksik**.
* *Interpretasi Metodologis*: Karena mayoritas (~59,4%) kemunculan kata tersebut di data latih menyertai serangan personal/ujaran kebencian, model rentan mengasosiasikan leksikon kata itu sendiri dengan toksisitas secara naif, sehingga berpotensi menghasilkan *false positive* pada 40,6% komentar yang sebenarnya netral.

##### Contoh Kasus Non-Ofensif (Ilustrasi Rekayasa Peneliti):
*(Catatan: Tiga kalimat di bawah ini adalah **ilustrasi buatan sendiri / konstruksi peneliti** untuk menguji potensi false positive, BUKAN kutipan dari dataset asli Wikipedia)*
1. **Identitas Agama / Etnis**:
   * *Ilustrasi*: `"I am a proud Muslim and I love reading Wikipedia articles about world history."`
   * *Risiko*: Overfitting pada kata identitas memicu false positive `identity_hate`.
2. **Orientasi Seksual**:
   * *Ilustrasi*: `"Many gay and lesbian youth struggle with mental health and deserve our support."`
   * *Risiko*: Kata identitas memicu false positive `insult` / `identity_hate` meskipun kalimat bersifat afirmatif.
3. **Slang / Pujian Bahasa Gaul**:
   * *Ilustrasi*: `"That guitar solo was totally wicked and sick! Absolute killer performance."`
   * *Risiko*: Kata dengan makna leksikal harfiah agresif (*sick, wicked, killer*) memicu false positive `toxic`.

---

### 4. Strategi Split & Protokol Anti-Leakage (ML-T4)

#### 4.1 Pembagian Split Terstratifikasi Multi-Label (ML-T4-01 & ML-T4-02)
Karena data tidak memiliki validation split resmi dan memiliki label yang sangat langka (`threat` hanya 478 sampel dari 160 ribu ulasan), random split biasa berisiko membagi label langka secara timpang.

Digunakan metode **`MultilabelStratifiedShuffleSplit`** (rasio 90:10, `SEED = 42`):
* **Train Split**: **143.613 ulasan**
* **Validation Split**: **15.958 ulasan**
* **Verifikasi Distribusi Label (Train vs Val)**:
  | Label | Proporsi Train (%) | Proporsi Val (%) | Selisih Absolut (%) |
  |---|---|---|---|
  | `toxic` | 9,585% | 9,581% | 0,0034% |
  | `severe_toxic` | 0,999% | 1,003% | 0,0034% |
  | `obscene` | 5,295% | 5,295% | 0,0004% |
  | `threat` | 0,299% | 0,301% | 0,0014% |
  | `insult` | 4,936% | 4,938% | 0,0018% |
  | `identity_hate` | 0,880% | 0,884% | 0,0034% |

Indeks split telah disimpan secara permanen di [`data/split_ids.json`](../data/split_ids.json) untuk menjamin 100% *reproducibility*.

#### 4.2 Keterbatasan Ukuran Sampel Validation (Catatan Metodologis)
* Pada Validation Split (15.958 baris), label `threat` **hanya memiliki 48 sampel positif (0,301%)**.
* *Keterbatasan*: Evaluasi metrik Precision, Recall, dan F1 pada label `threat` di validation set akan sangat rentan terhadap fluktuasi (*noisy*). Satu atau dua kesalahan prediksi dapat mengubah skor F1 secara drastis. Hal ini dicatat sebagai keterbatasan intrinsik dari distribusi data.

#### 4.3 Protokol Anti-Leakage (ML-T4-03)
* Data test **TIDAK PERNAH** disentuh selama proses pelatihan, pemilihan checkpoint, penentuan hyperparameter, maupun pemilihan threshold probabilitas.
* Seluruh keputusan pemodelan dibekukan (*frozen*) hanya berdasarkan data validation.

---

### 5. Desain Model, Loss, Hyperparameter, dan Metrik (ML-T5)

1. **Pemilihan Model**:
   * Rekomendasi: `roberta-base` (alternatif ringan untuk efisiensi komputasi Colab: `distilroberta-base`).
   * *Alasan*: RoBERTa dilatih dengan korpus teks masif tanpa *next-sentence prediction*, memiliki tokenizer Byte-Pair Encoding (BPE) yang sangat tangguh terhadap variasi typo dan slang di internet.
2. **Kebijakan Preprocessing**:
   * **Tidak melakukan** stemming, pembuangan stop-word, atau penghapusan tanda baca. Tanda seru (*!*), huruf kapital (*SHOUTING*), dan kata seru adalah sinyal kuat dari agresivitas/toksisitas teks.
3. **Ruang Eksperimen Hyperparameter Terkontrol**:
   * **Max Length**: $\{128, 256\}$ (dieksplorasi secara resmi untuk membandingkan dampak truncation).
   * **Learning rate**: $\{2 \times 10^{-5}, 3 \times 10^{-5}, 5 \times 10^{-5}\}$.
   * **Batch size**: 16 (atau 32).
   * **Epoch**: 3.
   * **Weight decay**: 0,01.
   * **Class Weighting (`pos_weight`)**: Direncanakan sebagai satu eksperimen terkontrol pada `BCEWithLogitsLoss` dengan menghitung bobot positif dari train split: $w_j = \frac{N_{\text{neg}, j}}{N_{\text{pos}, j}}$, untuk menguji apakah dapat mengangkat recall pada label langka.
4. **Metrik Evaluasi & Rencana Threshold**:
   * **Macro-F1 Validation** ditetapkan sebagai metrik penentu utama sebelum melihat test.
   * **Grid Threshold Search**: Akan dilakukan pencarian threshold global pada validation set pada kandidat $\tau \in \{0.3, 0.4, 0.5, 0.6, 0.7\}$ untuk memaksimalkan Macro-F1 sebelum membekukan konfigurasi akhir (`final_config.json`).
   * Metrik komplementer: Micro-F1, per-label P/R/F1, dan Hamming Loss.

---

### 6. Kerangka Modular & Prototipe GUI (ML-T6)
Seluruh arsitektur modular telah dibuat dan siap untuk Tahap II:
* [`src/preprocessing.py`](../src/preprocessing.py): Tokenisasi dinamis & target float32.
* [`src/model.py`](../src/model.py): Head klasifikasi multi-label & Sigmoid probabilities.
* [`src/evaluation.py`](../src/evaluation.py): Metrik Macro/Micro F1, per-label P/R/F1, Hamming Loss, dan pencarian threshold.
* [`src/train.py`](../src/train.py): Loop fine-tuning terstandar & logging.
* [`src/inference.py`](../src/inference.py): Kelas inferensi tunggal untuk CLI dan antarmuka pengguna.
* [`gui/app.py`](../gui/app.py): Prototipe GUI interaktif Streamlit dengan visualisasi metrik 6 dimensi, pengatur threshold dinamis, dan catatan etika/bias model.
