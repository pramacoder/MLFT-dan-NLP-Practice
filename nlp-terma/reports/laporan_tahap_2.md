# Laporan Tahap II: Pelatihan Model, Evaluasi Span, Analisis Kesalahan, dan Antarmuka Pengguna
## Proyek 2 NLP — TermA: Aspect & Sentiment Extraction pada Ulasan Hotel

---

### 1. Ringkasan Eksekutif & Konfigurasi Eksperimen
* **Tugas**: Ekstraksi span berbasis *sequence labeling BIO* untuk `ASPECT` dan `SENTIMENT` pada ulasan hotel AiryRooms (IndoNLU).
* **Metrik Evaluasi Utama**: **Exact-Match Micro span F1** (standar seqeval). Label `O` diabaikan dalam evaluasi.
* **Protokol Anti-Leakage**: Seluruh hyperparameter dituning secara ketat hanya pada Validation Split (1.000 ulasan). Evaluasi pada Test Split (1.000 ulasan) **hanya dijalankan satu kali** setelah konfigurasi final dibekukan.

---

### 2. Pelatihan & Tuning Hyperparameter CRF (NLP-T8)

#### 2.1 Hasil Grid Search Hyperparameter pada Validation Split
Dieksplorasi ruang parameter keteraturan L1 ($c_1$) dan L2 ($c_2$) sebanyak 9 kombinasi:
| $c_1$ | $c_2$ | Validation Micro-F1 | ASPECT F1 (Val) | SENTIMENT F1 (Val) | Status / Catatan |
|---|---|---|---|---|---|
| **0,01** | **0,01** | **0,9718** | **0,9632** | **0,9781** | **Konfigurasi Terbaik (Terpilih)** |
| 0,01 | 0,10 | 0,9621 | 0,9512 | 0,9700 | Penurunan tipis |
| 0,01 | 1,00 | 0,9183 | 0,9069 | 0,9264 | Regulasi L2 terlalu ketat |
| 0,10 | 0,01 | 0,9679 | 0,9599 | 0,9736 | Kompetitif |
| 0,10 | 0,10 | 0,9586 | 0,9473 | 0,9667 | - |
| 0,10 | 1,00 | 0,9175 | 0,9051 | 0,9264 | - |
| 1,00 | 0,01 | 0,9087 | 0,8951 | 0,9184 | Regulasi L1 terlalu agresif membuang bobot fitur |
| 1,00 | 0,10 | 0,9046 | 0,8900 | 0,9150 | - |
| 1,00 | 1,00 | 0,8832 | 0,8683 | 0,8938 | Performa terendah |

*Model terbaik tersimpan permanen di: `experiments/crf_best_model.joblib`.*

#### 2.2 Hasil Evaluasi Final pada Test Set (Diuji SATU KALI)
Dengan model terpilih ($c_1=0.01, c_2=0.01$):
| Entitas / Level | Precision | Recall | F1-Score | Jumlah Ground Truth Spans |
|---|---|---|---|---|
| **GLOBAL (Micro)** | **0,8892** | **0,8577** | **0,8732** | 4.142 span |
| **ASPECT** | **0,8753** | **0,8330** | **0,8537** | 1.758 span |
| **SENTIMENT** | **0,8993** | **0,8759** | **0,8875** | 2.384 span |

*Interpretasi*:
* Model CRF mencapai Micro-F1 0,8732 pada test set yang belum pernah dilihat.
* Performa pada entitas `SENTIMENT` (F1 0,8875) lebih tinggi dibanding `ASPECT` (F1 0,8537), didukung oleh kuatnya fitur leksikal penanda khusus (`is_negation`, `is_intensifier`).

---

### 3. Analisis Kesalahan Komprehensif 6 Kategori (NLP-T11)
Berdasarkan audit otomatis terhadap seluruh hasil prediksi Test Set (tersimpan di `experiments/crf_error_analysis.json`):

| Kategori Kesalahan | Jumlah Kasus | Deskripsi Karakteristik | Contoh Riil dari Test Set |
|---|---|---|---|
| **1. Missed Aspect (FN)** | 228 | Entitas aspek tidak terdeteksi sama sekali oleh model (diprediksi `O`). Umumnya berupa kata serapan asing atau salah ketik (*typo*). | Ulasan: *"tidak ada westafle ."* $\rightarrow$ True: `ASPECT ('westafle')`, Pred: `O`. Typo kata *wastafel* menjadi OOV leksikal bagi CRF. |
| **2. Missed Opinion (FN)** | 219 | Opini tidak terdeteksi, umumnya berupa frasa ekspresi tidak biasa atau metafora. | Ulasan: *"karena warnanya putih kehitaman ."* $\rightarrow$ True: `SENTIMENT ('putih kehitaman')`, Pred: `O`. |
| **3. Aspect-Sentiment Confusion** | 19 | Posisi span tumpang tindih namun tipe entitas terbalik. | Ulasan: *"sarapan kurang variatif"* $\rightarrow$ True: `ASPECT ('sarapan')`, Pred salah mengelompokkan ke sentimen. Kasus ini sangat jarang (hanya 19 kasus), membuktikan representasi fitur membedakan peran sintaksis dengan baik. |
| **4. False Span (FP)** | 304 | Model memprediksi span pada kata yang sebenarnya dianotasi `O` di data uji. Seringkali akibat kata serapan yang valid namun terlewat oleh annotator manusia. | Ulasan: *"air hangat di bathroom tidak berfungsi"* $\rightarrow$ Pred: `ASPECT ('bathroom')`, True: `O`. Kata *bathroom* secara linguistik adalah aspek, namun tidak dianotasi dalam ground truth. |
| **5. Boundary Error** | 118 | Model dan ground truth mendeteksi tipe entitas yang sama dan saling tumpang tindih, tetapi batas token awal/akhir berbeda. | Ulasan: *"staf recepsionis tidak ramah"* $\rightarrow$ True: `ASPECT ('staf recepsionis')` (2 token), Pred: `ASPECT ('staf')` (1 token). |
| **6. Negation / Modifier Error** | 25 | Kesalahan batas span yang secara khusus melibatkan kata penyangkal (*tidak, belum*) atau kata penguat (*sangat, banget, cukup*). | Ulasan: *"enak banget dapat sarapan"* $\rightarrow$ True: `SENTIMENT ('enak banget dapat')`, Pred: `SENTIMENT ('enak')`. |

---

### 4. Demonstrasi Antarmuka Pengguna (GUI) (NLP-T13)
Aplikasi interaktif berbasis Streamlit telah diimplementasikan di [`gui/app.py`](../gui/app.py):
* **Fitur Utama**:
  1. Input teks ulasan bebas.
  2. Pemilihan arsitektur model (CRF hasil tuning vs IndoBERT).
  3. Visualisasi *inline highlighting* dengan warna pembeda:
     * Biru: entitas `ASPECT`
     * Oranye: entitas `SENTIMENT`
  4. Tabel terstruktur hasil rekonstruksi span lengkap dengan batas indeks token.

Untuk menjalankan GUI:
```bash
cd nlp-terma/gui
streamlit run app.py
```

---

### 5. Reproducibility & Keterbatasan (NLP-T14)
* **Spesifikasi Reproducibility**:
  * Random Seed: `42`
  * Versi Library: `python 3.10.6`, `sklearn-crfsuite 0.5.0`, `torch 2.14.0`, `transformers 5.17.0`, `datasets 5.0.1`
  * Dataset: IndoNLU `terma` (split resmi 3.000 / 1.000 / 1.000).
* **Keterbatasan Model CRF**:
  * CRF bergantung pada kamus leksikal n-gram token lokal $\pm 1$. Akibatnya, kata-kata typo (seperti *westafle*) atau kata serapan baru di luar kosakata data latih gagal dideteksi (*out-of-vocabulary limitation*).
* **Pernyataan Penggunaan AI**:
  * Kode modular, script audit ekstraksi, dan penyusunan draf laporan dibantu oleh Agent AI (Google Antigravity IDE), dan telah divalidasi manual oleh praktikan.
