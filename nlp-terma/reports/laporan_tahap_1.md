# Laporan Tahap I: Analisis, Validasi Anotasi, dan Desain Sistem
## Proyek 2 NLP — TermA: Aspect & Sentiment Extraction pada Ulasan Hotel

---

### 1. Deskripsi Task & Formulasi Masalah
* **Tugas Pokok**: Ekstraksi span berbasis *sequence labeling BIO* untuk dua entitas utama:
  1. `ASPECT`: Objek, fasilitas, atribut layanan, atau entitas yang dinilai (mis. *kamar mandi*, *ac*, *pelayanan*, *sarapan*).
  2. `SENTIMENT`: Frasa opini atau kata sifat/evaluatif yang merujuk pada penilaian aspek (mis. *bersih*, *tidak berfungsi*, *sangat ramah*).
* **Catatan Penting**: Task ini **bukan** klasifikasi sentimen dokumen (ulasan secara keseluruhan), melainkan ekstraksi pasangan span tingkat token/frasa.
* **Format Pelabelan BIO**:
  * `O`: Token di luar span entitas.
  * `B-ASPECT` / `I-ASPECT`: Awal (*Beginning*) dan kelanjutan (*Inside*) dari span aspek.
  * `B-SENTIMENT` / `I-SENTIMENT`: Awal (*Beginning*) dan kelanjutan (*Inside*) dari span sentimen.
* **Unit Evaluasi**: Exact-match span-level Precision, Recall, dan F1-score (evaluasi berdasarkan ketepatan pasangan `(tipe, start_token, end_token)`). Label `O` diabaikan dalam metrik evaluasi.

---

### 2. Dataset & Split Resmi
Dataset yang digunakan adalah **IndoNLU (config `terma`)** yang berasal dari ulasan hotel platform AiryRooms:
* **Train Split**: 3.000 ulasan | 47.734 token | Rata-rata 15,91 token/ulasan
* **Validation Split**: 1.000 ulasan | 16.124 token | Rata-rata 16,12 token/ulasan
* **Test Split**: 1.000 ulasan | 15.498 token | Rata-rata 15,50 token/ulasan
* **Integritas Metodologi**: Split resmi dipertahankan tanpa perubahan. Test split dikunci dan hanya disentuh satu kali saat pengujian akhir.

---

### 3. Validasi Anotasi & Temuan Kritis (NLP-T2)

#### 3.1 Pengecekan Otomatis (NLP-T2-03)
* Jumlah token = jumlah label pada 100% data (tidak ada data kosong atau panjang berbeda).
* Seluruh label berada dalam `label_list` sah: `['O', 'B-ASPECT', 'I-ASPECT', 'B-SENTIMENT', 'I-SENTIMENT']`.
* **Deteksi Transisi Ilegal pada Seluruh Split**:
  * **Train Set**: Ditemukan **8 kasus transisi ilegal** (misal: token berlabel `O` langsung diikuti `I-ASPECT` tanpa `B-ASPECT`, atau `B-ASPECT` langsung diikuti `I-SENTIMENT`).
  * **Validation Set**: Ditemukan **0 kasus transisi ilegal** (anotasi validasi bersih).
  * **Test Set**: Ditemukan **5 kasus transisi ilegal**.
  * *Solusi Normalisasi*: Telah diimplementasikan fungsi `normalize_bio()` pada `src/spans.py` yang otomatis memperbaiki tag `I-X` tanpa pendahulu sah menjadi `B-X` sebelum rekonstruksi span, sehingga aman bagi CRF maupun Transformer.

#### 3.2 Sampel Validasi Manual (25 Contoh)
Sebanyak 25 sampel representatif (mencakup span multi-token, beberapa span per ulasan, dan kasus transisi ilegal) telah diekstraksi ke file [`data/sampel_validasi.csv`](../data/sampel_validasi.csv).

Beberapa contoh representatif:
| ID | Ulasan | Span ASPECT | Span SENTIMENT | Catatan Validasi |
|---|---|---|---|---|
| `train_218` | *tidak nyaman kamar mandi bau tidak ada gantungan baju dan lingkunganya berisik .* | `kamar mandi`, `gantungan`, `lingkunganya` | `tidak nyaman`, `bau`, `tidak ada`, `baju` *(salah anotasi)*, `berisik` | Terdapat transisi ilegal: token "baju" dianotasi `I-SENTIMENT` padahal bagian dari `gantungan baju`. |
| `train_607` | *peralatan berantakan . selebihnya oke banget .* | `peralatan`, `selebihnya` | `berantakan`, `oke banget` | Transisi ilegal: "peralatan" (`B-ASPECT`) diikuti "berantakan" (`I-SENTIMENT`) tanpa `B-SENTIMENT`. |
| `train_237` | *kamarnya bersih , nyaman , ac nya dingin , sarapannya juga enak . dan sangat rekomended untuk liburan .* | `kamarnya`, `ac nya`, `sarapannya` | `bersih`, `nyaman`, `dingin`, `enak`, `sangat rekomended` | Kata penguat "sangat" dimasukkan ke dalam span SENTIMENT (*sangat rekomended*). Klitik "nya" dimasukkan ke dalam ASPECT (*ac nya*). |
| `train_817` | *segala sesuatunya oke tetapi kunci kamar mandinya perlu di perbaiki .* | `segala sesuatunya`, `kunci kamar mandinya` | `oke`, `perlu di perbaiki` | Aspek majemuk panjang 3 token (`kunci kamar mandinya`). |
| `train_2814` | *pelayanan memuaskan , makanan enak , cuma kamar mandi agak bau . over all oke .* | `pelayanan`, `makanan`, `kamar mandi`, `over all` | `memuaskan`, `enak`, `agak bau`, `oke` | Modifikator pelemah "agak" masuk ke SENTIMENT (*agak bau*). Ungkapan "over all" dijadikan entitas ASPECT. |

#### 3.3 Ringkasan & Bukti Statistik Batas Ambigu (NLP-T2-05)
Berdasarkan verifikasi pada seluruh Training Set (47.734 token), klaim penempatan intensifier dan negasi ke dalam span `SENTIMENT` terbukti secara empiris:
| Penanda Leksikal | Total Kemunculan di Train | Di Dalam Span `SENTIMENT` (%) | Berlabel `O` (%) | Keterangan Distribusi |
|---|---|---|---|---|
| `kurang` | 465 | **434 (93,33%)** | 31 (6,67%) | Dominan kuat sebagai awal span opini negatif (*kurang bersih*). |
| `sangat` | 384 | **356 (92,71%)** | 28 (7,29%) | Sangat konsisten menjadi bagian dari span penguat (*sangat nyaman*). |
| `banget` | 215 | **180 (83,72%)** | 34 (15,81%) | Mayoritas di dalam span opini (*enak banget*); sisanya `O` saat berdiri sendiri di akhir klausa. |
| `tidak` | 1.217 | **844 (69,35%)** | 373 (30,65%) | 69,35% menyatu dengan kata sifat (*tidak dingin*); 30,65% berlabel `O` pada klausa deskriptif/faktual (*tidak ada lift*). |

1. **Frasa Benda Majemuk**: Aspek fisik majemuk (*kamar mandi, kolam renang, air panas*) umumnya dianotasi sebagai 1 span (`B-ASPECT I-ASPECT`). Namun batas antara fasilitas dan atribut kualitasnya (*kunci kamar mandinya*, *kebersihan seprai dan selimut*) rentan tidak seragam.
2. **Sentimen Tanpa Aspek Implisit**: Opini umum seperti *lumayan*, *recomended*, *puas* berdiri sendiri sebagai span `SENTIMENT` tanpa entitas `ASPECT` eksplisit.

---

### 4. Temuan EDA Khusus Task & Implikasi Pemodelan (NLP-T3)

Grafik tersimpan di folder `reports/figures/`:
* `distribusi_panjang_ulasan.png`
* `frekuensi_label_bio.png`
* `panjang_span_tipe.png`

| Aspek Analisis | Temuan Statistik (Train Set) | Implikasi ke Desain Model |
|---|---|---|
| **Distribusi Panjang & Truncation** | Median: 12 token, Rata-rata: 15,9 token, Maksimum: 96 token. Truncation rate pada `MAX_LEN=128`: **0.00%**. | Batas panjang `MAX_LEN=128` sangat aman dan optimal. Tidak ada informasi yang terpotong, sekaligus menghemat memori komputasi Colab. |
| **Ketimpangan Label (Imbalance)** | Label `O`: 63,28% (30.205 token). Rasio `O` terhadap non-`O`: **1,72 : 1**. | Ketimpangan 1,72:1 tergolong ringan hingga moderat. Akurasi per-token tetap berisiko terdistorsi oleh `O`, sehingga metrik utama yang ditetapkan adalah **Micro span F1** (standar seqeval). |
| **Komposisi Span (Single vs Multi-Token)** | Aspek multi-token: **27,19%**. Sentimen multi-token: **37,70%**. Rata-rata: 4,19 span per ulasan. | Tingginya persentase multi-token menimbulkan risiko *boundary error*. Model CRF membutuhkan fitur konteks lokal $\pm 1$ dan n-gram afiks; Transformer membutuhkan representasi kontekstual subword. |
| **Penanda Leksikal (Negasi & Penguat)** | Negasi: 1.764 kemunculan (*tidak*: 1.217, *kurang*: 465). Intensifier: 1.325 kemunculan (*sangat*: 384, *banget*: 215, *cukup*: 190). | Bukti empiris membuktikan >83–93% kemunculan kata penguat/negasi ini berada di dalam span opini. Menambahkan fitur biner (`is_negation`, `is_intensifier`, `-1:is_negation`) pada CRF sangat beralasan secara linguistik. |

---

### 5. Desain Fitur CRF (NLP-T4)

#### 5.1 Spesifikasi Fitur
| Kelompok Fitur | Nama Fitur | Contoh Nilai | Alasan / Peran Linguistik |
|---|---|---|---|
| **Bias** | `bias` | `1.0` | Bobot dasar CRF untuk baseline frekuensi label. |
| **Leksikal** | `word.lower()` | `"kamar"`, `"dingin"` | Kata dasar menentukan leksikon aspek dan opini. |
| **Morfologis / Afiks** | `word[-3:]`, `word[-2:]`, `word[:3]`, `word[:2]` | `"-nya"`, `"-kan"`, `"ber-"`, `"di-"` | Menangkap pola morfem bahasa Indonesia (mis. sufiks *-nya* sering pada aspek, prefiks *ter-* atau *ber-* pada kondisi/sentimen). |
| **Ortografis & Bentuk** | `word.isupper()`, `word.istitle()`, `word.isdigit()`, `word.shape` | `False`, `False`, `False`, `"xxxxx"` | Kerangka dosen mendefinisikan *word shape* per karakter (misal *kamar* $\rightarrow$ *xxxxx*, *AC* $\rightarrow$ *XX*, *200rb* $\rightarrow$ *dddxx*). |
| **Leksikal Domain Khusus** | `is_negation`, `is_intensifier` | `True` / `False` | Membedakan awal frasa opini negatif (*tidak*, *kurang*) dan penguat (*sangat*, *banget*). |
| **Konteks Lokal $\pm 1$** | `-1:word.lower()`, `+1:word.lower()`, `-1:is_negation` | `"-1:word.lower()='kamar'"` | Memberikan konteks frasa majemuk (*kamar mandi*) dan pergeseran polaritas (*tidak + berfungsi*). |
| **Batas Kalimat** | `BOS`, `EOS` | `True` / `False` | Menandai token pertama (*Beginning of Sentence*) dan terakhir (*End of Sentence*). |

#### 5.2 Contoh Dictionary Fitur Otomatis (Output Riil Kode)
Dihasilkan langsung dari pemanggilan `sentence2features(["kamar", "mandi", "sangat", "bersih", "."])`:

* **Token 0 (`"kamar"`) — Token Awal Kalimat (BOS)**:
  ```json
  {
    "bias": 1.0,
    "word.lower()": "kamar",
    "word[-3:]": "mar",
    "word[-2:]": "ar",
    "word[:3]": "kam",
    "word[:2]": "ka",
    "word.isupper": false,
    "word.istitle": false,
    "word.isdigit": false,
    "word.shape": "xxxxx",
    "is_negation": false,
    "is_intensifier": false,
    "BOS": true,
    "+1:word.lower()": "mandi",
    "+1:word.istitle()": false,
    "+1:word.isupper()": false,
    "+1:word.shape": "xxxxx"
  }
  ```

* **Token 2 (`"sangat"`) — Kata Penguat**:
  ```json
  {
    "bias": 1.0,
    "word.lower()": "sangat",
    "word[-3:]": "gat",
    "word[-2:]": "at",
    "word[:3]": "san",
    "word[:2]": "sa",
    "word.isupper": false,
    "word.istitle": false,
    "word.isdigit": false,
    "word.shape": "xxxxxx",
    "is_negation": false,
    "is_intensifier": true,
    "-1:word.lower()": "mandi",
    "-1:word.istitle()": false,
    "-1:word.isupper()": false,
    "-1:word.shape": "xxxxx",
    "-1:is_negation": false,
    "-1:is_intensifier": false,
    "+1:word.lower()": "bersih",
    "+1:word.istitle()": false,
    "+1:word.isupper()": false,
    "+1:word.shape": "xxxxxx"
  }
  ```

---

### 6. Desain Transformer & Kebijakan Alignment (NLP-T5)
* **Model Pilihan**: `indobenchmark/indobert-base-p1`
  * Model pretrained bahasa Indonesia ukuran base (~124M parameter, 12 layer), memiliki tokenizer fast yang teruji stabil, dan efisien saat dijalankan pada GPU Colab T4.
* **Kebijakan Alignment Label Subword**:
  * **Subword pertama** dari sebuah kata asli mendapatkan label ground truth (mis. ID `B-ASPECT`).
  * **Subword lanjutan** (mis. `##nya`, `##kan`) dan token khusus (`[CLS]`, `[SEP]`, `[PAD]`) diberi label `-100` agar diabaikan dalam kalkulasi `CrossEntropyLoss`.
* **Rencana Prediksi ke Token Asli**:
  * Saat evaluasi/inferensi, prediksi diambil hanya dari logit subword pertama untuk setiap kata asli menggunakan `word_ids()`.

---

### 7. Rancangan Eksperimen & Anti-Leakage (NLP-T6)
1. **Metrik Utama Tunggal**:
   * Ditetapkan **Micro span F1** (exact-match span level, standar seqeval) sebagai metrik acuan tunggal untuk pemilihan model, checkpoint, dan tuning parameter.
2. **Model 1 (CRF Baseline)**:
   * Algoritma optimasi: `lbfgs`.
   * Ruang tuning hyperparameter pada Validation Split: $c_1, c_2 \in \{0.01, 0.1, 1.0\}$.
   * Evaluasi metrik exact-match Micro span F1 pada Validation untuk memilih kombinasi $c_1, c_2$ terbaik.
3. **Model 2 (IndoBERT Token Classification)**:
   * Base architecture: `AutoModelForTokenClassification` dengan 5 label (`O`, `B-ASPECT`, `I-ASPECT`, `B-SENTIMENT`, `I-SENTIMENT`).
   * Hyperparameter: `learning_rate=2e-5` (opsional $5 \times 10^{-5}$), `batch_size=8`, `epochs=3`, `weight_decay=0.01`, `max_length=128`.
   * Checkpoint terpilih berdasarkan Micro span F1 tertinggi pada Validation set.
4. **Protokol Anti-Leakage**:
   * Test split dikunci dan sama sekali tidak digunakan dalam pemilihan kata negasi/intensifier, pembentukan fitur, penentuan hyperparameter, maupun pemilihan checkpoint model.