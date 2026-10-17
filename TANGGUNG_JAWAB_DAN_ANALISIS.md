# Tanggung Jawab & Penyelesaian — NLP (EDA) dan MLFT-Jigsaw (Eksplorasi Transformer)

> Dokumen ini adalah bahan penjelasan untuk presentasi. Semua angka diambil dari artefak di repo
> (notebook, `experiments/*.json`, `reports/*.md`). Bagian **"Catatan Kritis"** sengaja jujur
> tentang kelemahan metodologi supaya Anda siap menjawab pertanyaan dosen.

---

## 1. Peta Tanggung Jawab

| Proyek | Peran Anda | Artefak utama |
|---|---|---|
| **NLP — TermA** (ekstraksi ASPECT & SENTIMENT, IndoNLU) | **EDA** | [`NLP_Semester_6_EDA.ipynb`](nlp-terma/notebooks/NLP_Semester_6_EDA.ipynb) |
| **MLFT — Jigsaw Toxicity** (multi-label, 6 label) | **Eksplorasi Transformer** | [`model.py`](mlft-jigsaw/src/model.py), [`preprocessing.py`](mlft-jigsaw/src/preprocessing.py), [`run_experiments.py`](mlft-jigsaw/src/run_experiments.py), [`run_improvement_v2.py`](mlft-jigsaw/src/run_improvement_v2.py), `experiments/exp1..exp5` |

---

## 2. NLP — Tanggung Jawab EDA

### 2.1 Pertanyaan yang harus dijawab EDA
1. Seperti apa data (ukuran, skema, label BIO)?
2. Seberapa sulit tugasnya (imbalance, ambiguitas, OOV, boundary)?
3. Keputusan model apa yang dipengaruhi (metrik, fitur CRF, `max_length`, alignment subword)?

### 2.2 Yang sudah Anda kerjakan (sel 0–15, tidak diubah)

| Analisis | Temuan kunci | Dampak keputusan |
|---|---|---|
| Ukuran data | 3.000 train / 1.000 val / 1.000 test; kolom `tokens`, `seq_label` | Dataset kecil → pantau variansi |
| Panjang review | median 12 token, p95 42, p99 64, maks 96 | Kalimat pendek, padding dinamis cukup |
| Frekuensi label | O 30.205, B-ASPECT 5.300, B-SENTIMENT 7.264, I-ASPECT 1.716, I-SENTIMENT 3.249 | Token `O` dominan → jangan pakai accuracy |
| Span extraction | rata-rata 1,77 aspek & 2,42 sentimen/review; 17,2% review tanpa aspek | Model harus bisa prediksi "kosong" |
| Struktur ilegal | 4 baris (0,13%) memuat `I-` yatim | Data hampir bersih |
| Negasi/intensifier | NEG di dalam span sentimen 1.299, INT 1.123 | Sentimen berupa frasa multi-kata |
| Panjang subword IndoBERT | p99 = 71 → `MAX_LEN=72`, hanya 0,73% terpotong | Pilihan `max_length` berbasis data |

### 2.3 Yang saya tambahkan (sel 16–37, **tanpa menghapus apa pun**)

| Sel | Tambahan | Mengapa penting |
|---|---|---|
| A | Proporsi label (%) per split + rata-rata panjang span + rasio `O:entitas` | Membuktikan split konsisten dan alasan memakai span-F1 |
| B | Distribusi panjang span, top kata ASPECT/SENTIMENT, kata yang jadi keduanya | Ambiguitas peran & kesulitan boundary |
| C | OOV rate per label (vs vocab train) + kata paling ambigu (entropi label) | Dasar CRF (fitur leksikal) vs Transformer (subword) |
| D | **Baseline lookup** (label mayoritas per kata) + seqeval report | Lantai performa; membuktikan model perlu konteks |
| E | Jarak aspek→sentimen terdekat, arah (sebelum/sesudah), crosstab jumlah span | Lebar jendela fitur konteks CRF |
| F | Noise ortografi (digit, elongasi, kapital, token 1 huruf) | Batas preprocessing |
| G | Fertility subword IndoBERT per jenis label | Strategi alignment label BIO ke subword |
| H | Duplikasi & overlap train/val/test + duplikat berlabel beda | Cek *leakage* & konsistensi anotasi |
| I | Kepadatan entitas vs panjang kalimat | Padding & efisiensi |
| Ringkasan | Tabel temuan → implikasi pemodelan + keterbatasan | Narasi presentasi |

### 2.4 Cara menjelaskan EDA saat presentasi (alur 4 menit)
1. **Data**: review hotel, 3 split, label BIO 5 kelas.
2. **Masalah utama**: `O` dominan, 17% review tanpa aspek → span-F1 bukan accuracy.
3. **Bukti kesulitan**: baseline lookup (sel D) menunjukkan token-acc tinggi tetapi span-F1 rendah.
4. **Keputusan**: `MAX_LEN=72`, dynamic padding, CRF perlu fitur sufiks/char n-gram bila OOV entitas tinggi, label hanya di subword pertama.
5. **Keterbatasan**: satu domain (hotel), ~3.000 data, daftar negasi dibuat manual.

> [!NOTE]
> Sel baru **belum dijalankan di Colab** (butuh unduh dataset & IndoBERT). Jalankan berurutan dari sel 0 lalu isi
> interpretasi pada kolom "Dampak" dengan angka keluaran sebenarnya sebelum presentasi.

---

## 3. MLFT-Jigsaw — Tanggung Jawab Eksplorasi Transformer

### 3.1 Cakupan tugas (selaras ID rencana `ML-T*`)

| ID | Pekerjaan | Implementasi | Status |
|---|---|---|---|
| ML-T5-01 | Pilih model + head multi-label | `AutoModelForSequenceClassification`, `problem_type="multi_label_classification"`, 6 logit, sigmoid + BCEWithLogits ([model.py](mlft-jigsaw/src/model.py)) | Selesai |
| ML-T5-02 | Alur input → tokenizer → subword → attention mask | [preprocessing.py](mlft-jigsaw/src/preprocessing.py) | Selesai |
| ML-T5-03 | Tanpa stemming/stop-word | Teks asli dipertahankan | Selesai |
| ML-T5-04 / T8 | Eksplorasi hyperparameter terkontrol | exp1–exp4 | Selesai |
| ML-T5-06 | `pos_weight` sebagai satu eksperimen terkontrol | exp4 | Selesai |
| Perbaikan v2 | Head+Tail truncation (exp5) + threshold per-label | [run_improvement_v2.py](mlft-jigsaw/src/run_improvement_v2.py) | Selesai |
| Perbaikan v2 | Focal loss (`exp6`) | — | **Belum dikerjakan** |

### 3.2 Hasil eksplorasi (validation, τ = 0,5)

| Eksperimen | Perubahan dari baseline | Val Macro-F1 | Val Micro-F1 | Catatan |
|---|---|---|---|---|
| exp1_baseline | max_len 128, lr 3e-5 | 0,2957 | 0,5641 | label langka F1 = 0 |
| exp2_maxlen_256 | max_len 256 | 0,3062 | 0,5932 | +0,0105, waktu ~2× |
| exp3_lr_5e5 | lr 5e-5 | 0,3079 | 0,5972 | recall naik, presisi turun |
| **exp4_pos_weight** | `pos_weight` | **0,4129** | 0,5974 | efek terbesar (+0,117) |
| exp5_head_tail | Head 64 + Tail 64 | 0,3840 | — | hanya dilaporkan, bukan exp tunggal |

**Pelajaran teknis utama dari eksplorasi Anda:**
1. Penanganan imbalance (`pos_weight`) memberi dampak jauh lebih besar daripada tuning `lr` atau `max_len`.
2. Memperbesar `max_len` ke 256 hanya memberi kenaikan kecil dengan biaya komputasi ~2×.
3. Kesalahan terbesar adalah *False Negative pada teks panjang* (rata-rata 159,5 kata; 50% > 128 token), memotivasi Head+Tail truncation.
4. Threshold global (0,4) lebih baik dari 0,5 pada validation (Macro-F1 0,4640 vs 0,4129).

### 3.3 Cara menjelaskan alur Transformer (siap presentasi)
```
komentar -> tokenizer BPE (RoBERTa) -> input_ids + attention_mask (max 128)
         -> DistilRoBERTa (6 layer, 768 hidden, 12 head) -> representasi [CLS]/<s>
         -> head linear 6 logit -> sigmoid per label -> bandingkan dengan threshold
Loss: BCEWithLogitsLoss(pos_weight)   (bukan softmax: label tidak saling eksklusif)
```

---

## 4. Catatan Kritis (penting untuk Q&A)

> [!WARNING]
> Poin berikut ditemukan saat menelaah artefak. Perbaiki di laporan atau siapkan jawabannya.

| # | Temuan | Bukti | Saran |
|---|---|---|---|
| 1 | **Eksperimen memakai subsampel kecil, tetapi laporan menyebut 15.958 sampel validation** | [`subsample_split.json`](mlft-jigsaw/data/subsample_split.json): train 1.000, val 300; `run_improvement_v2.py` test = 100 baris. Support val `toxic`=29, `threat`=1 | Tulis jelas "subsampel 1.000/300/100" di laporan; jangan klaim 15.958 |
| 2 | Support sangat kecil → metrik rapuh (`threat` 1 sampel, `identity_hate` 3) | `hyperparameter_exploration_summary.json` | Laporkan interval kepercayaan (bootstrap) atau perbesar sampel |
| 3 | Selisih exp1–exp3 (0,296–0,308) kemungkinan masih **dalam noise** satu seed | 1 run per konfigurasi, 2 epoch | Ulang dengan ≥3 seed sebelum menyimpulkan |
| 4 | v2 "membaik" hanya pada recall; **Macro-F1 test turun 0,4359 → 0,2512** | [`laporan_tahap_3_perbaikan_v2.md`](mlft-jigsaw/reports/laporan_tahap_3_perbaikan_v2.md) | Jangan sebut v2 "lebih baik" tanpa kualifikasi; jelaskan sebagai trade-off recall vs presisi |
| 5 | Val Macro-F1 v2 (0,447) **lebih rendah** dari v1 (0,464), padahal v2 dipilih sebagai final | `final_config_v2.json` vs laporan tahap 2 | Seleksi model seharusnya mengikuti validation; jelaskan alasan memilih v2 |
| 6 | Head-Tail diuji bersamaan dengan perubahan lain (pos_weight teredam, threshold per-label) → efeknya **tidak terisolasi** | exp5 vs exp4 | Ablation bersih: standar vs head-tail dengan konfigurasi sama |
| 7 | Bukti "head-tail berhasil" hanya 2 contoh kualitatif (0,1974→0,2362; 0,0447→0,0697), probabilitas tetap di bawah banyak threshold | Laporan tahap 3 §4.1 | Sebut sebagai ilustrasi, bukan bukti |
| 8 | Model final `distilroberta-base`, rencana awal `roberta-base` | `config.py`, `final_config_v2.json` | Sebut alasan: batas Colab/CPU |
| 9 | `eda_summary.json` menyebut "86,4% < 128 token", tetapi `truncation_pct_128` = 18,78% (→ 81,2%) | `reports/figures/eda_summary.json` | Samakan angka |
| 10 | `model.py` hardcode path DLL `C:\laragon\...` | baris 13–15 | Hapus/bungkus agar portabel |
| 11 | `mlft-jigsaw/PLAN.md` & indeks duplikat dengan root | struktur folder | Rapikan satu sumber |

---

## 5. Saran Pengembangan "Eksplorasi Transformer" (jika masih ada waktu)

Urut dari yang paling bernilai terhadap pemahaman Transformer:

1. **Ablation bersih Head+Tail**: 2 run (standar vs head-tail), konfigurasi identik, 3 seed.
2. **Bandingkan `distilroberta-base` vs `roberta-base`** pada setup sama → seberapa besar kapasitas model mempengaruhi label langka.
3. **Analisis attention / token importance** pada contoh False Negative panjang dan False Positive "meta-talk" (misal dengan attention rollout atau integrated gradients) → menjelaskan *mengapa* model salah.
4. **Layer freezing**: bekukan N layer bawah vs full fine-tuning (hemat waktu, uji apakah representasi bawah sudah cukup).
5. **Focal loss (`exp6`)** yang sudah direncanakan di `03_IMPROVEMENT_PLAN.md`.
6. **Skema chunking + max-pooling** untuk komentar panjang (alternatif head-tail).
7. **Kalibrasi probabilitas** (temperature scaling) sebelum memilih threshold per label.

---

## 6. Ringkasan Satu Paragraf (untuk dibacakan)

> Pada proyek NLP TermA saya bertanggung jawab pada EDA: saya menganalisis distribusi label BIO, panjang review,
> struktur span aspek-sentimen, kualitas anotasi, dan panjang subword IndoBERT sehingga menetapkan span-level F1
> sebagai metrik dan `max_length=72`. Pada MLFT Jigsaw saya mengeksplorasi Transformer (DistilRoBERTa dengan head
> 6 logit, sigmoid + BCE): saya menguji panjang input, learning rate, class weighting, dan Head+Tail truncation.
> Temuan utamanya: class weighting adalah faktor paling berpengaruh pada Macro-F1 (0,296 → 0,413), kesalahan terbesar
> berasal dari pemotongan teks panjang, dan strategi threshold per-label menaikkan recall dengan mengorbankan presisi.
> Keterbatasan: eksperimen memakai subsampel kecil (1.000/300/100) sehingga hasil perlu diverifikasi dengan data dan seed lebih banyak.
