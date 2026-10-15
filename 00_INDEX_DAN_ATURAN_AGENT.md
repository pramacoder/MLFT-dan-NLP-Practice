# 00 — Indeks & Aturan Bersama (baca ini dulu)

> Dokumen ini adalah pintu masuk. Setiap file plan bersifat modular: satu file = satu proyek. Setiap langkah punya ID unik (mis. `NLP-T3-02`, `ML-T5-01`) supaya mudah dicari (Ctrl+F) dan dirujuk ke agent.

## 1. Peta Dokumen

| File | Isi | Mata kuliah |
|---|---|---|
| `00_INDEX_DAN_ATURAN_AGENT.md` | Indeks, link dataset, aturan umum, cara memakai agent | Keduanya |
| `01_NLP_TermA_plan.md` | Plan Project 2 TermA (ekstraksi ASPECT dan SENTIMENT, CRF vs Transformer) | Project NLP |
| `02_MLFT_Jigsaw_plan.md` | Plan P3 Jigsaw Toxicity (multi-label, threshold, imbalance, bias) | Machine Learning for Text |
| `03_IMPROVEMENT_PLAN.md` | Roadmap rencana aksi perbaikan bertarget berdasarkan error analysis | Keduanya |

## 2. Link Dataset (terverifikasi saat pencarian, 30 Sep 2026)

### Project NLP — TermA (IndoNLU)
- Halaman dataset: https://huggingface.co/datasets/indonlp/indonlu (pilih config `terma`)
- Folder file TermA: https://huggingface.co/datasets/indonlp/indonlu/tree/main/terma (berisi split train, validation, test dalam parquet)
- Paper IndoNLU (rujukan sitasi): https://arxiv.org/abs/1809.03391
- Lisensi: MIT

### Machine Learning for Text — Jigsaw Toxicity
- Mirror Hugging Face (paling praktis, train ±160 ribu baris, test ±306 ribu baris): https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge
- Dataset resmi di HF (dirujuk dosen): https://huggingface.co/datasets/google/jigsaw_toxicity_pred
- Sumber asli Kaggle: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
- Catatan lisensi: label CC0, teks komentar mengikuti CC-SA-3.0 Wikipedia.

> **Hal yang WAJIB dicek agent sebelum lanjut** (belum bisa saya pastikan dari pencarian):
> 1. TermA: apakah `load_dataset("indonlp/indonlu", "terma")` masih jalan di versi `datasets` yang terpasang. Dataset ini berbasis script, sehingga pada `datasets` versi baru bisa gagal. Fallback: baca file parquet langsung dari folder `terma/` di atas.
> 2. Jigsaw: apakah split test di mirror punya label asli atau masih berisi `-1` (pada data Kaggle, baris tak-dinilai berlabel `-1`). Baris berlabel `-1` harus dibuang sebelum evaluasi test.

## 3. Aturan Bersama (berlaku untuk kedua proyek)

1. **Leakage**: test set hanya disentuh SATU kali di akhir. Tidak ada tuning, pemilihan threshold, pemilihan fitur, atau keputusan preprocessing berdasarkan test.
2. **Seed**: `SEED = 42` di `config.py`, dipakai di semua library (python, numpy, torch, transformers).
3. **Reproducibility**: simpan konfigurasi tiap eksperimen (JSON/YAML) di `experiments/`, log metrik per epoch ke CSV.
4. **Modular**: tidak boleh semua logika di satu notebook. Notebook hanya untuk EDA dan laporan; logika di `src/`.
5. **Fungsi yang sama** untuk semua model yang dibandingkan (metrik, rekonstruksi span, normalisasi BIO).
6. **Colab standar**: batch kecil, `fp16=True`, simpan checkpoint ke Google Drive agar tidak hilang saat sesi putus.
7. **Etika data (Jigsaw)**: jangan menaruh contoh ofensif di laporan kecuali perlu; sensor sebagian jika harus ditampilkan.
8. **Kejujuran AI**: dosen meminta pencantuman penggunaan AI di laporan. Catat bagian mana yang dibantu agent.

## 4. Cara Memakai Agent (saran alur kerja)

1. Buat repo kosong per proyek, lalu taruh file plan yang sesuai di root sebagai `PLAN.md`.
2. Beri agent instruksi: *"Baca PLAN.md dan 00_INDEX. Kerjakan HANYA satu tahap per sesi. Setelah selesai, centang checklist, ringkas hasil, dan berhenti menunggu review saya."*
3. Minta agent menjalankan **Definition of Done (DoD)** di akhir tiap tahap sebelum lanjut.
4. Review manual pada titik kontrol: validasi anotasi (NLP), pembagian split dan threshold (MLFT), karena di situlah risiko kesalahan metodologi terbesar.
5. Jangan biarkan agent mengganti model, split, atau metrik tanpa persetujuan Anda.

## 5. Garis Waktu Kasar

| Fase | NLP TermA | MLFT Jigsaw |
|---|---|---|
| Tahap I (analisis dan desain) | Validasi anotasi, EDA, fitur CRF, desain alignment | Formulasi, EDA, split, desain model/loss/metrik |
| Tahap II (implementasi) | CRF, Transformer, evaluasi span, error analysis, GUI | Fine-tuning, hyperparameter, threshold, test, error/bias analysis, GUI |
