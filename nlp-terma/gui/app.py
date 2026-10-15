import streamlit as st
import os
import sys
import joblib
import pandas as pd

# Path setup
sys.path.append(os.path.abspath(".."))
from src.config import EXPERIMENTS_DIR
from src.crf_features import sentence2features
from src.spans import reconstruct_span_texts, normalize_bio

st.set_page_config(
    page_title="TermA — Ekstraksi Aspek & Opini Ulasan Hotel",
    page_icon="🏨",
    layout="wide"
)

st.title("🏨 TermA: Aspect & Opinion Span Extractor")
st.markdown("""
Aplikasi demonstrasi ekstraksi span **ASPECT** (fasilitas/objek) dan **SENTIMENT** (opini evaluatif)
pada ulasan hotel platform AiryRooms berbahasa Indonesia.
""")

# Load saved CRF model
crf_model_path = os.path.join(EXPERIMENTS_DIR, "crf_best_model.joblib")
crf_model = None
if os.path.exists(crf_model_path):
    crf_model = joblib.load(crf_model_path)

# Sidebar
st.sidebar.header("⚙️ Pengaturan Model")
model_choice = st.sidebar.selectbox(
    "Pilih Arsitektur Model:",
    ["CRF (Baseline Terpilih: c1=0.01, c2=0.01)", "IndoBERT Token Classification"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Keterangan Warna Entitas:**
* <span style="background-color: #2980b9; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold;">ASPECT</span>: Fasilitas, ruangan, layanan, atribut hotel.
* <span style="background-color: #e67e22; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold;">SENTIMENT</span>: Frasa evaluatif, opini, atau keluhan.
""", unsafe_allow_html=True)

# Main input
default_review = "kamar mandi sangat bersih dan ac nya dingin banget , tapi pelayanannya kurang ramah ."
user_review = st.text_area("Masukkan teks ulasan hotel:", value=default_review, height=100)

def render_highlighted_text(tokens, labels):
    clean_labels = normalize_bio(labels)
    highlighted_tokens = []
    for tok, lbl in zip(tokens, clean_labels):
        if "ASPECT" in lbl:
            highlighted_tokens.append(f"<span style='background-color: #d4e6f1; border-bottom: 2px solid #2980b9; font-weight: 600; padding: 1px 4px; border-radius: 3px;'>{tok}</span>")
        elif "SENTIMENT" in lbl:
            highlighted_tokens.append(f"<span style='background-color: #fdebd0; border-bottom: 2px solid #e67e22; font-weight: 600; padding: 1px 4px; border-radius: 3px;'>{tok}</span>")
        else:
            highlighted_tokens.append(tok)
    return " ".join(highlighted_tokens)

if st.button("🔍 Ekstraksi Span", type="primary"):
    if not user_review.strip():
        st.warning("Silakan masukkan ulasan hotel terlebih dahulu.")
    else:
        # Preprocessing sederhana split kata
        tokens = user_review.strip().split()
        
        if "CRF" in model_choice:
            if crf_model is None:
                st.error("Model CRF belum ditemukan di folder experiments/. Jalankan train_crf.py terlebih dahulu.")
            else:
                feats = [sentence2features(tokens)]
                pred_labels = crf_model.predict(feats)[0]
                spans_dict = reconstruct_span_texts(tokens, pred_labels)

                st.subheader("📝 Teks Ber-Highlight")
                rendered_html = render_highlighted_text(tokens, pred_labels)
                st.markdown(f"<div style='line-height: 2.2; font-size: 1.15rem; background: #f8f9fa; padding: 16px; border-radius: 8px; border: 1px solid #e9ecef;'>{rendered_html}</div>", unsafe_allow_html=True)

                st.subheader("📋 Tabel Hasil Rekonstruksi Span")
                table_rows = []
                for s, start, end in spans_dict["ASPECT"]:
                    table_rows.append({"Tipe": "ASPECT", "Teks Span": s, "Posisi Token": f"{start} - {end}"})
                for s, start, end in spans_dict["SENTIMENT"]:
                    table_rows.append({"Tipe": "SENTIMENT", "Teks Span": s, "Posisi Token": f"{start} - {end}"})

                if table_rows:
                    df_res = pd.DataFrame(table_rows)
                    st.dataframe(df_res, use_container_width=True)
                else:
                    st.info("Tidak ada span ASPECT maupun SENTIMENT yang terdeteksi.")
        else:
            st.info("Model IndoBERT sedang dalam antrian pelatihan fine-tuning GPU Colab. Silakan gunakan model CRF untuk demonstrasi real-time saat ini.")
