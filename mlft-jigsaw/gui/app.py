import os
import sys

# Windows PyTorch DLL fix
torch_lib = r"C:\laragon\bin\python\python-3.10\lib\site-packages\torch\lib"
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)

import streamlit as st
import pandas as pd
import numpy as np

# Menambahkan root project ke sys.path
sys.path.append(os.path.abspath(".."))
from src.config import LABEL_COLS, EXPERIMENTS_DIR
from src.inference import ToxicityPredictor, load_trained_predictor

st.set_page_config(
    page_title="Jigsaw Toxicity Classifier",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Jigsaw Multi-label Toxicity Analyzer")
st.markdown("""
Aplikasi analisis toksisitas teks berbasis model **Pretrained Transformer (RoBERTa / DistilRoBERTa)**.
Model mengestimasi probabilitas independen pada 6 dimensi toksisitas menggunakan aktivasi Sigmoid.
""")

@st.cache_resource
def get_cached_predictor():
    exp_dir = os.path.join(os.path.abspath(".."), "experiments")
    return load_trained_predictor(checkpoint_dir=exp_dir)

base_predictor = get_cached_predictor()
is_live_model = base_predictor.model is not None

# Sidebar settings
st.sidebar.header("⚙️ Status Model & Pengaturan")
if is_live_model:
    st.sidebar.success("🟢 Model: Checkpoint Fine-tuned Aktif")
else:
    st.sidebar.info("🟡 Mode: Mock Heuristic (Eksperimen sedang berjalan)")

default_threshold = float(base_predictor.threshold) if base_predictor.threshold else 0.5
threshold = st.sidebar.slider(
    "Decision Threshold (Batas Keputusan)",
    min_value=0.1,
    max_value=0.9,
    value=default_threshold,
    step=0.05,
    help="Threshold global untuk menentukan apakah label dianggap aktif."
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Dimensi Toksisitas:**
1. `toxic`
2. `severe_toxic`
3. `obscene`
4. `threat`
5. `insult`
6. `identity_hate`
""")

# Input section
comment_input = st.text_area(
    "Masukkan komentar teks (Bahasa Inggris):",
    value="This is a really great and informative Wikipedia article, thank you for writing it!",
    height=120
)

# Set dynamic threshold on predictor
predictor = ToxicityPredictor(
    model=base_predictor.model,
    tokenizer=base_predictor.tokenizer,
    threshold=threshold
)

if st.button("🔍 Analisis Toksisitas", type="primary"):
    if not comment_input.strip():
        st.warning("Silakan masukkan teks komentar terlebih dahulu.")
    else:
        result = predictor.predict_one(comment_input)
        probs = result["probabilities"]
        active = result["active_labels"]

        st.subheader("📊 Hasil Prediksi Multi-label")
        
        # Ringkasan Status
        if result["is_toxic"]:
            st.error(f"🚨 **Terdeteksi Toksik!** Label aktif: `{'`, `'.join(active)}`")
        else:
            st.success("✅ **Komentar Bersih (Non-toksik)** — Tidak ada dimensi yang melampaui threshold.")

        # Tampilan bar probabilitas
        prob_df = pd.DataFrame({
            "Label": LABEL_COLS,
            "Probabilitas (%)": [probs[col] * 100 for col in LABEL_COLS],
            "Threshold (%)": [threshold * 100] * len(LABEL_COLS),
            "Status": ["AKTIF ⚠️" if col in active else "Aman ✔️" for col in LABEL_COLS]
        })

        st.dataframe(prob_df, use_container_width=True)

        # Progress bar per label
        cols = st.columns(3)
        for i, label in enumerate(LABEL_COLS):
            with cols[i % 3]:
                p = probs[label]
                st.metric(
                    label=label.upper().replace("_", " "),
                    value=f"{p * 100:.1f}%",
                    delta="Di atas threshold" if p >= threshold else None,
                    delta_color="inverse"
                )
                st.progress(float(p))

st.markdown("---")
st.caption("""
⚠️ **Catatan Etika & Keterbatasan**: Model dilatih pada dataset Wikipedia Talk Page dan mungkin memiliki bias kontekstual terhadap kata identitas atau istilah slang tertentu. Hasil prediksi adalah estimasi probabilitas statistik dan tidak boleh digunakan sebagai acuan tunggal moderasi tanpa tinjauan manusia.
""")
