import streamlit as st
import librosa
import joblib
import numpy as np
import matplotlib.pyplot as plt

from utils.process import process_file_advanced
from utils.plot import plot_waveform, plot_spectrogram

st.set_page_config(page_title="Cyberpunk Vehicle Classifier", layout="wide")

cyberpunk_css = """
<style>
h1 {
    text-shadow: 0 0 15px #ff00e6, 0 0 30px #ff00e6;
}

.cyber-card {
    padding: 20px;
    border-radius: 12px;
    background: rgba(255, 0, 153, 0.1);
    border: 1px solid #ff009d;
    box-shadow: 0 0 15px #ff009d;
    margin-bottom: 20px;
}

.pred-box {
    padding: 15px;
    background: rgba(255, 0, 238, 0.15);
    border-left: 5px solid #ff00e6;
    margin-bottom: 18px;
    border-radius: 8px;
}
</style>
"""
st.markdown(cyberpunk_css, unsafe_allow_html=True)


model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")


st.sidebar.title("🟣 Cyberpunk Control Panel")
page = st.sidebar.radio(
    "Navigate",
    ("Upload File", "Realtime Microphone Input (Disabled on Cloud)", "About")
)


def show_probability_chart(classes, probs):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(classes, probs)
    ax.set_title("Probability Distribution")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30)
    st.pyplot(fig)

if page == "Upload File":
    st.title("Vehicle Sound Classifier")

    st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📁 Upload Audio File", type=["wav", "mp3", "ogg", "flac"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:
        with open("input.wav", "wb") as f:
            f.write(uploaded_file.read())

        st.audio("input.wav")
        y, sr = librosa.load("input.wav", sr=22050)

        st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
        st.subheader("Waveform")
        st.pyplot(plot_waveform(y, sr))

        st.subheader("Spectrogram")
        st.pyplot(plot_spectrogram(y, sr))
        st.markdown("</div>", unsafe_allow_html=True)

        features = process_file_advanced("input.wav")
        scaled = scaler.transform([features])
        probs = model.predict_proba(scaled)[0]

        idx = np.argmax(probs)
        pred = label_encoder.classes_[idx]
        confidence = probs[idx] * 100

        st.markdown(
            f"<div class='pred-box'><h2 style='color:#ff00e6'>Prediction: {pred.upper()}</h2>"
            f"<h4>Confidence Score: {confidence:.2f}%</h4></div>",
            unsafe_allow_html=True
        )

        st.subheader("📊 Full Probability Distribution")
        for cls, pr in zip(label_encoder.classes_, probs):
            st.write(f"**{cls}** → {pr*100:.2f}%")

        show_probability_chart(label_encoder.classes_, probs)

elif page == "Realtime Microphone Input (Disabled on Cloud)":
    st.title("🎤 Realtime Vehicle Detection")
    st.warning(
        "Fitur microphone dinonaktifkan karena Streamlit Cloud tidak mendukung akses audio "
        "(sounddevice tidak bisa digunakan di server)."
    )
    st.info(
        "Namun aplikasi tetap berfungsi penuh dengan fitur Upload File."
    )


else:
    st.title("🟣 Cyberpunk Dashboard Info")
    st.write("Final Project – Digital Signal Processing (Filtering + Feature Extraction)")
    st.write("Created with 💜 using Streamlit Cyberpunk Theme")
    st.write("Featuring:")
    st.write("- Upload Audio Classifier")
    st.write("- Waveform & Spectrogram Visualizer")
    st.write("- MFCC + Spectral Features")
    st.write("- Optimized for Streamlit Cloud")
