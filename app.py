import streamlit as st
import librosa
import joblib
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import matplotlib.pyplot as plt

from utils.process import process_file_advanced
from utils.plot import plot_waveform, plot_spectrogram

st.set_page_config(page_title="Cyberpunk Vehicle Classifier", layout="wide")

cyberpunk_css = """
<style>

body {
    background: #06070d !important;
    color: #e0e0e0;
}

/* HOLOGRAM GRID BACKGROUND */
body::before {
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background:
        linear-gradient(90deg, rgba(0,255,255,0.12) 1px, transparent 1px),
        linear-gradient(rgba(0,255,255,0.12) 1px, transparent 1px);
    background-size: 45px 45px;
    opacity: 0.35;
    animation: gridMove 18s linear infinite;
    z-index: -1;
}

@keyframes gridMove {
    from { transform: translateY(0px); }
    to   { transform: translateY(45px); }
}

/* TITLE */
h1, .main-title {
    text-align: center;
    color: #00eaff !important;
    text-shadow: 0 0 12px #00eaff, 0 0 28px #00eaff;
    font-weight: 900;
    letter-spacing: 2px;
}

/* NEON CYBER CARD */
.cyber-card {
    padding: 20px;
    border-radius: 14px;
    background: rgba(15, 15, 25, 0.65);
    border: 1px solid #00eaff88;
    box-shadow:
        0 0 18px #00eaff55,
        inset 0 0 12px #00ccff33;
    backdrop-filter: blur(6px);
    margin-bottom: 22px;
}

/* PREDICTION BOX */
.pred-box {
    padding: 18px;
    border-radius: 10px;
    background: rgba(255, 0, 200, 0.18);
    border-left: 6px solid #ff00dd;
    box-shadow: 0 0 14px #ff00ff88;
    backdrop-filter: blur(4px);
    margin-bottom: 18px;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: rgba(20, 15, 30, 0.85);
    border-right: 2px solid #00eaff66;
    box-shadow: inset 0 0 20px #00eaff33;
}

/* BUTTONS */
div.stButton > button {
    background: linear-gradient(90deg, #ff00d4, #00eaff);
    color: white;
    padding: 10px 20px;
    font-weight: 700;
    border-radius: 8px;
    border: none;
    box-shadow: 0 0 15px #00eaff;
    transition: 0.25s;
}

div.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 22px #ff00d4;
}

/* METRIC NUMBERS */
[data-testid="stMetricValue"] {
    color: #00eaff !important;
    text-shadow: 0 0 6px #00eaff;
}

/* PROBABILITY TABLE */
.prob-row {
    padding: 6px 10px;
    margin-bottom: 5px;
    border-radius: 6px;
    background: rgba(0, 238, 255, 0.1);
    border-left: 4px solid #00eaff;
}

</style>
"""
st.markdown(cyberpunk_css, unsafe_allow_html=True)

model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def record_audio(duration=3, sr=22050):
    st.info("🎙 Recording…")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    st.success("Recording Finished!")

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp.name, sr, audio)
    return temp.name

st.sidebar.title("🟣 Cyberpunk Control Panel")
page = st.sidebar.radio("Navigate", ("Upload File", "Realtime Microphone Input", "About"))

def show_probability_chart(classes, probs):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(classes, probs, color="#00eaff")
    ax.set_title("Probability Distribution", color="white")
    ax.set_ylabel("Probability", color="white")
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', colors='cyan', rotation=30)
    ax.tick_params(axis='y', colors='cyan')
    fig.patch.set_facecolor("#0a0b10")
    ax.set_facecolor("#0a0b10")
    st.pyplot(fig)

if page == "Upload File":
    st.markdown("<h1 class='main-title'> VEHICLE SOUND CLASSIFIER</h1>", unsafe_allow_html=True)

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
            f"<h4 style='color:white;'>Confidence Score: {confidence:.2f}%</h4></div>",
            unsafe_allow_html=True
        )

        st.subheader("📊 Full Probability Distribution")
        for cls, pr in zip(label_encoder.classes_, probs):
            st.markdown(f"<div class='prob-row'>**{cls}** → {pr*100:.2f}%</div>", unsafe_allow_html=True)

        show_probability_chart(label_encoder.classes_, probs)

elif page == "Realtime Microphone Input":
    st.markdown("<h1 class='main-title'>🎤 REALTIME VEHICLE DETECTION</h1>", unsafe_allow_html=True)

    duration = st.slider("Durasi Rekaman (detik)", 1, 10, 3)

    if st.button("🎙 Start Recording"):
        audio_path = record_audio(duration)
        st.audio(audio_path)

        y, sr = librosa.load(audio_path, sr=22050)

        st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
        st.subheader("Waveform")
        st.pyplot(plot_waveform(y, sr))

        st.subheader("Spectrogram")
        st.pyplot(plot_spectrogram(y, sr))
        st.markdown("</div>", unsafe_allow_html=True)

        features = process_file_advanced(audio_path)
        scaled = scaler.transform([features])
        probs = model.predict_proba(scaled)[0]

        idx = np.argmax(probs)
        pred = label_encoder.classes_[idx]
        confidence = probs[idx] * 100

        st.markdown(
            f"<div class='pred-box'><h2 style='color:#ff00e6'>Prediction: {pred.upper()}</h2>"
            f"<h4 style='color:white;'>Confidence Score: {confidence:.2f}%</h4></div>",
            unsafe_allow_html=True
        )

        st.subheader("📊 Full Probability Distribution")
        for cls, pr in zip(label_encoder.classes_, probs):
            st.markdown(f"<div class='prob-row'>**{cls}** → {pr*100:.2f}%</div>", unsafe_allow_html=True)

        show_probability_chart(label_encoder.classes_, probs)

else:
    st.markdown("<h1 class='main-title'>🟣 CYBERPUNK DASHBOARD INFO</h1>", unsafe_allow_html=True)
    st.write("Final Project – Digital Signal Processing (Filtering + Feature Extraction)")
    st.write("Cyberpunk Neon UI with hologram grid background ✨")
    st.write("Features:")
    st.write("- Upload Audio Classification")
    st.write("- Realtime Microphone Vehicle Detection")
    st.write("- Waveform & Spectrogram Visualizer")
    st.write("- Confidence & Probability Distribution")
