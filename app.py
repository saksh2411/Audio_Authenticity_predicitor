import os
import hashlib
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import streamlit as st

from utils.predict import predict_file

APP_TITLE = "Voice Authenticity Checker"
TARGET_SR = 16000
DURATION_SEC = 4


def file_signature(file_obj):
    data = file_obj.getvalue()
    return hashlib.md5(data).hexdigest()


def standardize_audio_to_wav(audio_file, out_sr=TARGET_SR, duration_sec=DURATION_SEC):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as in_tmp:
        in_tmp.write(audio_file.getbuffer())
        in_path = in_tmp.name

    y, sr = librosa.load(in_path, sr=out_sr, mono=True)
    y = librosa.util.normalize(y)

    target_len = out_sr * duration_sec
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    out_tmp.close()
    sf.write(out_tmp.name, y, out_sr)

    try:
        os.remove(in_path)
    except Exception:
        pass

    return out_tmp.name


def cleanup_temp_file(path):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
:root {
    --bg: #08111f;
    --panel: #0f172a;
    --panel2: #142033;
    --border: #25344d;
    --text: #e8eef7;
    --muted: #9db0c9;
    --accent: #38bdf8;
    --good: #22c55e;
    --bad: #f43f5e;
    --warn: #f59e0b;
}
.stApp {
    background: linear-gradient(180deg, #07101d 0%, #0b1220 100%);
    color: var(--text);
}
.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
[data-testid="stSidebar"] {
    background: #0e1729;
    border-right: 1px solid var(--border);
}
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #12203a 50%, #10243f 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.4rem 1.4rem 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.hero-title {
    color: #ffffff;
    font-size: 2.1rem;
    font-weight: 800;
    margin-bottom: 0.35rem;
}
.hero-sub {
    color: var(--muted);
    line-height: 1.6;
}
.section-card {
    background: rgba(15, 23, 42, 0.92);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.metric-card {
    background: linear-gradient(180deg, #101827 0%, #121d31 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1rem;
    min-height: 118px;
}
.metric-label {
    color: var(--muted);
    font-size: 0.92rem;
    margin-bottom: 0.35rem;
}
.metric-value {
    color: white;
    font-size: 1.8rem;
    font-weight: 800;
}
.metric-note {
    color: #8ca1ba;
    font-size: 0.84rem;
    margin-top: 0.25rem;
}
.real-box {
    background: rgba(34, 197, 94, 0.14);
    border: 1px solid rgba(34, 197, 94, 0.35);
    color: #bbf7d0;
    padding: 1rem 1.1rem;
    border-radius: 16px;
    font-weight: 700;
}
.fake-box {
    background: rgba(244, 63, 94, 0.14);
    border: 1px solid rgba(244, 63, 94, 0.35);
    color: #fecdd3;
    padding: 1rem 1.1rem;
    border-radius: 16px;
    font-weight: 700;
}
.warn-box {
    background: rgba(245, 158, 11, 0.14);
    border: 1px solid rgba(245, 158, 11, 0.35);
    color: #fde68a;
    padding: 1rem 1.1rem;
    border-radius: 16px;
    font-weight: 700;
}
.info-box {
    background: rgba(17, 24, 39, 0.9);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1rem 1.1rem;
    color: var(--muted);
    line-height: 1.7;
}
.footer-box {
    margin-top: 1.25rem;
    border-top: 1px solid var(--border);
    padding-top: 1rem;
    text-align: center;
    color: var(--muted);
    font-size: 0.9rem;
}
.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 3rem;
    border: 1px solid #1d4ed8;
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
    color: white;
    font-weight: 700;
}
.stButton > button:hover {
    border: 1px solid #3b82f6;
    background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
    color: white;
}
[data-baseweb="tab"] {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

default_state = {
    "result": None,
    "audio": None,
    "sr": None,
    "temp_path": None,
    "processed_signature": None,
    "source_name": None
}
for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

with st.sidebar:
    st.markdown("## Control Panel")
    debug_mode = st.checkbox("Show debug information", value=True)
    st.markdown("""
**Pipeline**
- Standardized audio input
- Spectral analysis branch
- Handcrafted voice-feature branch
- Fusion-based final decision
""")
    st.markdown("""
**Accepted input**
- WAV
- MP3
- M4A
- Microphone recording
""")
    st.markdown("""
**Result logic**
- Real: low fake score
- Fake: high fake score
- Suspicious: borderline score
""")

st.markdown(f"""
<div class="hero">
    <div class="hero-title">{APP_TITLE}</div>
    <div class="hero-sub">
        Analyze an uploaded or recorded voice sample using fused spectral and voice-pattern evidence.
        This version uses stable rerun handling and aligned preprocessing for both input methods.
    </div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1.1], gap="large")

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Provide Voice Sample")

    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=["wav", "mp3", "m4a"],
        key="audio_uploader"
    )

    recorded_audio = st.audio_input(
        "Or record a voice sample",
        sample_rate=TARGET_SR
    )

    selected_file = None
    selected_name = None

    if uploaded_file is not None:
        selected_file = uploaded_file
        selected_name = uploaded_file.name
        st.markdown("#### Uploaded Audio")
        st.audio(uploaded_file)

    elif recorded_audio is not None:
        selected_file = recorded_audio
        selected_name = "recorded_audio.wav"
        st.markdown("#### Recorded Audio")
        st.audio(recorded_audio)

    analyze_btn = st.button("Check Voice Authenticity", use_container_width=True)
    clear_btn = st.button("Clear Current Result", use_container_width=True)

    if clear_btn:
        cleanup_temp_file(st.session_state.temp_path)
        st.session_state.result = None
        st.session_state.audio = None
        st.session_state.sr = None
        st.session_state.temp_path = None
        st.session_state.processed_signature = None
        st.session_state.source_name = None
        st.rerun()

    if analyze_btn:
        if selected_file is None:
            st.warning("Please upload or record an audio sample first.")
        else:
            current_signature = file_signature(selected_file)

            if st.session_state.processed_signature == current_signature and st.session_state.result is not None:
                st.info("This audio was already analyzed. Reusing the latest result.")
            else:
                cleanup_temp_file(st.session_state.temp_path)

                try:
                    temp_path = standardize_audio_to_wav(
                        selected_file,
                        out_sr=TARGET_SR,
                        duration_sec=DURATION_SEC
                    )

                    with st.spinner("Analyzing voice sample..."):
                        result, audio, sr = predict_file(temp_path)

                    st.session_state.result = result
                    st.session_state.audio = audio
                    st.session_state.sr = sr
                    st.session_state.temp_path = temp_path
                    st.session_state.processed_signature = current_signature
                    st.session_state.source_name = selected_name

                    st.success(f"Analysis completed for: {selected_name}")
                except Exception as e:
                    st.error(f"Audio processing failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Overview")
    st.markdown("""
<div class="info-box">
This app uses a fusion pipeline. One model reads normalized mel-spectrogram structure, while another
model reads handcrafted speech features such as MFCC behavior, silence ratio, pitch statistics,
spectral shape, and signal distribution. The final score combines both branches into one decision.
</div>
""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.result is not None:
    result = st.session_state.result
    audio = st.session_state.audio
    sr = st.session_state.sr

    st.markdown("## Analysis Results")

    duration = len(audio) / sr if sr else 0.0
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">CNN Score</div>
            <div class="metric-value">{result['cnn_prob']:.4f}</div>
            <div class="metric-note">Spectral branch fake probability</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">RF Score</div>
            <div class="metric-value">{result['rf_prob']:.4f}</div>
            <div class="metric-note">Voice-feature branch fake probability</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Fused Score</div>
            <div class="metric-value">{result['fused_prob']:.4f}</div>
            <div class="metric-note">Combined fake probability</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Duration</div>
            <div class="metric-value">{duration:.2f}s</div>
            <div class="metric-note">Standardized analysis length</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Authenticity Result")
    if result["final_label"] == "Fake":
        st.markdown(
            '<div class="fake-box">Result: AI-generated or manipulated voice likely detected.</div>',
            unsafe_allow_html=True
        )
    elif result["final_label"] == "Real":
        st.markdown(
            '<div class="real-box">Result: Voice sample appears authentic.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="warn-box">Result: Borderline output. Manual review is recommended.</div>',
            unsafe_allow_html=True
        )

    conf = float(min(max(result["fused_prob"], 0.0), 1.0))
    st.markdown("### Confidence Meter")
    st.progress(conf)
    st.caption(f"Displayed fused fake probability: {conf:.2%}")

    tabs = st.tabs(["Spectrogram", "Waveform", "Voice Features", "Debug"])

    with tabs[0]:
        fig, ax = plt.subplots(figsize=(10, 4))
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
        ax.set_title("Mel Spectrogram")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        st.pyplot(fig)

    with tabs[1]:
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(audio, sr=sr, ax=ax2)
        ax2.set_title("Waveform")
        st.pyplot(fig2)

    with tabs[2]:
        feature_df = pd.DataFrame([result["features"]])
        st.dataframe(feature_df, use_container_width=True)

    with tabs[3]:
        if debug_mode:
            st.write(f"Source: {st.session_state.source_name}")
            st.write(f"CNN fake probability: {result['cnn_prob']:.6f}")
            st.write(f"RF fake probability: {result['rf_prob']:.6f}")
            st.write(f"Fused fake probability: {result['fused_prob']:.6f}")
            st.write(f"Final label: {result['final_label']}")
            st.write("Threshold policy: Real <= 0.48, Fake >= 0.52, otherwise Suspicious")
        else:
            st.info("Enable debug mode from the sidebar to see internal scores.")

st.markdown("""
<div class="footer-box">
Voice Authenticity Checker
</div>
""", unsafe_allow_html=True)