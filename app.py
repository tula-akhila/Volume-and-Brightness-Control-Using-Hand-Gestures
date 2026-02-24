import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import tempfile
import os

st.set_page_config(page_title="Voice Emotion Analyzer")

st.title("Dashboard")
st.write("Upload a WAV file to analyze emotions")

uploaded_file = st.file_uploader(
    "Upload a WAV audio file",
    type=["wav", "webm", "ogg"]
)

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    st.audio(uploaded_file)

    try:
        # ✅ Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        # ✅ Load audio safely
        audio, sr = librosa.load(temp_path, sr=None, mono=True)

        # ✅ Remove temp file
        os.remove(temp_path)

        # Feature extraction
        rms = np.mean(librosa.feature.rms(y=audio))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))

        # Simple emotion logic
        if rms > 0.05:
            emotion = "Angry"
        elif rms > 0.03:
            emotion = "Happy"
        else:
            emotion = "Neutral"

        st.success("Audio processed successfully!")
        st.write(f"### 🎧 Detected Emotion: **{emotion}**")

    except Exception as e:
        st.error("Failed to process audio")
        st.code(str(e))