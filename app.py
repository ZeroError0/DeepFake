import tempfile
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from src.predict import DeepfakeDetector


st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="",
    layout="centered",
)

st.title("Deepfake Audio/Video Detector")
st.markdown("Upload an audio or video file to check if it's real or AI-generated.")

# Sidebar for model loading
st.sidebar.header("Model Settings")
audio_model_path = st.sidebar.text_input("Audio model path", "checkpoints/best_audio_model.pth")
video_model_path = st.sidebar.text_input("Video model path", "checkpoints/best_video_model.pth")

# Initialize detector
@st.cache_resource
def load_detector():
    detector = DeepfakeDetector()

    if Path(audio_model_path).exists():
        detector.load_audio_model(audio_model_path)
    if Path(video_model_path).exists():
        detector.load_video_model(video_model_path)

    return detector

detector = load_detector()

# File uploader
uploaded_file = st.file_uploader(
    "Choose a file",
    type=["wav", "mp3", "flac", "mp4", "avi", "mov", "mkv"],
    help="Supported: WAV, MP3, FLAC (audio) | MP4, AVI, MOV, MKV (video)",
)

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.info(f"Analyzing: **{uploaded_file.name}**")

    try:
        with st.spinner("Running detection model..."):
            result = detector.predict(tmp_path)

        # Display results
        st.subheader("Result")

        label = result["label"]
        confidence = result["confidence"]
        probs = result["probabilities"]

        if label == "fake":
            st.error(f"**AI-Generated (Deepfake)** — {confidence*100:.1f}% confidence")
        else:
            st.success(f"**Real** — {confidence*100:.1f}% confidence")

        # Probability bars
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Real", f"{probs['real']*100:.1f}%")
            st.progress(probs["real"])
        with col2:
            st.metric("Fake", f"{probs['fake']*100:.1f}%")
            st.progress(probs["fake"])

        # Frame-level results for video
        if result.get("frame_results"):
            st.subheader("Frame-Level Analysis")
            import pandas as pd
            df = pd.DataFrame(result["frame_results"])
            st.dataframe(df, use_container_width=True)

            # Visualize frame predictions
            fake_frames = sum(1 for f in result["frame_results"] if f["prediction"] == "fake")
            total_frames = len(result["frame_results"])
            st.write(f"Fake frames: **{fake_frames}/{total_frames}**")

    except Exception as e:
        st.error(f"Error processing file: {e}")

    finally:
        # Cleanup temp file
        Path(tmp_path).unlink(missing_ok=True)

# Instructions
st.markdown("---")
st.markdown("""
### How to use
1. Train a model first (see README)
2. Place checkpoint in `checkpoints/`
3. Upload an audio or video file
4. View the real/fake prediction

### Datasets
- **Audio**: [ASVspoof](https://www.asvspoof.org/)
- **Video**: [DFDC Sample](https://www.kaggle.com/datasets) or [FaceForensics++](https://github.com/ondyari/FaceForensics)
""")
