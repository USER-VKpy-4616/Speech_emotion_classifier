import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import joblib
import tempfile
import os

import gdown
"""
MODEL_PATH = "emotion_cnn_mel.keras"

with st.spinner("Downloading model..."):
    gdown.download("https://drive.google.com/uc?id=1onb8BIr7D6-b6Pnyp6rDn_JXcJOmBe5W", MODEL_PATH, quiet=False)
"""
import urllib.request

MODEL_URL = "https://huggingface.co/KV4661/Emotional_classifier/resolve/main/emotional_cnn_mel.keras"
LABEL_URL = "https://huggingface.co/KV4661/Emotional_classifier/resolve/main/label_encoder.pkl"

MODEL_PATH = "emotion_cnn_mel.keras"
LABEL_PATH = "label_encoder.pkl"

urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
urllib.request.urlretrieve(LABEL_URL, LABEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)
le = joblib.load(LABEL_PATH)

SAMPLE_RATE = 22050
DURATION = 3
N_MELS = 128
FIXED_FRAMES = 128

# Extract features from uploaded audio
def extract_log_mel_from_file(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, res_type='kaiser_fast')
    
    if len(audio) < SAMPLE_RATE * DURATION:
        pad_width = SAMPLE_RATE * DURATION - len(audio)
        audio = np.pad(audio, (0, pad_width))

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, hop_length=512)
    log_mel = librosa.power_to_db(mel)

    if log_mel.shape[1] < FIXED_FRAMES:
        pad_width = FIXED_FRAMES - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)))
    else:
        log_mel = log_mel[:, :FIXED_FRAMES]

    return log_mel[..., np.newaxis]  

# Streamlit UI
st.title("Speech Emotion Classifier")

uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"])

if uploaded_file is not None:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.audio(uploaded_file, format="audio/wav")

    # Feature extraction
    features = extract_log_mel_from_file(tmp_path)
    features = np.expand_dims(features, axis=0)  # Add batch dim

    # Predict
    prediction = model.predict(features)
    predicted_label = le.inverse_transform([np.argmax(prediction)])

    st.markdown(f"###  Predicted Emotion: **{predicted_label[0].capitalize()}**")

    os.remove(tmp_path)
