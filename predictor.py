import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import numpy as np
import librosa
import tensorflow as tf
import joblib

# -------------------- Constants --------------------
SAMPLE_RATE = 22050
DURATION = 3
N_MELS = 128
FIXED_FRAMES = 128

# -------------------- Load Model + Encoder --------------------
model = tf.keras.models.load_model("emotion_cnn_mel.keras")  # or .h5 if you saved in legacy format
le = joblib.load("label_encoder.pkl")

# -------------------- Feature Extraction --------------------
def extract_log_mel(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, res_type='kaiser_fast')

    if len(audio) < SAMPLE_RATE * DURATION:
        pad = SAMPLE_RATE * DURATION - len(audio)
        audio = np.pad(audio, (0, pad))

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, hop_length=512)
    log_mel = librosa.power_to_db(mel)

    if log_mel.shape[1] < FIXED_FRAMES:
        pad = FIXED_FRAMES - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)))
    else:
        log_mel = log_mel[:, :FIXED_FRAMES]

    return log_mel[..., np.newaxis]  # shape: (128, 128, 1)

# -------------------- Prediction Function --------------------
def predict_emotion(file_path):
    features = extract_log_mel(file_path)
    features = np.expand_dims(features, axis=0)  # (1, 128, 128, 1)
    probs = model.predict(features)[0]
    predicted_index = np.argmax(probs)
    predicted_label = le.inverse_transform([predicted_index])[0]
    confidence = probs[predicted_index]
    return predicted_label, confidence

# -------------------- Main --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion Prediction on Audio File(s)")
    parser.add_argument("--file", type=str, help="Path to a .wav file", default=None)
    parser.add_argument("--folder", type=str, help="Path to folder containing .wav files", default=None)

    args = parser.parse_args()

    if args.file:
        pred, conf = predict_emotion(args.file)
        print(f"[{os.path.basename(args.file)}] → Emotion: {pred} (Confidence: {conf:.2f})")

    elif args.folder:
        for fname in os.listdir(args.folder):
            if fname.endswith(".wav"):
                fpath = os.path.join(args.folder, fname)
                pred, conf = predict_emotion(fpath)
                print(f"[{fname}] → Emotion: {pred} (Confidence: {conf:.2f})")
    else:
        print("Please provide either --file or --folder")
