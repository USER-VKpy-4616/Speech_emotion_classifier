import os
import librosa
import numpy as np
import pandas as pd

audio_files = []
root_dir = "Audio_Song_Actors"  # or "Audio_Speech_Actors"

for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".wav"):
            audio_files.append(os.path.join(root, file))

print(audio_files) 

file_path = audio_files[0]  # Example: first file in the list
y, sr = librosa.load(file_path, sr=None)  # y: audio time series, sr: sampling rate

print(f"Loaded {file_path} with sample rate {sr}")