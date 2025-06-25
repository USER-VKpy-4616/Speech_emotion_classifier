# Speech Emotion Classification
The objective of this project is to design and implement an end-to-end pipeline for emotion classification using speech data. The system leverages audio processing techniques and deep learning models to accurately identify and categorize emotional states conveyed in speech or song.

## Repository Contents
1. app.py
   
Purpose: Streamlit-based web application for real-time emotion classification.

Details:
- Loads a trained CNN model (emotion_cnn_mel.keras) and label encoder (label_encoder.pkl)
- Allows the user to upload a .wav file via the UI
- Extracts log-Mel spectrograms from the audio using librosa
- Runs inference and displays the predicted emotion

2. aud.ipynb

Purpose: Jupyter Notebook containing the entire training and evaluation pipeline.

Contents:
- Data loading and audio preprocessing (using Mel spectrograms)
- Label encoding
- Model architecture definition (CNN)
- Model training and validation
- Evaluation metrics (accuracy, F1 score, confusion matrix)

3. emotion_cnn_mel.keras

Purpose: Trained deep learning model file saved in Keras HDF5 format.

Details:
- CNN-based model trained on speech emotion dataset
- Takes 128x128x1 log-Mel spectrograms as input
- Outputs emotion predictions via softmax layer
- Used by both app.py and predictor.py for inference.

4. label_encoder.pkl
Purpose: Serialized LabelEncoder object used to map numerical class indices back to emotion labels.

Details:
- Created during training (in aud.ipynb)
- Required for decoding prediction outputs from the model
- Load with joblib.load() for fast and reliable use in both app.py and predictor.py.

5. predictor.py

Purpose: Standalone Python script for testing the trained model on audio files without using the web app.

Likely includes:
- Command-line or function-based input for test .wav files
- Audio preprocessing logic (same as in app.py)
- Loads model and label encoder
- Returns the predicted emotion.

6. requirements.txt
Purpose: Lists all required Python packages and their versions.

7. runtime.txt
Purpose: Specifies the Python runtime version (especially for Streamlit Cloud or Heroku deployments).

## Model Experimentation

An initial version of the model was developed using MFCC features combined with a simple feedforward neural network:

- Audio Preprocessing:
Audio was loaded at a sampling rate of 22,050 Hz and trimmed or padded to 3 seconds. From each audio clip, 40 MFCC coefficients were extracted and averaged over time to produce fixed-size input vectors.

- Model Architecture:
A basic dense neural network with two fully connected layers and dropout regularization was used. The final layer used softmax activation for multi-class emotion classification.

- Training and Evaluation:
The model was trained using categorical cross-entropy loss and the Adam optimizer. Despite being computationally efficient, this approach did not meet the required accuracy or F1 score thresholds and was not used in the final deployment.

This experimental setup was later replaced by a convolutional neural network using log-Mel spectrograms, which provided significantly better performance.

### Final Model Overview
Preprocessing
Audio clips (3 seconds, 22050 Hz) are converted into 128×128 log-Mel spectrograms.

All spectrograms are padded/truncated to ensure a fixed shape and reshaped to include a channel dimension (128, 128, 1).

Model Architecture
A Convolutional Neural Network (CNN) was used, consisting of:

- 3 convolutional blocks with Batch Normalization, MaxPooling, and Dropout
- A dense layer with 256 units followed by dropout
- Softmax output for emotion classification

Training
- Optimizer: Adam (lr=0.0005)
- Loss: Categorical crossentropy
- Early stopping and learning rate reduction callbacks were used

Evaluation
- Final Accuracy: ~77.26%
- Macro F1 Score: ~76.88%

This model was used in the final deployment.
