import sounddevice as sd
import numpy as np
import librosa
import joblib

MODEL_PATH = "scream_model.pkl"

model = joblib.load(MODEL_PATH)

SAMPLE_RATE = 16000
DURATION = 1

def extract_features(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
    return np.mean(mfcc.T, axis=0).reshape(1, -1)

def audio_callback(indata, frames, time, status):
    audio = indata[:, 0]

    features = extract_features(audio)
    prediction = model.predict(features)[0]

    if prediction == 1:
        print("🚨 SCREAM DETECTED!")
    else:
        print("Normal")

print("🎤 Listening...")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    callback=audio_callback,
    blocksize=SAMPLE_RATE
):
    input("Press Enter to stop...\n")