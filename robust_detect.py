import sounddevice as sd
import numpy as np
import librosa
import joblib


MODEL_PATH = "scream_model.pkl"
SAMPLE_RATE = 16000

THRESHOLD = 0.6        
HISTORY_SIZE = 5       


model = joblib.load(MODEL_PATH)


def extract_features(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
    return np.mean(mfcc.T, axis=0).reshape(1, -1)


history = []


def audio_callback(indata, frames, time, status):
    global history


    audio = indata[:, 0]

    
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    
    features = extract_features(audio)

    
    proba = model.predict_proba(features)[0][1]

   
    history.append(proba)
    if len(history) > HISTORY_SIZE:
        history.pop(0)

    avg_proba = sum(history) / len(history)

    
    print(f"Confidence: {avg_proba:.2f}", end=" → ")

    
    if avg_proba > THRESHOLD:
        print("🚨 SCREAM DETECTED!")
    else:
        print("Normal")


print("🎤 Listening... (Press Enter to stop)")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    callback=audio_callback,
    blocksize=SAMPLE_RATE
):
    input()