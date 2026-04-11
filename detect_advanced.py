import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import time
import soundfile as sf
from scipy.signal import butter, lfilter

# ==========================
# CONFIG
# ==========================
MODEL_PATH = "cnn_scream_model.h5"
SAMPLE_RATE = 16000

THRESHOLD = 0.30
HISTORY_SIZE = 8
MOMENTUM_LIMIT = 4
COOLDOWN_TIME = 3   # seconds

# ==========================
# LOAD MODEL
# ==========================
model = tf.keras.models.load_model(MODEL_PATH)

# ==========================
# HIGH-PASS FILTER
# ==========================
def highpass(audio, cutoff=500):
    b, a = butter(1, cutoff / (0.5 * SAMPLE_RATE), btype='high')
    return lfilter(b, a, audio)

# ==========================
# ENERGY CHECK
# ==========================
def is_loud(audio):
    return np.mean(audio**2) > 0.003

# ==========================
# FEATURE EXTRACTION
# ==========================
def extract_spec(audio):
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=64,
        fmax=8000
    )

    spec_db = librosa.power_to_db(spec, ref=np.max)
    spec_db = (spec_db - np.mean(spec_db)) / (np.std(spec_db) + 1e-6)

    spec_db = librosa.util.fix_length(spec_db, size=64, axis=1)

    return spec_db[:64, :64][..., np.newaxis]

# ==========================
# STATE
# ==========================
history = []
momentum = 0
last_trigger_time = 0

# ==========================
# CALLBACK
# ==========================
def audio_callback(indata, frames, time_info, status):
    global history, momentum, last_trigger_time

    audio = indata[:, 0]

    # Energy gate
    if not is_loud(audio):
        print("🔇 Quiet")
        return

    # Remove low-frequency noise
    audio = highpass(audio)

    # Extract features
    features = extract_spec(audio)
    features = np.expand_dims(features, axis=0)

    # Predict
    proba = model.predict(features, verbose=0)[0][0]

    # Smooth prediction
    history.append(proba)
    if len(history) > HISTORY_SIZE:
        history.pop(0)

    avg = sum(history) / len(history)

    # Momentum logic
    if avg > THRESHOLD:
        momentum += 1
    else:
        momentum = max(0, momentum - 1)

    # Alert levels
    if avg > 0.5:
        level = "🔥 HIGH SCREAM"
    elif avg > 0.35:
        level = "⚠️ POSSIBLE SCREAM"
    else:
        level = "Normal"

    print(f"Confidence: {avg:.2f} | Momentum: {momentum} → {level}")

    # Trigger detection
    if momentum >= MOMENTUM_LIMIT:
        current_time = time.time()

        if current_time - last_trigger_time > COOLDOWN_TIME:
            print("🚨🚨 SCREAM CONFIRMED 🚨🚨")

            # Save audio snapshot
            filename = f"alert_{int(current_time)}.wav"
            sf.write(filename, audio, SAMPLE_RATE)

            print(f"📁 Saved: {filename}")

            last_trigger_time = current_time

        momentum = 0

# ==========================
# RUN
# ==========================
print("🎤 Advanced Detection Running...")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    callback=audio_callback,
    blocksize=int(SAMPLE_RATE * 0.5)  # 0.5 sec sliding window
):
    input("Press Enter to stop...\n")