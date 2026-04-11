import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


DATA_PATH = "processed_dataset"
MODEL_PATH = "cnn_scream_model.h5"
IMG_SIZE = (64, 64)


def extract_spectrogram(file):
    audio, sr = librosa.load(file, sr=16000)

    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=64,
        fmax=8000
    )

    spec_db = librosa.power_to_db(spec, ref=np.max)
    spec_db = (spec_db - np.mean(spec_db)) / (np.std(spec_db) + 1e-6)

    spec_db = librosa.util.fix_length(spec_db, size=64, axis=1)

    return spec_db[:64, :64]


X = []
y = []

print("📂 Loading dataset...")

for label, value in [("scream", 1), ("non_scream", 0)]:
    folder = os.path.join(DATA_PATH, label)

    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)

            try:
                features = extract_spectrogram(path)
                X.append(features)
                y.append(value)
            except:
                continue

X = np.array(X)
y = np.array(y)

X = X[..., np.newaxis]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)


model = tf.keras.models.load_model(MODEL_PATH)


y_pred = model.predict(X_test)
y_pred = (y_pred > 0.3).astype(int)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🧠 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


print("\n⚠️ Note: Training curves only available during training.")