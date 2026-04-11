import os
import numpy as np
import librosa
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping

# ==========================
# CONFIG
# ==========================
DATA_PATH = "processed_dataset"
MODEL_PATH = "cnn_scream_model.h5"

# ==========================
# FEATURE EXTRACTION
# ==========================
def extract_spectrogram(file):
    audio, sr = librosa.load(file, sr=16000)

    # normalize audio
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    # mel spectrogram
    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=64,
        fmax=8000
    )

    spec_db = librosa.power_to_db(spec, ref=np.max)

    # normalize spectrogram
    spec_db = (spec_db - np.mean(spec_db)) / (np.std(spec_db) + 1e-6)

    # fix size
    spec_db = librosa.util.fix_length(spec_db, size=64, axis=1)

    return spec_db[:64, :64]

# ==========================
# LOAD DATA
# ==========================
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

# add channel dimension
X = X[..., np.newaxis]

print("✅ Data loaded:", X.shape)

# ==========================
# SPLIT DATA
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# MODEL
# ==========================
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),   # 🔥 increased to reduce overfitting
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================
# EARLY STOPPING (KEY FIX)
# ==========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ==========================
# TRAIN MODEL
# ==========================
print("🚀 Training model...")

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# ==========================
# SAVE MODEL
# ==========================
model.save(MODEL_PATH)
print("✅ Model saved!")

# ==========================
# SAVE HISTORY (FOR GRAPH)
# ==========================
with open("history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("📊 Training history saved!")

# ==========================
# EVALUATION
# ==========================
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.3).astype(int)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🧠 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))