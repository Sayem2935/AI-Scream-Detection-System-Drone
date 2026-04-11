import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = "processed_dataset"

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

X = []
y = []

print(" Extracting features...")

for label, value in [("scream", 1), ("non_scream", 0)]:
    folder = os.path.join(DATA_PATH, label)
    
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)
            features = extract_features(path)
            X.append(features)
            y.append(value)

X = np.array(X)
y = np.array(y)

print(" Feature extraction done")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("🚀 Training model...")

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\n Accuracy:", accuracy_score(y_test, y_pred))
print("\n Report:\n", classification_report(y_test, y_pred))

import joblib
joblib.dump(model, "scream_model.pkl")
print(" Model saved!")