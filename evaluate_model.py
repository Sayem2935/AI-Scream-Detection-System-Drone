import os
import pickle

import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from detect_cnn_live import CustomBatchNorm, CustomDense


DATA_PATH = "processed_dataset"
MODEL_PATH = "cnn_scream_model.h5"
LABELS_PATH = "label_config.pkl"

SAMPLE_RATE = 16000
RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
VAL_SIZE_WITHIN_TRAIN = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)
DEFAULT_CLASS_NAMES = ["scream", "cough", "clap", "speech", "noise"]


def extract_spectrogram(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
        fmin=80,
        fmax=6000,
    )

    spec_db = librosa.power_to_db(spec, ref=np.max)
    spec_db = (spec_db - np.mean(spec_db)) / (np.std(spec_db) + 1e-6)
    spec_db = librosa.util.fix_length(spec_db, size=64, axis=1)
    return spec_db[:64, :64]


def load_labels():
    if not os.path.exists(LABELS_PATH):
        return DEFAULT_CLASS_NAMES

    with open(LABELS_PATH, "rb") as file_obj:
        config = pickle.load(file_obj)
    return config.get("class_names", DEFAULT_CLASS_NAMES)


def load_dataset(class_names):
    X = []
    y = []

    print("Loading dataset...")

    for class_index, class_name in enumerate(class_names):
        folder = os.path.join(DATA_PATH, class_name)
        if not os.path.isdir(folder):
            print(f"Skipping missing folder: {folder}")
            continue

        for root, _, files in os.walk(folder):
            for file_name in files:
                if not file_name.endswith(".wav"):
                    continue

                file_path = os.path.join(root, file_name)
                try:
                    X.append(extract_spectrogram(file_path))
                    y.append(class_index)
                except Exception as exc:
                    print(f"Failed to process {file_path}: {exc}")

    X = np.array(X, dtype=np.float32)[..., np.newaxis]
    y = np.array(y, dtype=np.int32)

    print(f"Data loaded: {X.shape}")
    for class_index, class_name in enumerate(class_names):
        print(f"{class_name}: {int(np.sum(y == class_index))}")
    return X, y


def split_dataset(X, y):
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VAL_SIZE_WITHIN_TRAIN,
        stratify=y_train_full,
        random_state=RANDOM_STATE,
    )

    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def plot_confusion_matrix(y_true, y_pred, class_names):
    fig, ax = plt.subplots(figsize=(9, 7))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred),
        display_labels=class_names,
    )
    disp.plot(ax=ax, xticks_rotation=30, colorbar=False)
    ax.set_title("Multi-Class Confusion Matrix")
    plt.tight_layout()
    plt.show()


def main():
    class_names = load_labels()
    X, y = load_dataset(class_names)
    _, _, X_test, _, _, y_test = split_dataset(X, y)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "BatchNormalization": CustomBatchNorm,
            "Dense": CustomDense,
        },
        compile=False,
    )

    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    accuracy = float(np.mean(y_pred == y_test))
    print(f"\nOverall accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred, class_names)


if __name__ == "__main__":
    main()
