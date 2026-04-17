import os
import pickle

import librosa
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


DATA_PATH = "processed_dataset"
MODEL_PATH = "cnn_scream_model.h5"
HISTORY_PATH = "history.pkl"
THRESHOLD_PATH = "threshold_config.pkl"
LABELS_PATH = "label_config.pkl"

CLASS_CANDIDATES = ["scream", "cough", "clap", "speech", "noise", "non_scream"]

SAMPLE_RATE = 16000
INPUT_SHAPE = (64, 64, 1)
RANDOM_STATE = 42
RNG = np.random.default_rng(RANDOM_STATE)

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
VAL_SIZE_WITHIN_TRAIN = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)

L2_WEIGHT = 1e-4
PATIENCE = 8
EPOCHS = 50
BATCH_SIZE = 24
THRESHOLD_GRID = np.arange(0.10, 0.61, 0.02)
THRESHOLD_SELECTION_BETA = 2.0
MIN_VALIDATION_PRECISION = 0.60
MAX_NON_SCREAM_RATIO = 1.5

AUGMENT_TRAINING = True
AUGMENT_COPIES = {
    "scream": 3,
    "cough": 2,
    "clap": 2,
    "speech": 1,
    "noise": 1,
    "non_scream": 1,
}
NOISE_STD_RANGE = (0.002, 0.008)
PITCH_SHIFT_RANGE = (-1.0, 1.0)
TIME_SHIFT_RATIO = 0.10


def load_audio(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def augment_audio(audio):
    augmented = np.copy(audio)
    operations = ["noise", "pitch", "shift"]
    num_operations = int(RNG.integers(1, 3))
    selected_operations = RNG.choice(operations, size=num_operations, replace=False)

    for operation in selected_operations:
        if operation == "noise":
            noise_std = float(RNG.uniform(*NOISE_STD_RANGE))
            noise = RNG.normal(0.0, noise_std, size=augmented.shape).astype(np.float32)
            augmented = augmented + noise

        elif operation == "pitch":
            steps = float(RNG.uniform(*PITCH_SHIFT_RANGE))
            augmented = librosa.effects.pitch_shift(augmented, sr=SAMPLE_RATE, n_steps=steps)

        elif operation == "shift":
            max_shift = int(len(augmented) * TIME_SHIFT_RATIO)
            shift_amount = int(RNG.integers(-max_shift, max_shift + 1))
            augmented = np.roll(augmented, shift_amount)
            if shift_amount > 0:
                augmented[:shift_amount] = 0.0
            elif shift_amount < 0:
                augmented[shift_amount:] = 0.0

    augmented = augmented / (np.max(np.abs(augmented)) + 1e-6)
    return augmented.astype(np.float32)


def audio_to_spectrogram(audio):
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
        fmin=80,
        fmax=6000,
    )

    spec_db = librosa.power_to_db(spec, ref=np.max)
    spec_db = (spec_db - np.mean(spec_db)) / (np.std(spec_db) + 1e-6)
    spec_db = librosa.util.fix_length(spec_db, size=64, axis=1)

    return spec_db[:64, :64].astype(np.float32)


def discover_class_names():
    class_names = []
    for class_name in CLASS_CANDIDATES:
        folder = os.path.join(DATA_PATH, class_name)
        if os.path.isdir(folder):
            class_names.append(class_name)
    return class_names


def collect_dataset_index(class_names, class_to_index):
    file_paths = []
    labels = []

    print("Loading dataset index...")

    for class_name in class_names:
        folder = os.path.join(DATA_PATH, class_name)
        if not os.path.isdir(folder):
            print(f"Skipping missing folder: {folder}")
            continue

        for root, _, files in os.walk(folder):
            for file_name in files:
                if not file_name.endswith(".wav"):
                    continue
                file_paths.append(os.path.join(root, file_name))
                labels.append(class_to_index[class_name])

    file_paths = np.array(file_paths)
    labels = np.array(labels, dtype=np.int32)
    sort_order = np.argsort(file_paths)
    file_paths = file_paths[sort_order]
    labels = labels[sort_order]

    print(f"Indexed {len(file_paths)} files")
    for class_name, class_index in class_to_index.items():
        print(f"{class_name}: {int(np.sum(labels == class_index))} samples")
    return file_paths, labels


def validate_dataset(labels, class_names):
    if "scream" not in class_names:
        raise ValueError(
            "Training requires a scream class folder. "
            "Expected at least processed_dataset/scream."
        )

    present_indices = sorted(np.unique(labels).tolist())
    present_classes = [class_names[index] for index in present_indices]

    if len(present_indices) < 2:
        raise ValueError(
            "Training data contains fewer than 2 classes after loading. "
            "Check processed_dataset/ and rebuild the dataset before training."
        )

    print(f"Training with available classes: {', '.join(present_classes)}")


def rebalance_file_paths(file_paths, labels, class_names):
    if "scream" not in class_names or "non_scream" not in class_names:
        return file_paths, labels

    scream_index = class_names.index("scream")
    non_scream_index = class_names.index("non_scream")

    scream_mask = labels == scream_index
    non_scream_mask = labels == non_scream_index

    scream_count = int(np.sum(scream_mask))
    non_scream_count = int(np.sum(non_scream_mask))

    if scream_count == 0 or non_scream_count == 0:
        return file_paths, labels

    allowed_non_scream = int(np.floor(scream_count * MAX_NON_SCREAM_RATIO))
    if non_scream_count <= allowed_non_scream:
        print(
            f"Class ratio already acceptable: scream={scream_count}, "
            f"non_scream={non_scream_count}"
        )
        return file_paths, labels

    scream_indices = np.where(scream_mask)[0]
    non_scream_indices = np.where(non_scream_mask)[0]
    kept_non_scream_indices = RNG.choice(non_scream_indices, size=allowed_non_scream, replace=False)

    other_indices = np.where(~(scream_mask | non_scream_mask))[0]
    selected_indices = np.concatenate([scream_indices, kept_non_scream_indices, other_indices])
    selected_indices.sort()

    balanced_file_paths = file_paths[selected_indices]
    balanced_labels = labels[selected_indices]

    print(
        "Applied non_scream downsampling:"
        f" scream={scream_count}, non_scream={non_scream_count} -> {allowed_non_scream}"
    )
    return balanced_file_paths, balanced_labels


def build_feature_dataset(file_paths, labels, class_names, augment=False):
    X = []
    y = []

    for file_path, label in zip(file_paths, labels):
        try:
            audio = load_audio(file_path)
            X.append(audio_to_spectrogram(audio))
            y.append(label)

            if augment:
                class_name = class_names[int(label)]
                augment_copies = AUGMENT_COPIES.get(class_name, 1)
                for _ in range(augment_copies):
                    augmented_audio = augment_audio(audio)
                    X.append(audio_to_spectrogram(augmented_audio))
                    y.append(label)
        except Exception as exc:
            print(f"Failed to process {file_path}: {exc}")

    X = np.array(X, dtype=np.float32)[..., np.newaxis]
    y = np.array(y, dtype=np.int32)
    return X, y


def build_model(num_classes):
    l2_reg = regularizers.l2(L2_WEIGHT)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=INPUT_SHAPE),

        tf.keras.layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2_reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2_reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.15),

        tf.keras.layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2_reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2_reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.20),

        tf.keras.layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2_reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2_reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, kernel_regularizer=l2_reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_accuracy"),
        ],
    )
    return model


def compute_training_class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def find_best_scream_threshold(y_true, y_prob_matrix, scream_class_index):
    y_true_binary = (y_true == scream_class_index).astype(int)
    scream_prob = y_prob_matrix[:, scream_class_index]
    rows = []
    beta_sq = THRESHOLD_SELECTION_BETA ** 2

    for threshold in THRESHOLD_GRID:
        y_pred = (scream_prob >= threshold).astype(int)
        precision = precision_score(y_true_binary, y_pred, zero_division=0)
        recall = recall_score(y_true_binary, y_pred, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred, zero_division=0)
        if precision == 0.0 and recall == 0.0:
            f_beta = 0.0
        else:
            f_beta = (1 + beta_sq) * precision * recall / ((beta_sq * precision) + recall + 1e-8)
        rows.append({
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "f_beta": float(f_beta),
        })

    eligible_rows = [row for row in rows if row["precision"] >= MIN_VALIDATION_PRECISION]
    candidate_rows = eligible_rows or rows
    best_row = max(
        candidate_rows,
        key=lambda row: (row["f_beta"], row["recall"], row["f1"], -abs(row["threshold"] - 0.30)),
    )
    return best_row, rows


def print_threshold_table(rows):
    print("\nValidation scream-threshold sweep:")
    print("threshold  precision  recall  f1      f2")
    for row in rows:
        print(
            f"{row['threshold']:.2f}      "
            f"{row['precision']:.4f}     "
            f"{row['recall']:.4f}  "
            f"{row['f1']:.4f}  "
            f"{row['f_beta']:.4f}"
        )


CLASS_NAMES = discover_class_names()
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
SCREAM_CLASS_INDEX = CLASS_TO_INDEX.get("scream", 0)

file_paths, labels = collect_dataset_index(CLASS_NAMES, CLASS_TO_INDEX)
validate_dataset(labels, CLASS_NAMES)
file_paths, labels = rebalance_file_paths(file_paths, labels, CLASS_NAMES)

train_paths_full, test_paths, y_train_full, y_test = train_test_split(
    file_paths,
    labels,
    test_size=TEST_SIZE,
    stratify=labels,
    random_state=RANDOM_STATE,
)

train_paths, val_paths, y_train, y_val = train_test_split(
    train_paths_full,
    y_train_full,
    test_size=VAL_SIZE_WITHIN_TRAIN,
    stratify=y_train_full,
    random_state=RANDOM_STATE,
)

print(f"Train files: {len(train_paths)} | Val files: {len(val_paths)} | Test files: {len(test_paths)}")

print("Building training features...")
X_train, y_train = build_feature_dataset(train_paths, y_train, CLASS_NAMES, augment=AUGMENT_TRAINING)

print("Building validation features...")
X_val, y_val = build_feature_dataset(val_paths, y_val, CLASS_NAMES, augment=False)

print("Building test features...")
X_test, y_test = build_feature_dataset(test_paths, y_test, CLASS_NAMES, augment=False)

print(f"Training tensors: {X_train.shape} | Validation tensors: {X_val.shape} | Test tensors: {X_test.shape}")

class_weight_map = compute_training_class_weights(y_train)
print(f"Class weights (train only): {class_weight_map}")

model = build_model(len(CLASS_NAMES))
model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=PATIENCE,
    min_delta=0.002,
    restore_best_weights=True,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1,
)

print("Training multi-class model with realistic audio augmentation...")
print(
    f"Augmentations: noise std {NOISE_STD_RANGE}, pitch {PITCH_SHIFT_RANGE} semitones, "
    f"time shift up to {int(TIME_SHIFT_RATIO * 100)}%"
)
print(
    f"Augment copies by class: "
    + ", ".join(f"{class_name}={AUGMENT_COPIES[class_name]}" for class_name in CLASS_NAMES)
)
print(
    f"Training strategy: train/val/test={TRAIN_SIZE:.2f}/{VAL_SIZE:.2f}/{TEST_SIZE:.2f}, "
    f"epochs={EPOCHS}, batch_size={BATCH_SIZE}, early_stopping_patience={PATIENCE}"
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight_map,
    verbose=1,
)

model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

with open(HISTORY_PATH, "wb") as file_obj:
    pickle.dump(history.history, file_obj)
print(f"Training history saved to {HISTORY_PATH}")

with open(LABELS_PATH, "wb") as file_obj:
    pickle.dump(
        {
            "class_names": CLASS_NAMES,
            "class_to_index": CLASS_TO_INDEX,
            "scream_class_index": SCREAM_CLASS_INDEX,
        },
        file_obj,
    )
print(f"Label config saved to {LABELS_PATH}")

val_prob_matrix = model.predict(X_val, verbose=0)
best_threshold_row, threshold_rows = find_best_scream_threshold(y_val, val_prob_matrix, SCREAM_CLASS_INDEX)
print_threshold_table(threshold_rows)

recommended_threshold = best_threshold_row["threshold"]
print(
    "\nBest validation scream threshold:"
    f" {recommended_threshold:.2f}"
    f" | precision={best_threshold_row['precision']:.4f}"
    f" recall={best_threshold_row['recall']:.4f}"
    f" f1={best_threshold_row['f1']:.4f}"
    f" f2={best_threshold_row['f_beta']:.4f}"
)

with open(THRESHOLD_PATH, "wb") as file_obj:
    pickle.dump(
        {
            "recommended_threshold": recommended_threshold,
            "selection_beta": THRESHOLD_SELECTION_BETA,
            "min_validation_precision": MIN_VALIDATION_PRECISION,
            "threshold_rows": threshold_rows,
            "mode": "multiclass_scream_threshold",
            "scream_class_index": SCREAM_CLASS_INDEX,
        },
        file_obj,
    )
print(f"Threshold config saved to {THRESHOLD_PATH}")

test_metrics = model.evaluate(X_test, y_test, verbose=0)
print(
    "Test metrics:"
    f" loss={test_metrics[0]:.4f}"
    f" accuracy={test_metrics[1]:.4f}"
    f" top2_accuracy={test_metrics[2]:.4f}"
)

y_prob_matrix = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_prob_matrix, axis=1)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=list(range(len(CLASS_NAMES)))))
