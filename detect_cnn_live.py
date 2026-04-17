import collections
import os
import pickle
import time

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
import tensorflow as tf
from scipy.signal import butter, lfilter
from tensorflow.keras.layers import BatchNormalization, Dense


MODEL_PATH = "cnn_scream_model.h5"
THRESHOLD_PATH = "threshold_config.pkl"
LABELS_PATH = "label_config.pkl"
SAMPLE_RATE = 16000

CHUNK_SECONDS = 0.25
WINDOW_SECONDS = 1.0
BLOCK_SIZE = int(SAMPLE_RATE * CHUNK_SECONDS)
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)

SMOOTHING_WINDOW = 5
MIN_POSITIVE_FRAMES = 2
COOLDOWN_SECONDS = 2.0

RMS_THRESHOLD = 0.005
MIN_ACTIVE_RATIO = 0.30
MAX_SPECTRAL_FLATNESS = 0.32
MAX_ZCR = 0.18
MAX_CENTROID = 2400
IMPULSE_PENALTY = 0.35
QUIET_PENALTY = 0.55
LOW_THRESHOLD = 0.18
MID_THRESHOLD = 0.28
HIGH_THRESHOLD = 0.33
CONFIDENCE_BOOST = 1.2

DEFAULT_CLASS_NAMES = ["scream", "cough", "clap", "speech", "noise"]


class CustomBatchNorm(BatchNormalization):
    def __init__(self, **kwargs):
        kwargs.pop("renorm", None)
        kwargs.pop("renorm_clipping", None)
        kwargs.pop("renorm_momentum", None)
        super().__init__(**kwargs)


class CustomDense(Dense):
    def __init__(self, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(**kwargs)


def load_threshold():
    default_threshold = 0.35
    if not os.path.exists(THRESHOLD_PATH):
        return default_threshold

    try:
        with open(THRESHOLD_PATH, "rb") as file_obj:
            config = pickle.load(file_obj)
        return float(config.get("recommended_threshold", default_threshold))
    except Exception as exc:
        print(f"Failed to load threshold config: {exc}")
        return default_threshold


def load_labels():
    if not os.path.exists(LABELS_PATH):
        return DEFAULT_CLASS_NAMES, 0

    try:
        with open(LABELS_PATH, "rb") as file_obj:
            config = pickle.load(file_obj)
        class_names = config.get("class_names", DEFAULT_CLASS_NAMES)
        scream_class_index = int(config.get("scream_class_index", 0))
        return class_names, scream_class_index
    except Exception as exc:
        print(f"Failed to load label config: {exc}")
        return DEFAULT_CLASS_NAMES, 0


def bandpass(audio, low_hz=250, high_hz=4000, order=3):
    nyquist = 0.5 * SAMPLE_RATE
    low = low_hz / nyquist
    high = high_hz / nyquist
    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, audio)


def extract_spec(audio):
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

    return spec_db[:64, :64][..., np.newaxis]


def compute_event_features(audio):
    rms = float(np.sqrt(np.mean(audio ** 2)))

    frame = 512
    hop = 128
    frame_rms = librosa.feature.rms(y=audio, frame_length=frame, hop_length=hop)[0]
    peak_rms = float(np.max(frame_rms) + 1e-6)
    active_ratio = float(np.mean(frame_rms > (0.45 * peak_rms)))

    zcr = float(
        np.mean(
            librosa.feature.zero_crossing_rate(
                y=audio,
                frame_length=frame,
                hop_length=hop,
            )[0]
        )
    )

    flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio, n_fft=frame, hop_length=hop)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE, n_fft=frame, hop_length=hop)))

    return {
        "rms": rms,
        "active_ratio": active_ratio,
        "zcr": zcr,
        "flatness": flatness,
        "centroid": centroid,
    }


def adjust_scream_probability(raw_prob, event):
    adjusted = raw_prob
    is_quiet = event["rms"] < RMS_THRESHOLD
    is_impulsive = event["active_ratio"] < MIN_ACTIVE_RATIO
    is_noisy = (
        event["flatness"] > MAX_SPECTRAL_FLATNESS
        or event["zcr"] > MAX_ZCR
        or event["centroid"] > MAX_CENTROID
    )

    reasons = []
    if is_quiet:
        adjusted *= QUIET_PENALTY
        reasons.append("quiet")
    if is_impulsive:
        adjusted *= IMPULSE_PENALTY
        reasons.append("short-burst")
    elif is_noisy:
        adjusted *= 0.60
        reasons.append("broadband")

    return float(adjusted), reasons


class ScreamDetector:
    def __init__(self, model_path=MODEL_PATH):
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "BatchNormalization": CustomBatchNorm,
                "Dense": CustomDense,
            },
            compile=False,
        )
        self.class_names, self.scream_class_index = load_labels()
        self.raw_trigger_threshold = max(HIGH_THRESHOLD, load_threshold())
        self.reset()

    def reset(self):
        self.audio_buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)
        self.score_history = collections.deque(maxlen=SMOOTHING_WINDOW)
        self.consecutive_positive = 0
        self.in_alarm_state = False
        self.last_trigger_time = 0.0

    def update_moving_average(self, new_value):
        self.score_history.append(new_value)
        return float(np.mean(self.score_history))

    def process_audio_chunk(self, chunk, save_alert_audio=False, alert_directory="."):
        chunk = np.asarray(chunk, dtype=np.float32).flatten()
        if chunk.size == 0:
            return None

        self.audio_buffer = np.roll(self.audio_buffer, -len(chunk))
        self.audio_buffer[-len(chunk):] = chunk

        normalized_audio = self.audio_buffer / (np.max(np.abs(self.audio_buffer)) + 1e-6)
        filtered = bandpass(normalized_audio)
        event = compute_event_features(filtered)
        rms = event["rms"]

        # Hard silence filter: ignore near-silent input before model inference.
        if rms < RMS_THRESHOLD:
            raw_scream_prob = 0.0
            adjusted_prob = 0.0
            smooth_prob = self.update_moving_average(0.0)
            self.consecutive_positive = 0
            self.in_alarm_state = False
            predicted_class = "noise"
            predicted_confidence = 1.0
            class_probabilities = {
                name: (1.0 if name == "noise" else 0.0)
                for name in self.class_names
            }
            decision = "NORMAL"
            return {
                "timestamp": time.time(),
                "raw_confidence": raw_scream_prob,
                "adjusted_confidence": adjusted_prob,
                "confidence": smooth_prob,
                "predicted_class": predicted_class,
                "predicted_confidence": predicted_confidence,
                "class_probabilities": class_probabilities,
                "vote_count": self.consecutive_positive,
                "vote_window": SMOOTHING_WINDOW,
                "consecutive_positive": self.consecutive_positive,
                "weak_positive_frames": 0,
                "state": decision,
                "alert_triggered": False,
                "alert_audio_path": None,
                "trigger_mode": None,
                "thresholds": {
                    "low": LOW_THRESHOLD,
                    "mid": MID_THRESHOLD,
                    "high": HIGH_THRESHOLD,
                },
                "event": event,
                "reasons": ["silence"],
                "debug": {
                    "rms": rms,
                    "raw_prediction": raw_scream_prob,
                    "smoothed_prediction": smooth_prob,
                    "decision": decision,
                },
            }

        features = extract_spec(filtered)
        features = np.expand_dims(features, axis=0)
        class_probs = self.model.predict(features, verbose=0)[0]
        raw_scream_prob = float(class_probs[self.scream_class_index])
        predicted_index = int(np.argmax(class_probs))
        predicted_class = self.class_names[predicted_index]
        predicted_confidence = float(class_probs[predicted_index])

        adjusted_prob, reasons = adjust_scream_probability(raw_scream_prob, event)
        adjusted_prob = min(1.0, adjusted_prob * CONFIDENCE_BOOST)
        smooth_prob = self.update_moving_average(adjusted_prob)
        vote_count = sum(1 for value in self.score_history if value >= HIGH_THRESHOLD)

        strong_trigger = (
            predicted_class == "scream"
            and self.consecutive_positive >= MIN_POSITIVE_FRAMES
            and smooth_prob > HIGH_THRESHOLD
        )

        if smooth_prob > HIGH_THRESHOLD and predicted_class == "scream":
            self.consecutive_positive += 1
        else:
            self.consecutive_positive = 0

        alert_triggered = False
        alert_path = None
        now = time.time()
        if not self.in_alarm_state and strong_trigger:
            if now - self.last_trigger_time >= COOLDOWN_SECONDS:
                alert_triggered = True
                self.last_trigger_time = now
                if save_alert_audio:
                    os.makedirs(alert_directory, exist_ok=True)
                    alert_path = os.path.join(alert_directory, f"alert_{int(now)}.wav")
                    sf.write(alert_path, self.audio_buffer, SAMPLE_RATE)
            self.in_alarm_state = True

        if smooth_prob < LOW_THRESHOLD:
            state = "NORMAL"
            if now - self.last_trigger_time >= COOLDOWN_SECONDS:
                self.in_alarm_state = False
        elif smooth_prob <= MID_THRESHOLD:
            state = "POSSIBLE SCREAM"
        else:
            state = "SCREAM DETECTED" if self.in_alarm_state else "POSSIBLE SCREAM"

        debug = {
            "rms": rms,
            "raw_prediction": raw_scream_prob,
            "smoothed_prediction": smooth_prob,
            "decision": state,
        }

        return {
            "timestamp": now,
            "raw_confidence": raw_scream_prob,
            "adjusted_confidence": adjusted_prob,
            "confidence": smooth_prob,
            "predicted_class": predicted_class,
            "predicted_confidence": predicted_confidence,
            "class_probabilities": {name: float(class_probs[index]) for index, name in enumerate(self.class_names)},
            "vote_count": vote_count,
            "vote_window": SMOOTHING_WINDOW,
            "consecutive_positive": self.consecutive_positive,
            "weak_positive_frames": 0,
            "state": state,
            "alert_triggered": alert_triggered,
            "alert_audio_path": alert_path,
            "trigger_mode": "strong" if alert_triggered else None,
            "thresholds": {
                "low": LOW_THRESHOLD,
                "mid": MID_THRESHOLD,
                "high": HIGH_THRESHOLD,
            },
            "event": event,
            "reasons": reasons,
            "debug": debug,
        }


def run_terminal_detector():
    detector = ScreamDetector()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")

        result = detector.process_audio_chunk(indata[:, 0], save_alert_audio=True)
        if result is None:
            return

        if result["alert_triggered"]:
            print(
                f"ALERT | mode={result['trigger_mode']} smooth={result['confidence']:.2f} "
                f"scream_raw={result['raw_confidence']:.2f} class={result['predicted_class']} "
                f"saved={result['alert_audio_path']}"
            )

        event = result["event"]
        reason_text = ",".join(result["reasons"]) if result["reasons"] else "none"
        print(
            f"RMS={result['debug']['rms']:.3f} raw={result['debug']['raw_prediction']:.2f} "
            f"smooth={result['debug']['smoothed_prediction']:.2f} decision={result['debug']['decision']} "
            f"class={result['predicted_class']} class_conf={result['predicted_confidence']:.2f} "
            f"votes={result['vote_count']}/{result['vote_window']} frames={result['consecutive_positive']} "
            f"active={event['active_ratio']:.2f} "
            f"flat={event['flatness']:.2f} zcr={event['zcr']:.2f} centroid={event['centroid']:.0f} "
            f"reason={reason_text} state={result['state']}"
        )

    print("Listening for screams... Press Enter to stop.")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback,
        blocksize=BLOCK_SIZE,
    ):
        input()


if __name__ == "__main__":
    run_terminal_detector()
