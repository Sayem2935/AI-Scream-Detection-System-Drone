import collections
import io
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

from detect_cnn_live import BLOCK_SIZE, SAMPLE_RATE, ScreamDetector, extract_spec


ALERT_CLIP_SECONDS = 3
WAVEFORM_SECONDS = 3


class DetectionBackend:
    def __init__(self, history_size=20):
        self.detector = ScreamDetector()
        self.history = collections.deque([0.0] * history_size, maxlen=history_size)
        self.alert_history = collections.deque(maxlen=20)
        self.waveform_buffer = np.zeros(int(SAMPLE_RATE * WAVEFORM_SECONDS), dtype=np.float32)
        self.lock = threading.Lock()
        self.stream = None
        self.running = False
        self.last_error = None
        self.last_result = self._default_result()
        self.last_alert_audio_bytes = None
        self.last_alert_filename = None
        self.last_alert_timestamp = None

    def _default_result(self):
        return {
            "timestamp": time.time(),
            "confidence": 0.0,
            "raw_confidence": 0.0,
            "adjusted_confidence": 0.0,
            "predicted_class": "noise",
            "predicted_confidence": 0.0,
            "class_probabilities": {},
            "state": "NORMAL",
            "alert_triggered": False,
            "trigger_mode": None,
            "vote_count": 0,
            "vote_window": 0,
            "consecutive_positive": 0,
            "weak_positive_frames": 0,
            "thresholds": {
                "low": 0.18,
                "mid": 0.28,
                "high": 0.33,
            },
            "reasons": [],
            "event": {
                "rms": 0.0,
                "active_ratio": 0.0,
                "zcr": 0.0,
                "flatness": 0.0,
                "centroid": 0.0,
            },
            "debug": {
                "rms": 0.0,
                "raw_prediction": 0.0,
                "smoothed_prediction": 0.0,
                "decision": "NORMAL",
            },
        }

    def _build_spectrogram_image(self):
        spec = extract_spec(self.detector.audio_buffer).squeeze()
        spec_min = float(np.min(spec))
        spec_max = float(np.max(spec))
        normalized = (spec - spec_min) / (spec_max - spec_min + 1e-6)
        heat = np.stack(
            [
                normalized * 0.20,
                normalized * 0.85,
                0.25 + normalized * 0.75,
            ],
            axis=-1,
        )
        return np.clip(heat, 0.0, 1.0)

    def _create_alert_audio_bytes(self):
        clip_audio = self.waveform_buffer.copy()
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, clip_audio, SAMPLE_RATE, format="WAV")
        audio_buffer.seek(0)
        return audio_buffer.read()

    def _append_alert_history(self, result):
        event_time = time.strftime("%H:%M:%S", time.localtime(result["timestamp"]))
        self.alert_history.appendleft(
            {
                "time": event_time,
                "confidence": round(float(result["confidence"]), 3),
                "status": result["state"],
                "class": result["predicted_class"],
            }
        )

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            with self.lock:
                self.last_error = str(status)

        try:
            audio_chunk = indata[:, 0].astype(np.float32)
            self.waveform_buffer = np.roll(self.waveform_buffer, -len(audio_chunk))
            self.waveform_buffer[-len(audio_chunk):] = audio_chunk

            result = self.detector.process_audio_chunk(audio_chunk, save_alert_audio=False)
            if result is None:
                return

            spectrogram_image = self._build_spectrogram_image()

            with self.lock:
                self.last_result = result
                self.history.append(float(result["confidence"]))

                if result["alert_triggered"]:
                    self.last_alert_audio_bytes = self._create_alert_audio_bytes()
                    self.last_alert_filename = f"alert_{int(result['timestamp'])}.wav"
                    self.last_alert_timestamp = result["timestamp"]
                    self._append_alert_history(result)
        except Exception as exc:
            with self.lock:
                self.last_error = str(exc)
                spectrogram_image = None

        with self.lock:
            self.last_result["waveform"] = self.waveform_buffer.copy()
            self.last_result["spectrogram_image"] = spectrogram_image

    def start(self):
        with self.lock:
            if self.running:
                return
            self.last_error = None
            self.detector.reset()
            self.history.clear()
            self.history.extend([0.0] * self.history.maxlen)
            self.waveform_buffer = np.zeros_like(self.waveform_buffer)

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=self._audio_callback,
            blocksize=BLOCK_SIZE,
        )
        self.stream.start()

        with self.lock:
            self.running = True
            self.last_result = self._default_result()
            self.last_result["waveform"] = self.waveform_buffer.copy()
            self.last_result["spectrogram_image"] = self._build_spectrogram_image()

    def stop(self):
        with self.lock:
            if not self.running:
                return
            self.running = False

        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            finally:
                self.stream = None

    def get_snapshot(self):
        with self.lock:
            snapshot = dict(self.last_result)
            snapshot["event"] = dict(self.last_result["event"])
            snapshot["reasons"] = list(self.last_result["reasons"])
            snapshot["class_probabilities"] = dict(self.last_result.get("class_probabilities", {}))
            snapshot["thresholds"] = dict(self.last_result.get("thresholds", {}))
            snapshot["debug"] = dict(self.last_result.get("debug", {}))
            snapshot["history"] = list(self.history)
            snapshot["waveform"] = snapshot.get("waveform", self.waveform_buffer.copy()).copy()
            snapshot["spectrogram_image"] = snapshot.get("spectrogram_image")
            snapshot["alert_history"] = list(self.alert_history)
            snapshot["running"] = self.running
            snapshot["last_error"] = self.last_error
            snapshot["last_alert_audio_bytes"] = self.last_alert_audio_bytes
            snapshot["last_alert_filename"] = self.last_alert_filename
            snapshot["last_alert_timestamp"] = self.last_alert_timestamp
        return snapshot


_backend_instance = None
_backend_lock = threading.Lock()


def get_backend():
    global _backend_instance

    with _backend_lock:
        if _backend_instance is None:
            _backend_instance = DetectionBackend()
        return _backend_instance
