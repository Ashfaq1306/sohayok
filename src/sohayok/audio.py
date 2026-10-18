"""Microphone capture with end-of-speech detection."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable

import numpy as np

log = logging.getLogger(__name__)
RATE = 16_000
BLOCK = 512  # 32 ms, the frame size Silero VAD expects at 16 kHz


class SpeechDetector:
    """Silero VAD (the ONNX model bundled with faster-whisper), one 32 ms frame at a time."""

    def __init__(self, threshold: float = 0.5) -> None:
        from faster_whisper.vad import get_vad_model

        self.model = get_vad_model()
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)

    def probability(self, frame: np.ndarray) -> float:
        """Speech probability of one 512-sample float32 frame (state carries across frames)."""
        if frame.shape[0] != BLOCK:
            return 0.0
        frame = frame.astype(np.float32)[None, :]
        batch = np.concatenate([self._context, frame], axis=1)
        out, self._h, self._c = self.model.session.run(
            None,
            {"input": batch, "h": self._h, "c": self._c},
        )
        self._context = frame[:, -64:]
        return float(np.asarray(out).reshape(-1)[0])


def normalize_level(
    audio: np.ndarray, target_peak: float = 0.5, max_gain: float = 20.0
) -> np.ndarray:
    """Bring quiet recordings up to a normal level (robust to clicks: uses the 99.5th
    percentile, not the single loudest sample)."""
    if audio.size == 0:
        return audio
    level = float(np.percentile(np.abs(audio), 99.5))
    if level < 1e-4:
        return audio
    gain = min(max_gain, target_peak / level)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def save_recording(folder, audio: np.ndarray, meta: dict, keep: int) -> None:
    """Store one voice command (16 kHz WAV + JSON) and keep only the newest `keep`."""
    import json
    from datetime import datetime
    from pathlib import Path

    import soundfile as sf

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    sf.write(stem.with_suffix(".wav"), audio, RATE)
    stem.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
    for old in sorted(folder.glob("*.wav"))[:-keep]:
        old.unlink(missing_ok=True)
        old.with_suffix(".json").unlink(missing_ok=True)


def energy_probability(frame: np.ndarray) -> float:
    """Fallback detector: loudness mapped to 0..1 (used if the VAD model cannot load)."""
    rms = float(np.sqrt(np.mean(frame**2)) + 1e-9)
    return min(1.0, max(0.0, (20 * np.log10(rms) + 50) / 20))


class Recorder:
    """Records one utterance: waits for speech, then stops after a pause."""

    def __init__(
        self,
        device: str | int | None = None,
        silence_seconds: float = 0.9,
        max_seconds: float = 20.0,
        start_timeout: float = 6.0,
    ) -> None:
        self.device = (
            int(device) if isinstance(device, str) and device.isdigit() else device or None
        )
        self.silence_seconds = silence_seconds
        self.max_seconds = max_seconds
        self.start_timeout = start_timeout
        self._stop = threading.Event()
        try:
            self.detector: SpeechDetector | None = SpeechDetector()
        except Exception:
            log.exception("Silero VAD unavailable; using a loudness threshold")
            self.detector = None
        self.on_level: Callable[[float], None] = lambda level: None
        self._gain = 1.0

    def stop(self) -> None:
        """End the current recording now (push-to-talk released, or the user clicked stop)."""
        self._stop.set()

    def _speech(self, frame: np.ndarray) -> float:
        # A slowly adapting gain lets quiet microphones trigger the detector like loud ones.
        rms = float(np.sqrt(np.mean(frame**2))) + 1e-7
        if rms > 0.002:
            self._gain += (min(12.0, max(1.0, 0.05 / rms)) - self._gain) * 0.1
        frame = np.clip(frame * self._gain, -1, 1)
        if self.detector is None:
            return energy_probability(frame)
        return self.detector.probability(frame)

    def record(self) -> np.ndarray:
        """Return 16 kHz mono float32 audio of one utterance (empty if nobody spoke)."""
        import sounddevice as sd

        self._stop.clear()
        if self.detector:
            self.detector.reset()
        frames: queue.Queue[np.ndarray] = queue.Queue()

        def callback(data, _frames, _time, status):
            if status:
                log.debug("audio status: %s", status)
            frames.put(data[:, 0].copy())

        kept: list[np.ndarray] = []
        preroll: list[np.ndarray] = []  # 0.3 s before speech starts, so first syllables survive
        started = False
        silent_for = 0.0
        began = time.monotonic()
        frame_seconds = BLOCK / RATE
        with sd.InputStream(
            samplerate=RATE,
            channels=1,
            dtype="float32",
            blocksize=BLOCK,
            device=self.device,
            callback=callback,
        ):
            while not self._stop.is_set():
                try:
                    frame = frames.get(timeout=0.5)
                except queue.Empty:
                    continue
                self.on_level(float(np.sqrt(np.mean(frame**2))))
                speaking = self._speech(frame) >= 0.5
                if not started:
                    preroll = (preroll + [frame])[-10:]
                    if speaking:
                        started = True
                        kept.extend(preroll)
                    elif time.monotonic() - began > self.start_timeout:
                        break
                    continue
                kept.append(frame)
                silent_for = 0.0 if speaking else silent_for + frame_seconds
                if silent_for >= self.silence_seconds:
                    break
                if len(kept) * frame_seconds >= self.max_seconds:
                    break
        if not started and self._stop.is_set():
            # Push-to-talk released before the detector fired: keep what was heard.
            while not frames.empty():
                preroll.append(frames.get())
            kept = preroll
        return np.concatenate(kept) if kept else np.zeros(0, dtype=np.float32)


def input_devices() -> list[str]:
    import sounddevice as sd

    default = sd.default.device[0]
    return [
        f"{i}: {d['name']}{' (default)' if i == default else ''}"
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]


def play(samples: np.ndarray, rate: int, stop: threading.Event | None = None) -> None:
    """Play audio, returning early if `stop` is set (the user interrupted)."""
    import sounddevice as sd

    sd.play(samples, rate)
    duration = len(samples) / rate
    deadline = time.monotonic() + duration + 0.2
    while time.monotonic() < deadline:
        if stop is not None and stop.wait(0.05):
            sd.stop()
            return
        if stop is None:
            time.sleep(0.05)
    sd.wait()


def chime(frequency: float = 880.0, seconds: float = 0.12, volume: float = 0.2) -> None:
    """A short tone that says "I'm listening"; it finishes before recording starts, so the
    microphone never hears it."""
    import sounddevice as sd

    t = np.arange(int(48_000 * seconds)) / 48_000
    envelope = np.minimum(1.0, np.minimum(t, seconds - t) / 0.01)
    sd.play((volume * envelope * np.sin(2 * np.pi * frequency * t)).astype(np.float32), 48_000)
    sd.wait()
    time.sleep(0.08)  # let the speaker ring out
