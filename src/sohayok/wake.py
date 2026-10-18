"""Hands-free activation: say "Tammy" (or the built-in "hey jarvis"), fully offline.

openWakeWord turns the microphone into speech embeddings 12.5 times a second on the CPU, and a
tiny classifier scores each window for the wake word. When the score crosses the threshold,
the last two seconds of audio can be double-checked by Whisper before the assistant wakes.
"""

from __future__ import annotations

import collections
import logging
import time
from collections.abc import Callable
from importlib import resources
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
FRAME = 1280  # 80 ms at 16 kHz, openWakeWord's frame size
BUILT_IN = {"hey_jarvis": "hey jarvis", "alexa": "alexa", "hey_mycroft": "hey mycroft"}


def resolve_wake_model(setting: str) -> tuple[str, str]:
    """(model name or .onnx path, the phrase to say). "tammy" is Sohayok's own model."""
    if setting in BUILT_IN:
        return setting, BUILT_IN[setting]
    if setting.endswith(".onnx"):
        return setting, Path(setting).stem.replace("_", " ")
    bundled = resources.files("sohayok").joinpath(f"assets/{setting}.onnx")
    if bundled.is_file():
        return str(bundled), setting.replace("_", " ").title()
    raise FileNotFoundError(f"no wake word model named {setting!r}")


class WakeWordListener:
    def __init__(
        self, model: str = "tammy", threshold: float = 0.5, device: str | int | None = None
    ) -> None:
        from openwakeword import Model, utils

        try:
            self.model = Model(wakeword_models=[model], inference_framework="onnx")
        except Exception:
            if model.endswith(".onnx"):
                raise
            utils.download_models([model])
            self.model = Model(wakeword_models=[model], inference_framework="onnx")
        self.name = next(iter(self.model.models))
        self.threshold = threshold
        self.device = int(device) if isinstance(device, str) and device.isdigit() else device
        # Called with the audio that followed the wake word (float32, 16 kHz).
        self.on_wake: Callable[[np.ndarray], None] = lambda tail: None
        # Second opinion on the last 2 s of audio (float32, 16 kHz); True accepts the wake.
        self.confirm: Callable[[np.ndarray], bool] = lambda audio: True
        # While True (the assistant is busy or speaking), audio is ignored.
        self.paused: Callable[[], bool] = lambda: False
        self.gain = 1.0
        # At or above this score the wake is accepted immediately. On held-out audio, 97.7%
        # of "Tammy"s scored >= 0.8 and no frame of 1 h of Bangla speech did.
        self.instant_threshold = 0.8

    def score(self, frame: np.ndarray) -> float:
        """Wake-word probability for one 80 ms int16 frame."""
        return float(self.model.predict(frame).get(self.name, 0.0))

    def _boost(self, frame: np.ndarray) -> np.ndarray:
        """Slow automatic gain so a quiet microphone still reaches the model's usual level."""
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2))) + 1e-6
        if rms > 150:  # speech-like loudness: steer the gain towards ~1500 RMS
            target = min(8.0, max(1.0, 1500 / rms))
            self.gain += (target - self.gain) * 0.05
        return np.clip(frame.astype(np.float32) * self.gain, -32768, 32767).astype(np.int16)

    def detect(self, audio: np.ndarray) -> float:
        """Highest wake-word probability over float32 16 kHz audio (for tests and tuning)."""
        self.model.reset()
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        best = 0.0
        for start in range(0, len(pcm) - FRAME + 1, FRAME):
            best = max(best, self.score(pcm[start : start + FRAME]))
        return best

    def run(self) -> None:
        import sounddevice as sd

        log.info("listening for the wake word %r", self.name)
        recent: collections.deque[np.ndarray] = collections.deque(maxlen=25)  # 2 s
        with sd.InputStream(
            samplerate=16_000, channels=1, dtype="int16", blocksize=FRAME, device=self.device
        ) as stream:
            cooldown = 0.0
            while True:
                frame, _ = stream.read(FRAME)
                frame = self._boost(frame[:, 0])
                recent.append(frame)
                if self.paused() or time.monotonic() < cooldown:
                    continue
                score = self.score(frame)
                if score < self.threshold:
                    continue
                audio = np.concatenate(recent).astype(np.float32) / 32768
                self.model.reset()
                cooldown = time.monotonic() + 1.5
                if score >= self.instant_threshold:
                    accepted = True  # sure enough to wake at once, without waiting for Whisper
                else:
                    try:
                        accepted = self.confirm(audio)
                    except Exception:
                        log.exception("wake confirmation failed")
                        accepted = True
                if not accepted:
                    continue
                log.info("wake word heard")
                # Whatever was said while the wake word was being checked is the start of
                # the command; hand it on rather than losing it.
                waiting = stream.read_available
                tail = stream.read(waiting)[0][:, 0] if waiting else np.zeros(0, np.int16)
                stream.stop()  # release the microphone while the command is recorded
                try:
                    self.on_wake(self._boost(tail).astype(np.float32) / 32768)
                finally:
                    recent.clear()
                    stream.start()
