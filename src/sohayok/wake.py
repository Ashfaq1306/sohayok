"""Hands-free activation with openWakeWord ("hey jarvis"), fully offline on the CPU."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import numpy as np

log = logging.getLogger(__name__)
FRAME = 1280  # 80 ms at 16 kHz, openWakeWord's frame size


class WakeWordListener:
    def __init__(
        self, model: str = "hey_jarvis", threshold: float = 0.5, device: str | int | None = None
    ) -> None:
        from openwakeword import Model, utils

        try:
            self.model = Model(wakeword_models=[model], inference_framework="onnx")
        except Exception:
            utils.download_models([model])
            self.model = Model(wakeword_models=[model], inference_framework="onnx")
        self.name = model
        self.threshold = threshold
        self.device = int(device) if isinstance(device, str) and device.isdigit() else device
        self.on_wake: Callable[[], None] = lambda: None
        # While True (the assistant is busy or speaking), audio is ignored.
        self.paused: Callable[[], bool] = lambda: False

    def score(self, frame: np.ndarray) -> float:
        """Wake-word probability for one 80 ms int16 frame."""
        return float(self.model.predict(frame).get(self.name, 0.0))

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
        with sd.InputStream(
            samplerate=16_000, channels=1, dtype="int16", blocksize=FRAME, device=self.device
        ) as stream:
            cooldown = 0.0
            while True:
                frame, _ = stream.read(FRAME)
                if self.paused() or time.monotonic() < cooldown:
                    continue
                if self.score(frame[:, 0]) >= self.threshold:
                    log.info("wake word heard")
                    self.model.reset()
                    cooldown = time.monotonic() + 2.0
                    # Release the microphone while the command is recorded.
                    stream.stop()
                    try:
                        self.on_wake()
                    finally:
                        stream.start()
