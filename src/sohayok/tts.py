"""Speak replies: Microsoft neural voices (Bangla and English), with offline fallbacks."""

from __future__ import annotations

import asyncio
import io
import logging
import queue
import re
import threading

import numpy as np

from sohayok.lang import bangla_share

log = logging.getLogger(__name__)


def synthesize_edge(text: str, voice: str, rate: str = "+0%") -> tuple[np.ndarray, int]:
    """Synthesize with Edge's online neural voices; returns float32 mono samples and rate."""
    import edge_tts
    import soundfile as sf

    async def collect() -> bytes:
        audio = bytearray()
        async for chunk in edge_tts.Communicate(text, voice, rate=rate).stream():
            if chunk["type"] == "audio":
                audio.extend(chunk["data"])
        return bytes(audio)

    data = asyncio.run(collect())
    if not data:
        raise RuntimeError("edge-tts returned no audio")
    samples, rate_hz = sf.read(io.BytesIO(data), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples, rate_hz


SENTENCE_END = re.compile(r"(?<=[।.!?])\s+")


def sentences(text: str, shortest: int = 25) -> list[str]:
    """Split a reply into sentences, joining very short ones ("জি!") to the next."""
    parts, current = [], ""
    for piece in SENTENCE_END.split(text.strip()):
        current = f"{current} {piece}".strip()
        if len(current) >= shortest:
            parts.append(current)
            current = ""
    if current:
        if parts and len(current) < shortest:
            parts[-1] = f"{parts[-1]} {current}"
        else:
            parts.append(current)
    return parts


class Speaker:
    def __init__(
        self, bangla_voice: str, english_voice: str, rate: str = "+0%", enabled: bool = True
    ) -> None:
        self.voices = {"bn": bangla_voice, "en": english_voice}
        self.rate = rate
        self.enabled = enabled
        self.stop_event = threading.Event()
        self.speaking = threading.Event()
        self._lock = threading.Lock()
        self._sapi = None

    def voice_language(self, text: str) -> str:
        return "bn" if bangla_share(text) >= 0.3 else "en"

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        language = self.voice_language(text)
        try:
            return synthesize_edge(text, self.voices[language], self.rate)
        except Exception as exc:
            log.warning("online voice failed (%s); using the offline voice", exc)
            raise

    def say(self, text: str) -> None:
        """Speak `text`, blocking until done or interrupted with `stop()`."""
        from sohayok.audio import play

        text = text.strip()
        if not text or not self.enabled:
            return
        with self._lock:
            self.stop_event.clear()
            self.speaking.set()
            try:
                # Speak the first sentence as soon as it is ready and prepare the rest while it
                # plays: a 150-character answer waited 5.3 s for its first word otherwise.
                parts = sentences(text)
                ready: queue.Queue = queue.Queue()

                def produce() -> None:
                    for part in parts:
                        if self.stop_event.is_set():
                            break
                        try:
                            ready.put(self.synthesize(part))
                        except Exception:
                            ready.put(part)  # spoken offline instead
                    ready.put(None)

                threading.Thread(target=produce, name="tts", daemon=True).start()
                while not self.stop_event.is_set():
                    item = ready.get()
                    if item is None:
                        break
                    if isinstance(item, str):
                        self._say_offline(item)
                    else:
                        play(item[0], item[1], self.stop_event)
            finally:
                self.speaking.clear()

    def stop(self) -> None:
        self.stop_event.set()
        if self._sapi is not None:
            try:
                self._sapi.Speak("", 2)  # SVSFPurgeBeforeSpeak
            except Exception:
                pass

    def _say_offline(self, text: str) -> None:
        """Windows' built-in SAPI voice; it cannot read Bangla, so Bangla is skipped."""
        if self.voice_language(text) == "bn":
            log.warning("no offline Bangla voice; reply shown on screen only")
            return
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        self._sapi = win32com.client.Dispatch("SAPI.SpVoice")
        self._sapi.Speak(text)
