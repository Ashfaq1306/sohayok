"""Evaluate the two-stage wake word (detector, then Whisper check) on held-out audio.

Positives: "Tammy" phrases from the validation voices (never used in training), augmented
like real microphones. Negatives: FLEURS Bangla validation speech (not used in training),
streamed exactly like the live listener. Reports detection rate and false wakes per hour.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

import sohayok.training.wake_word as ww
from sohayok.training.wake_word import (
    CLIP,
    POSITIVE_PHRASES,
    RATE,
    augment,
    fleurs_windows,
    pick_voices,
    place,
    tts,
)


def stream_scores(listener, audio: np.ndarray) -> list[tuple[int, float]]:
    """Feed audio through the detector frame by frame; returns (end sample, score)."""
    from sohayok.wake import FRAME

    listener.model.reset()
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    out = []
    for start in range(0, len(pcm) - FRAME + 1, FRAME):
        out.append((start + FRAME, listener.score(pcm[start : start + FRAME])))
    return out


def run(
    fleurs_val: Path,
    cache: Path,
    model: str,
    threshold: float,
    minutes: float,
    device: str = "cpu",
    seed: int = 20261018,
) -> dict:
    from sohayok.desktop import heard_wake_name
    from sohayok.stt import CT2Whisper
    from sohayok.wake import WakeWordListener

    ww.CACHE = cache
    rng = random.Random(seed + 1)
    voices = pick_voices(seed)
    held_out = {lang: v[int(len(v) * 0.8) :] for lang, v in voices.items()}
    listener = WakeWordListener(model, threshold)
    whisper = CT2Whisper(
        "deepdml/faster-whisper-large-v3-turbo-ct2", device=device, compute_type="int8"
    )

    positives = []
    for lang, phrases in POSITIVE_PHRASES.items():
        for voice in held_out[lang]:
            for phrase in phrases:
                try:
                    audio = tts(phrase, voice, "+0%", "+0Hz")
                except Exception:
                    continue
                positives.append((phrase, voice, augment(place(audio, rng, True), rng)))
    detected = confirmed = 0
    misses = []
    for phrase, voice, clip in positives:
        # a second of silence first, as when the listener has been running for a while
        audio = np.concatenate([np.zeros(RATE, np.float32), clip, np.zeros(RATE // 2, np.float32)])
        scores = stream_scores(listener, audio)
        hits = [end for end, s in scores if s >= threshold]
        if not hits:
            misses.append((phrase, voice, "detector"))
            continue
        detected += 1
        end = hits[0]
        window = audio[max(0, end - CLIP) : end]
        text = whisper.transcribe(window, "en")
        if heard_wake_name(text):
            confirmed += 1
        else:
            misses.append((phrase, voice, f"whisper heard {text!r}"))

    windows = fleurs_windows(fleurs_val, int(minutes * 60), rng)  # 2 s windows, 1 s hop
    speech = np.concatenate([w[RATE:] for w in windows])  # contiguous ~1 s per window
    started = time.perf_counter()
    scores = stream_scores(listener, speech)
    flagged, false_wakes, last = 0, [], -10 * RATE
    for end, score in scores:
        if score < threshold or end - last < 1.5 * RATE:  # the listener's cooldown
            continue
        last = end
        flagged += 1
        text = whisper.transcribe(speech[max(0, end - CLIP) : end], "en")
        if heard_wake_name(text):
            false_wakes.append(text)
    hours = len(speech) / RATE / 3600
    result = {
        "threshold": threshold,
        "positives": len(positives),
        "detector_recall": round(detected / len(positives), 3),
        "two_stage_recall": round(confirmed / len(positives), 3),
        "negative_speech_hours": round(hours, 3),
        "detector_flags_per_hour": round(flagged / hours, 1),
        "false_wakes": len(false_wakes),
        "false_wakes_per_hour": round(len(false_wakes) / hours, 2),
        "false_wake_texts": false_wakes[:10],
        "misses": misses[:15],
        "seconds": round(time.perf_counter() - started, 1),
    }
    return result


if __name__ == "__main__":
    import sys

    out = run(
        Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))
