"""Replay kept wake attempts (%APPDATA%\\Sohayok\\recordings\\wake) through the live pipeline.

Each clip is streamed through the detector with the speech gate, exactly as the listener
does; borderline candidates go through the same double-check as the app. A clip counts as a
real "Tammy" attempt when it contains at least 0.3 s of speech (in the user's first session
every clip with speech was an attempt and every clip without was noise), so read the result as
"attempts that would wake" and "noise that would wake".
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np


def replay(
    folder: str, model: str | None = None, thresholds: tuple[float, float] | None = None
) -> dict:
    import soundfile as sf

    from sohayok.audio import BLOCK, SpeechDetector
    from sohayok.config import current
    from sohayok.desktop import heard_wake_name
    from sohayok.models import resolve_bangla_model
    from sohayok.stt import CT2Whisper
    from sohayok.wake import FRAME, WakeWordListener, resolve_wake_model

    config = current().activation
    listener = WakeWordListener(
        model or resolve_wake_model(config.wake_model)[0], config.wake_threshold
    )
    listener.instant_threshold = config.wake_instant_threshold
    if thresholds:
        listener.threshold, listener.instant_threshold = thresholds
    vad = SpeechDetector()
    bangla = CT2Whisper(resolve_bangla_model("auto"), "cuda", "int8_float16")
    whisper = CT2Whisper("deepdml/faster-whisper-large-v3-turbo-ct2", "cuda", "int8_float16")
    rows = []
    for path in sorted(glob.glob(os.path.join(folder, "*.wav"))):
        audio, _ = sf.read(path, dtype="float32")
        vad.reset()
        speech = (
            sum(
                vad.probability(audio[i : i + BLOCK]) >= 0.5
                for i in range(0, len(audio) - BLOCK + 1, BLOCK)
            )
            * BLOCK
            / 16_000
        )
        # stream with a second of silence first, like a running listener
        stream = np.concatenate([np.zeros(16_000, np.float32), audio])
        pcm = (np.clip(stream, -1, 1) * 32767).astype(np.int16)
        listener.model.reset()
        decision, best = "none", 0.0
        for start in range(0, len(pcm) - FRAME + 1, FRAME):
            score = listener.score(pcm[start : start + FRAME])
            best = max(best, score)
            if score < listener.threshold:
                continue
            window = stream[max(0, start + FRAME - 32_000) : start + FRAME]
            if listener.speech_in(window) < listener.min_speech:
                continue
            if score >= listener.instant_threshold:
                decision = "instant"
            else:
                texts = [bangla.transcribe(window, "bn")]
                if not heard_wake_name(texts[0]):
                    texts.append(whisper.transcribe(window, "en"))
                decision = "confirmed" if heard_wake_name(texts[-1]) else "refused"
            break
        rows.append(
            {
                "clip": os.path.basename(path),
                "speech": round(speech, 2),
                "best_score": round(best, 2),
                "decision": decision,
            }
        )
    attempts = [r for r in rows if r["speech"] >= 0.3]
    noise = [r for r in rows if r["speech"] < 0.3]
    woke = lambda rs: sum(r["decision"] in {"instant", "confirmed"} for r in rs)  # noqa: E731
    return {
        "attempts": len(attempts),
        "attempts_woke": woke(attempts),
        "attempts_instant": sum(r["decision"] == "instant" for r in attempts),
        "noise": len(noise),
        "noise_woke": woke(noise),
        "rows": rows,
    }


if __name__ == "__main__":
    folder = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.expandvars(r"%APPDATA%\Sohayok\recordings\wake")
    )
    result = replay(folder, sys.argv[2] if len(sys.argv) > 2 else None)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))
    for row in result["rows"]:
        if row["speech"] >= 0.3 or row["decision"] in {"instant", "confirmed"}:
            print(row)
