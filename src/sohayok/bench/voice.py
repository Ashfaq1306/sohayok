"""End-to-end voice benchmark: spoken command → transcript → tool choice.

The audio is synthesized with Microsoft's neural voices (two Bangladeshi Bangla voices, two
English voices), so this measures the pipeline on clean, synthetic speech. It is an upper
bound for real microphones and accents; FLEURS (bench.stt) covers real speakers.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np

from sohayok.bench.brain import Case, desktop_cases, evaluate, massive_cases, summarize
from sohayok.bench.stt import error_rates
from sohayok.stt import Transcriber

VOICES = {
    "bn": ["bn-BD-NabanitaNeural", "bn-BD-PradeepNeural"],
    "en": ["en-US-AvaMultilingualNeural", "en-IN-PrabhatNeural"],
}


def synthesize(text: str, voice: str) -> np.ndarray:
    from scipy.signal import resample_poly

    from sohayok.tts import synthesize_edge

    samples, rate = synthesize_edge(text, voice)
    audio = resample_poly(samples, 16_000, rate).astype(np.float32)
    pad = np.zeros(4_000, dtype=np.float32)  # 0.25 s of silence on each side, like a real take
    return np.concatenate([pad, audio, pad])


def audio_cache(cases: list[Case], folder: Path) -> list[tuple[Case, np.ndarray, str]]:
    import soundfile as sf

    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for index, case in enumerate(cases):
        voice = VOICES[case.language][index % 2]
        path = folder / f"{index:04d}-{voice}.wav"
        if not path.exists():
            sf.write(path, synthesize(case.text, voice), 16_000)
        audio, _ = sf.read(path, dtype="float32")
        out.append((case, audio, voice))
    return out


def run(transcriber: Transcriber, cases: list[Case], folder: Path, brain_factory) -> dict:
    takes = audio_cache(cases, folder / "audio")
    heard_cases, rows = [], []
    for case, audio, voice in takes:
        started = time.perf_counter()
        heard = transcriber.transcribe(audio)
        rows.append(
            {
                "text": case.text,
                "language": case.language,
                "voice": voice,
                "heard": heard.text,
                "heard_language": heard.language,
                "stt_seconds": round(time.perf_counter() - started, 3),
            }
        )
        heard_cases.append(
            Case(heard.text, heard.language, case.expect, case.source, case.label, case.args)
        )
    results = evaluate(brain_factory, heard_cases, use_router=True)
    for row, result, case in zip(rows, results, cases, strict=True):
        row.update(
            {k: result[k] for k in ("calls", "reply", "tool_ok", "args_ok", "fast", "seconds")}
        )
        row["variety"] = case.variety
    summary = {
        "pipeline": summarize(
            [
                {**r, "language": c.language, "variety": c.variety}
                for r, c in zip(results, cases, strict=True)
            ]
        )
    }
    for language in ("bn", "en"):
        subset = [r for r in rows if r["language"] == language]
        if subset:
            rates = error_rates([r["text"] for r in subset], [r["heard"] for r in subset])
            summary[f"stt_{language}"] = {
                "n": len(subset),
                **{k: round(v, 4) for k, v in rates.items()},
                "language_id": round(
                    sum(r["heard_language"] == language for r in subset) / len(subset), 3
                ),
                "median_seconds": round(statistics.median(r["stt_seconds"] for r in subset), 3),
            }
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "voice.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    (folder / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    import argparse

    from sohayok.brain.ollama import OllamaBrain
    from sohayok.config import current

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--massive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=current().brain.ollama_model)
    parser.add_argument("--per-intent", type=int, default=4)
    parser.add_argument("--whisper", default=current().speech.whisper_model)
    parser.add_argument("--bangla", default=current().speech.bangla_model)
    args = parser.parse_args(argv)
    transcriber = Transcriber(args.whisper, args.bangla)
    cases = massive_cases(args.massive, args.per_intent) + desktop_cases()
    summary = run(transcriber, cases, args.output, lambda: OllamaBrain(args.model, max_steps=6))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
