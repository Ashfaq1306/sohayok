"""Benchmark speech recognition on FLEURS (real read speech): WER, CER, language ID, speed."""

from __future__ import annotations

import io
import json
import random
import re
import time
import unicodedata
from pathlib import Path

import numpy as np

from sohayok.lang import strip_punctuation
from sohayok.stt import CT2Whisper, clean, load_model


def normalize_transcript(text: str) -> str:
    """Comparable form: NFC, lowercase, no punctuation (incl. ।); digits kept as written."""
    text = unicodedata.normalize("NFC", text).lower()
    text = text.replace("‌", "").replace("‍", "").replace("'", "")
    return strip_punctuation(text)


def load_fleurs(path: Path, count: int, seed: int = 20261018) -> list[dict]:
    import pyarrow.parquet as pq
    import soundfile as sf

    table = pq.read_table(path, columns=["id", "audio", "transcription", "raw_transcription"])
    rows = table.to_pylist()
    # FLEURS repeats each sentence with several speakers; keep one recording per sentence.
    unique: dict[int, dict] = {}
    for row in rows:
        unique.setdefault(row["id"], row)
    picked = sorted(unique.values(), key=lambda r: r["id"])
    random.Random(seed).shuffle(picked)
    out = []
    for row in picked[:count]:
        audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if rate != 16_000:
            raise ValueError(f"expected 16 kHz audio, got {rate}")
        out.append({"id": row["id"], "audio": audio, "reference": row["raw_transcription"]})
    return out


def error_rates(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    import jiwer

    refs = [normalize_transcript(r) for r in references]
    hyps = [normalize_transcript(h) for h in hypotheses]
    pairs = [(r, h) for r, h in zip(refs, hyps, strict=True) if r]
    refs, hyps = [p[0] for p in pairs], [p[1] for p in pairs]
    return {"wer": jiwer.wer(refs, hyps), "cer": jiwer.cer(refs, hyps)}


def run(
    data_dir: Path,
    models: list[str],
    output: Path,
    bn_count: int = 200,
    en_count: int = 100,
    device: str = "cuda",
    compute_type: str = "int8_float16",
) -> dict:
    sets = {
        "bn": load_fleurs(data_dir / "bn_in" / "test.parquet", bn_count),
        "en": load_fleurs(data_dir / "en_us" / "test.parquet", en_count),
    }
    output.mkdir(parents=True, exist_ok=True)
    summary: dict = {"data": {k: len(v) for k, v in sets.items()}, "models": {}}
    for name in models:
        model = load_model(name, device, compute_type, beam_size=1)
        result: dict = {}
        if isinstance(model, CT2Whisper) and all(sets.values()):
            correct, total = 0, 0
            for language, items in sets.items():
                for item in items:
                    guess, _ = model.detect(item["audio"], ["bn", "en"])
                    correct += guess == language
                    total += 1
            result["language_id_accuracy"] = correct / total
        for language, items in sets.items():
            if not items or (not isinstance(model, CT2Whisper) and language != "bn"):
                continue  # Bangla-only fine-tunes are not scored on English
            hypotheses, audio_seconds = [], 0.0
            model.transcribe(items[0]["audio"], language)  # warm up kernels
            started = time.perf_counter()
            for item in items:
                hypotheses.append(clean(model.transcribe(item["audio"], language)))
                audio_seconds += len(item["audio"]) / 16_000
            elapsed = time.perf_counter() - started
            rates = error_rates([i["reference"] for i in items], hypotheses)
            result[language] = {
                **{k: round(v, 4) for k, v in rates.items()},
                "real_time_factor": round(elapsed / audio_seconds, 4),
                "n": len(items),
            }
            safe = re.sub(r"[^\w.-]+", "_", name)
            with (output / f"{safe}.{language}.jsonl").open("w", encoding="utf-8") as handle:
                for item, hyp in zip(items, hypotheses, strict=True):
                    handle.write(
                        json.dumps(
                            {"id": item["id"], "reference": item["reference"], "hypothesis": hyp},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        summary["models"][name] = result
        print(name, json.dumps(result), flush=True)
        del model
        try:
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="folder with bn_in/ and en_us/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--bn", type=int, default=200)
    parser.add_argument("--en", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="int8_float16")
    args = parser.parse_args(argv)
    run(args.data, args.model, args.output, args.bn, args.en, args.device, args.compute_type)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
