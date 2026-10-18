"""Speech recognition: detect Bangla or English, then transcribe with the best model for it."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Whisper's well-known outputs for silence or noise.
HALLUCINATIONS = {
    "thank you",
    "thank you.",
    "thanks for watching",
    "thanks for watching!",
    "you",
    "bye",
    "bye.",
    "subtitles by the amara.org community",
    "ধন্যবাদ",
    "ধন্যবাদ।",
    "সাবটাইটেল",
    "আমার চ্যানেলটি সাবস্ক্রাইব করুন",
    "subscribe",
}


def add_cuda_dlls() -> None:
    """CTranslate2 needs cuBLAS/cuDNN DLLs; the CUDA torch wheel ships them in torch/lib."""
    try:
        import torch

        lib = Path(torch.__file__).parent / "lib"
        if lib.is_dir():
            os.add_dll_directory(str(lib))
            os.environ["PATH"] = f"{lib}{os.pathsep}{os.environ.get('PATH', '')}"
    except Exception:  # CPU-only installs simply skip this
        pass


@dataclass
class Transcript:
    text: str
    language: str
    language_probability: float
    seconds: float


def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if text.lower().strip(" .!।") in {h.strip(" .!।") for h in HALLUCINATIONS}:
        return ""
    return text


def plausible(segment) -> bool:
    """Drop segments Whisper invents from noise or music (low confidence or looping)."""
    if segment.no_speech_prob > 0.6 and segment.avg_logprob < -0.7:
        return False
    return segment.avg_logprob > -1.5 and segment.compression_ratio < 2.6


def choose_language(probs: dict[str, float], allowed: list[str]) -> tuple[str, float]:
    """Pick among the allowed languages from Whisper's language probabilities.

    For English plus one other language, anything not clearly English is the other one:
    Whisper hears short Bangla commands as Hindi or Gujarati far more often than as
    Bangla (in our spoken-command test, Bangla won only 44% of the time against English
    by plain argmax, and 95% with this rule; FLEURS sentences: 100% either way).
    """
    if "en" in allowed and len(allowed) == 2:
        other = next(code for code in allowed if code != "en")
        english = probs.get("en", 0.0)
        return ("en", english) if english >= 0.5 else (other, 1.0 - english)
    scores = {code: p for code, p in probs.items() if code in allowed}
    if not scores:
        return allowed[0], 0.0
    total = sum(scores.values()) or 1.0
    best = max(scores, key=scores.get)
    return best, scores[best] / total


class _Multilingual:
    """Delegates to a CTranslate2 Whisper model but reports it as multilingual."""

    is_multilingual = True

    def __init__(self, model) -> None:
        self._model = model

    def __getattr__(self, name):
        return getattr(self._model, name)


class CT2Whisper:
    """A Whisper model in CTranslate2 format, run by faster-whisper."""

    def __init__(
        self,
        model: str,
        device: str = "cuda",
        compute_type: str = "int8_float16",
        beam_size: int = 1,
    ) -> None:
        add_cuda_dlls()
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        self.beam_size = beam_size
        config = Path(model) / "config.json"
        if not self.model.model.is_multilingual and config.exists():
            if len(json.loads(config.read_text("utf-8")).get("lang_ids", [])) > 1:
                # Fine-tunes with a replaced tokenizer (BuzzASR) keep Whisper's language
                # tokens, but CTranslate2 infers "English-only" from the vocabulary layout,
                # which would make faster-whisper drop the <|bn|> prompt.
                self.model.model = _Multilingual(self.model.model)

    def detect(self, audio: np.ndarray, allowed: list[str]) -> tuple[str, float]:
        """Most likely language among `allowed`, with its probability."""
        _, _, probs = self.model.detect_language(audio)
        return choose_language(dict(probs), allowed)

    def transcribe(self, audio: np.ndarray, language: str) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language=language,
            beam_size=self.beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
        )
        return " ".join(s.text.strip() for s in segments if plausible(s))


class HFWhisper:
    """A Whisper fine-tune in Hugging Face format (used for Bangla-only models)."""

    def __init__(self, model: str, device: str = "cuda") -> None:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = WhisperProcessor.from_pretrained(model)
        self.model = WhisperForConditionalGeneration.from_pretrained(model, dtype=dtype)
        self.model.to(self.device).eval()
        self.dtype = dtype

    def transcribe(self, audio: np.ndarray, language: str = "bn") -> str:
        features = self.processor(audio, sampling_rate=16_000, return_tensors="pt").input_features
        features = features.to(self.device, self.dtype)
        with self.torch.inference_mode():
            ids = self.model.generate(
                features,
                num_beams=1,
                max_new_tokens=200,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
            )
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]


def load_model(name: str, device: str, compute_type: str, beam_size: int):
    """CTranslate2 directories/repos load with faster-whisper, others with transformers."""
    path = Path(name)
    is_ct2 = (path / "model.bin").exists() or "ct2" in name.lower() or "faster-whisper" in name
    if is_ct2:
        return CT2Whisper(name, device, compute_type, beam_size)
    return HFWhisper(name, device)


class Transcriber:
    def __init__(
        self,
        whisper_model: str,
        bangla_model: str = "",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        beam_size: int = 1,
        languages: list[str] | None = None,
    ) -> None:
        from sohayok.models import resolve_bangla_model

        bangla_model = resolve_bangla_model(bangla_model)
        self.languages = languages or ["bn", "en"]
        self.multi = CT2Whisper(whisper_model, device, compute_type, beam_size)
        self.bangla = (
            load_model(bangla_model, device, compute_type, beam_size) if bangla_model else None
        )
        self.names = {"multi": whisper_model, "bangla": bangla_model or whisper_model}

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> Transcript:
        from sohayok.audio import normalize_level

        started = time.monotonic()
        audio = normalize_level(audio.astype(np.float32))
        probability = 1.0
        if language is None:
            language, probability = self.multi.detect(audio, self.languages)
        model = self.bangla if language == "bn" and self.bangla is not None else self.multi
        text = clean(model.transcribe(audio, language))
        return Transcript(text, language, probability, time.monotonic() - started)
