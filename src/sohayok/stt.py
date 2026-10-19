"""Speech recognition: detect Bangla or English, then transcribe with the best model for it."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
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


def collapse_repeats(text: str) -> str:
    """Whisper sometimes loops ("close the tab and close the tab and close the tab ...");
    keep one copy of any phrase of up to 8 words repeated three or more times in a row."""
    words = text.split()
    for size in range(1, 9):
        i, out = 0, []
        while i < len(words):
            chunk = words[i : i + size]
            repeats = 1
            while words[i + repeats * size : i + (repeats + 1) * size] == chunk:
                repeats += 1
            if len(chunk) == size and repeats >= 3:
                out += chunk
                i += repeats * size
            else:
                out.append(words[i])
                i += 1
        words = out
    return " ".join(words)


def clean(text: str) -> str:
    # One Unicode form everywhere: the Bangla model writes "য়" as U+09DF, patterns use য + ়
    text = unicodedata.normalize("NFC", text)
    text = collapse_repeats(re.sub(r"\s+", " ", text).strip())
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


LANGUAGE_ID_REPO = "speechbrain/lang-id-voxlingua107-ecapa"
LANGUAGE_ID_REVISION = "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9"


class LanguageDetector:
    """VoxLingua107 ECAPA spoken-language classifier (21M parameters, on the CPU).

    Whisper's own language detection gave the user's Bangla 0% every time and sent half of
    their real Bangla commands to the English model. This one got all ten right, and 99-100%
    of FLEURS Bangla and English (even cut to 2.5 s), in 20-90 ms on the CPU.
    """

    def __init__(self, folder: Path | None = None) -> None:
        import warnings

        import torch

        logging.getLogger("speechbrain").setLevel(logging.WARNING)
        warnings.filterwarnings("ignore", module="speechbrain")
        warnings.filterwarnings("ignore", message=".*list_audio_backends.*")
        from speechbrain.inference.classifiers import EncoderClassifier
        from speechbrain.utils.fetching import FetchConfig, LocalStrategy

        for name in list(logging.root.manager.loggerDict):  # its modules set their own levels
            if name.startswith("speechbrain"):
                logging.getLogger(name).setLevel(logging.WARNING)

        if folder is None:
            from sohayok.config import app_dir

            folder = app_dir() / "models" / "lang-id-voxlingua107-ecapa"
        self.torch = torch
        self.model = EncoderClassifier.from_hparams(
            source=LANGUAGE_ID_REPO,
            savedir=str(folder),
            run_opts={"device": "cpu"},
            local_strategy=LocalStrategy.COPY,  # symlinks need admin rights on Windows
            fetch_config=FetchConfig(revision=LANGUAGE_ID_REVISION),
        )
        self.model.hparams.label_encoder.ignore_len()
        # labels look like "bn: Bengali"
        self.index = {
            label.split(":")[0]: i for i, label in self.model.hparams.label_encoder.ind2lab.items()
        }

    def scores(self, audio: np.ndarray):
        """Log-probabilities of all 107 languages (from at most the first 6 s)."""
        clip = self.torch.from_numpy(np.ascontiguousarray(audio[: 16_000 * 6], np.float32))
        with self.torch.inference_mode():
            return self.model.classify_batch(clip[None])[0][0]

    def margin(self, audio: np.ndarray, first: str, second: str) -> float:
        """How much more `first` than `second` (log-probability difference)."""
        scores = self.scores(audio)
        return float(scores[self.index[first]] - scores[self.index[second]])

    def detect(self, audio: np.ndarray, allowed: list[str]) -> tuple[str, float]:
        """Most likely language among `allowed`, with its probability among them."""
        scores = self.scores(audio)
        codes = [code for code in allowed if code in self.index]
        if not codes:
            return allowed[0], 0.0
        values = self.torch.stack([scores[self.index[code]] for code in codes])
        probs = self.torch.softmax(values, 0)
        best = int(probs.argmax())
        return codes[best], round(float(probs[best]), 3)


# Bangla-vs-English rule fitted on five sets (the user's real Bangla commands, synthetic
# Bangladeshi Bangla and Indian English commands, FLEURS Bangla and English cut to 3 s).
# Each detector fails where the other does not: Whisper hears the user's Bangla as English,
# VoxLingua hears Indian-accented English as Bangla (30%). Together: 97-100% on every set.
BANGLA_BIAS = 5.5
SURE_BANGLA = 14.7  # VoxLingua margins beyond these decide alone (Whisper cannot outvote them)
SURE_ENGLISH = -3.7
# Below this VoxLingua margin, Whisper's p(English) >= 0.9 decides English. The user's mixed
# Bangla scored 4.5-20 even when Whisper called it English (p up to 0.93).
UNSURE_BANGLA = 3.0


def bangla_or_english(margin: float, whisper_english: float | None) -> tuple[str, float]:
    """Combine VoxLingua's bn-minus-en margin with Whisper's p(English)."""
    if whisper_english is None:
        score = margin - BANGLA_BIAS
    else:
        p = min(max(whisper_english, 1e-4), 1 - 1e-4)
        score = margin - float(np.log(p / (1 - p))) - BANGLA_BIAS
    probability = 1 / (1 + float(np.exp(-abs(score))))
    return ("bn" if score > 0 else "en"), round(probability, 3)


def speech_seconds(audio: np.ndarray, enough: float = float("inf")) -> float:
    """Seconds of speech in the audio by Silero VAD, counting stops once `enough` is found
    (inf if the VAD is unavailable)."""
    try:
        from sohayok.audio import BLOCK, SpeechDetector

        vad = SpeechDetector()
    except Exception:
        return float("inf")
    found = 0.0
    for start in range(0, len(audio) - BLOCK + 1, BLOCK):
        if vad.probability(audio[start : start + BLOCK]) >= 0.5:
            found += BLOCK / 16_000
            if found >= enough:
                break
    return found


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
            # One greedy pass. Whisper's default retries at higher temperatures when output
            # looks repetitive; on silence the Bangla model then ran for up to 17 s.
            temperature=0.0,
            max_new_tokens=min(440, int(16 + 16 * len(audio) / 16_000)),
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
        language_detector: str = "voxlingua",
    ) -> None:
        from sohayok.models import resolve_bangla_model

        bangla_model = resolve_bangla_model(bangla_model)
        self.languages = languages or ["bn", "en"]
        self.multi = CT2Whisper(whisper_model, device, compute_type, beam_size)
        self.bangla = (
            load_model(bangla_model, device, compute_type, beam_size) if bangla_model else None
        )
        self.names = {"multi": whisper_model, "bangla": bangla_model or whisper_model}
        self.detector = None
        if language_detector == "voxlingua":
            try:
                self.detector = LanguageDetector()
            except Exception:
                log.exception("language detector unavailable; using Whisper's")

    def detect_language(self, audio: np.ndarray) -> tuple[str, float]:
        if self.detector is None:
            return self.multi.detect(audio, self.languages)
        if sorted(self.languages) != ["bn", "en"]:
            return self.detector.detect(audio, self.languages)
        margin = self.detector.margin(audio, "bn", "en")  # ~40 ms on the CPU
        if SURE_ENGLISH < margin < SURE_BANGLA:  # unsure: ask Whisper too (~190 ms)
            _, _, probs = self.multi.model.detect_language(audio)
            return bangla_or_english(margin, dict(probs).get("en", 0.0))
        return bangla_or_english(margin, None)

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> Transcript:
        from sohayok.audio import normalize_level
        from sohayok.gpu import priority

        started = time.monotonic()
        audio = normalize_level(audio.astype(np.float32))
        if speech_seconds(audio, enough=0.15) < 0.15:
            # nothing was said: decoding silence only invents text (and takes long)
            return Transcript("", language or self.languages[0], 0.0, time.monotonic() - started)
        with priority(getattr(self, "gpu_priority", "normal")):
            if language is None:
                text, language, probability = self._recognise(audio)
            else:
                text, probability = self._model_for(language).transcribe(audio, language), 1.0
        return Transcript(clean(text), language, probability, time.monotonic() - started)

    def _model_for(self, language: str):
        return self.bangla if language == "bn" and self.bangla is not None else self.multi

    def _recognise(self, audio: np.ndarray) -> tuple[str, str, float]:
        """Pick the language and transcribe, with as few GPU passes as possible.

        The user mixes English words into Bangla ("ইউটিউবের ট্যাবটা কেটে ফেলো"), which the
        Bangla model writes well and English Whisper turns into nonsense ("YouTube tab KT
        Fellow"). So anything not clearly English goes to the Bangla model first, and only a
        transcript with no Bangla grammar ("প্লিজ ওপেন ডিস পিসি") is redone in English.
        """
        from sohayok.lang import has_bangla_grammar

        bilingual = sorted(self.languages) == ["bn", "en"] and self.bangla is not None
        if self.detector is None or not bilingual:
            language, probability = self.detect_language(audio)
            return self._model_for(language).transcribe(audio, language), language, probability
        margin = self.detector.margin(audio, "bn", "en")  # ~40 ms on the CPU
        sure = round(1 / (1 + float(np.exp(-abs(margin - SURE_ENGLISH)))), 3)
        if margin < SURE_ENGLISH:
            return self.multi.transcribe(audio, "en"), "en", sure
        if margin < UNSURE_BANGLA:
            # the Bangla model turns English into real-looking Bangla here ("কোটি দিনের মধ্যেই
            # মার্চের…" for "what day of the week…"), so let Whisper settle it (~0.2 s)
            _, _, probs = self.multi.model.detect_language(audio)
            if dict(probs).get("en", 0.0) >= 0.9:
                return self.multi.transcribe(audio, "en"), "en", 0.9
        bangla = self.bangla.transcribe(audio, "bn")
        if not bangla.strip() or has_bangla_grammar(bangla, short_is_bangla=False):
            return bangla, "bn", sure
        if len(clean(bangla).split()) <= 2:
            # "ইয়েস", "স্টপ": too short for grammar; Whisper hears whether it was English
            _, _, probs = self.multi.model.detect_language(audio)
            if dict(probs).get("en", 0.0) < 0.5:
                return bangla, "bn", sure
        english = self.multi.transcribe(audio, "en")
        return (english, "en", 0.5) if clean(english) else (bangla, "bn", 0.5)
