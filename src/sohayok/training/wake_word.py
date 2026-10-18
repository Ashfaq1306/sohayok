"""Train the "Tammy" wake word on openWakeWord's frozen speech embeddings.

Positives: "Tammy" / "Hey Tammy" / "টামি" from Microsoft neural voices in many accents,
augmented with gain changes (quiet microphones), noise, and echo. Negatives: real speech from
FLEURS (Bangla and English) plus voices saying look-alike words. Voices and recordings used
for validation are held out from training. The result is a tiny ONNX classifier that
openWakeWord runs on the CPU, 12.5 times a second.
"""

from __future__ import annotations

import asyncio
import io
import json
import random
from pathlib import Path

import numpy as np

RATE = 16_000
CLIP = 32_000  # 2 s windows; the classifier sees the last 16 embedding frames (~1.3 s)
POSITIVE_PHRASES = {
    "en": ["Tammy", "Hey Tammy", "Tammy!", "Tammy?"],
    "bn": ["টামি", "ট্যামি", "হেই টামি"],
}
CONFUSABLE = {
    "en": [
        "Tommy",
        "tummy",
        "Timmy",
        "Sammy",
        "Jimmy",
        "mommy",
        "Amy",
        "timer",
        "Tom",
        "Tim",
        "tamil",
        "Danny",
        "Tony",
        "tiny",
        "Tammany",
        "Miami",
        "salami",
        "family",
        "Jamie",
    ],
    "bn": ["তুমি", "আমি", "টমেটো", "টাকা", "ট্যাক্সি", "তামা", "সামনে", "টিম", "মামি", "দামি", "জামি"],
}


async def _voices() -> list[dict]:
    import edge_tts

    return await edge_tts.list_voices()


CACHE: Path | None = None  # set by build(): synthesized clips are kept so reruns are quick


def tts(text: str, voice: str, rate: str = "+0%", pitch: str = "+0Hz") -> np.ndarray:
    if CACHE is not None:
        import hashlib

        key = hashlib.sha1(f"{text}|{voice}|{rate}|{pitch}".encode()).hexdigest()
        path = CACHE / f"{key}.npy"
        if path.exists():
            return np.load(path)
        audio = _tts(text, voice, rate, pitch)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(path, audio)
        return audio
    return _tts(text, voice, rate, pitch)


def _tts(text: str, voice: str, rate: str = "+0%", pitch: str = "+0Hz") -> np.ndarray:
    import edge_tts
    import soundfile as sf
    from scipy.signal import resample_poly

    async def run() -> bytes:
        audio = bytearray()
        async for chunk in edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).stream():
            if chunk["type"] == "audio":
                audio.extend(chunk["data"])
        return bytes(audio)

    samples, rate_hz = sf.read(io.BytesIO(asyncio.run(run())), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return resample_poly(samples, RATE, rate_hz).astype(np.float32)


def pick_voices(seed: int) -> dict[str, list[str]]:
    voices = asyncio.run(_voices())
    english = sorted(v["ShortName"] for v in voices if v["Locale"].startswith("en-"))
    south_asian = sorted(
        v["ShortName"]
        for v in voices
        if v["Locale"].split("-")[0] in {"bn", "hi", "ur", "ta", "te", "mr", "gu", "ne"}
    )
    rng = random.Random(seed)
    rng.shuffle(english)
    rng.shuffle(south_asian)
    return {"en": english, "bn": south_asian}


def place(audio: np.ndarray, rng: random.Random, end_at_tail: bool) -> np.ndarray:
    """Put `audio` in a 2 s window; positives end in the last 0.3 s, like live streaming."""
    audio = audio[:CLIP]
    out = np.zeros(CLIP, dtype=np.float32)
    if end_at_tail:
        end = CLIP - rng.randint(0, int(0.3 * RATE))
        start = max(0, end - len(audio))
        out[start:end] = audio[len(audio) - (end - start) :]
    else:
        start = rng.randint(0, max(0, CLIP - len(audio)))
        out[start : start + len(audio)] = audio
    return out


def change_speed(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    """Speed and pitch perturbation by resampling (0.85-1.15x), a standard speech augmentation."""
    from scipy.signal import resample_poly

    factor = rng.choice([85, 90, 95, 100, 100, 105, 110, 115])
    return audio if factor == 100 else resample_poly(audio, 100, factor).astype(np.float32)


def augment(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    audio = audio.copy()
    if rng.random() < 0.5:  # room echo: a short exponentially decaying impulse response
        ir = np.zeros(int(RATE * rng.uniform(0.05, 0.3)), dtype=np.float32)
        ir[0] = 1.0
        taps = rng.randint(5, 30)
        for _ in range(taps):
            i = rng.randint(1, len(ir) - 1)
            ir[i] += rng.uniform(0.05, 0.5) * np.exp(-6 * i / len(ir))
        audio = np.convolve(audio, ir)[: len(audio)].astype(np.float32)
    peak = float(np.abs(audio).max()) or 1.0
    audio = audio / peak * (10 ** rng.uniform(-2.2, -0.15))  # -44 dB (very quiet mic) to 0 dB
    if rng.random() < 0.7:
        noise = np.random.default_rng(rng.randint(0, 2**31)).standard_normal(len(audio))
        if rng.random() < 0.5:  # pinkish noise
            noise = np.cumsum(noise)
            noise -= np.convolve(noise, np.ones(64) / 64, mode="same")
        noise = (
            noise
            / (np.abs(noise).max() + 1e-9)
            * float(np.abs(audio).max())
            * 10 ** (-rng.uniform(5, 30) / 20)
        )
        audio = audio + noise.astype(np.float32)
    return np.clip(audio, -1, 1).astype(np.float32)


def fleurs_windows(path: Path, max_windows: int, rng: random.Random) -> list[np.ndarray]:
    import pyarrow.parquet as pq
    import soundfile as sf

    windows = []
    for row in pq.read_table(path, columns=["audio"]).to_pylist():
        audio, _ = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        for start in range(0, max(1, len(audio) - CLIP), RATE):  # 2 s windows, 1 s hop
            windows.append(
                audio[start : start + CLIP].copy()
                if len(audio) >= CLIP
                else place(audio, rng, False)
            )
            if len(windows) >= max_windows:
                return windows
    return windows


def build(fleurs: Path, out: Path, seed: int = 20261018) -> dict:
    """Synthesize and embed the training and validation sets; writes .npy feature files."""
    from openwakeword.utils import AudioFeatures

    global CACHE
    CACHE = out / "tts_cache"
    rng = random.Random(seed)
    voices = pick_voices(seed)
    split = {lang: (v[: int(len(v) * 0.8)], v[int(len(v) * 0.8) :]) for lang, v in voices.items()}
    data: dict[str, list[np.ndarray]] = {
        "train_pos": [],
        "train_neg": [],
        "val_pos": [],
        "val_neg": [],
    }
    for part, index in (("train", 0), ("val", 1)):
        for lang, phrases in POSITIVE_PHRASES.items():
            for voice in split[lang][index]:
                for phrase in phrases:
                    for rate in ("-15%", "+0%", "+20%"):
                        try:
                            audio = tts(phrase, voice, rate, rng.choice(["-10Hz", "+0Hz", "+10Hz"]))
                        except Exception:
                            continue
                        copies = 6 if part == "train" else 1
                        for _ in range(copies):
                            varied = change_speed(audio, rng) if part == "train" else audio
                            data[f"{part}_pos"].append(augment(place(varied, rng, True), rng))
            for voice in split[lang][index][: max(4, len(split[lang][index]) // 2)]:
                for word in CONFUSABLE[lang]:
                    try:
                        audio = tts(word, voice)
                    except Exception:
                        continue
                    data[f"{part}_neg"].append(augment(place(audio, rng, True), rng))
        print(
            part,
            "positives",
            len(data[f"{part}_pos"]),
            "confusables",
            len(data[f"{part}_neg"]),
            flush=True,
        )
    data["train_neg"] += [
        augment(w, rng) for w in fleurs_windows(fleurs / "bn_in" / "test.parquet", 6000, rng)
    ]
    val_speech = fleurs_windows(fleurs / "en_us" / "test.parquet", 4000, rng)
    data["train_neg"] += [augment(w, rng) for w in val_speech[:2000]]
    data["val_neg"] += [augment(w, rng) for w in val_speech[2000:]]
    silence = [augment(np.zeros(CLIP, dtype=np.float32) + 1e-4, rng) for _ in range(300)]
    data["train_neg"] += silence
    features = AudioFeatures()
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, clips in data.items():
        pcm = (np.clip(np.stack(clips), -1, 1) * 32767).astype(
            np.int16
        )  # the model wants 16-bit PCM
        emb = features.embed_clips(pcm, batch_size=64)[:, -16:, :].astype(np.float32)
        np.save(out / f"{name}.npy", emb)
        summary[name] = list(emb.shape)
    summary["val_neg_hours"] = round((len(val_speech) - 2000) / 3600, 3)  # 1 s hop windows
    (out / "summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
    return summary


def train(
    features: Path,
    output: Path,
    steps: int = 8000,
    seed: int = 20261018,
    max_false_per_hour: float = 10.0,
) -> dict:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    load = lambda name: torch.tensor(np.load(features / f"{name}.npy"))  # noqa: E731
    xp, xn, vp, vn = load("train_pos"), load("train_neg"), load("val_pos"), load("val_neg")
    model = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(16 * 96, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.LayerNorm(64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )
    hours = json.loads((features / "summary.json").read_text("utf-8"))["val_neg_hours"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.BCELoss(reduction="none")
    best = {"score": -1.0}
    for step in range(1, steps + 1):
        pi = torch.randint(0, len(xp), (128,))
        ni = torch.randint(0, len(xn), (384,))
        x = torch.cat([xp[pi], xn[ni]])
        y = torch.cat([torch.ones(128), torch.zeros(384)])
        p = model(x).squeeze(1)
        # False wake-ups annoy more than a missed one, so negatives weigh more as training goes on.
        weight = torch.where(y == 1, 1.0, 1.0 + 9.0 * step / steps)
        loss = (loss_fn(p, y) * weight).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            model.eval()
            with torch.no_grad():
                pp, pn = model(vp).squeeze(1), model(vn).squeeze(1)
            model.train()
            # Whisper double-checks every wake, so allow ~10 detector false accepts per hour
            # of speech and keep the checkpoint and threshold that catch the most "Tammy"s.
            for threshold in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
                recall = float((pp >= threshold).float().mean())
                false = int((pn >= threshold).sum())
                if false / hours > max_false_per_hour:
                    continue
                score = recall - 0.001 * false
                if score > best["score"]:
                    best = {
                        "score": score,
                        "step": step,
                        "threshold": threshold,
                        "recall": round(recall, 3),
                        "false_accepts": false,
                        "false_per_hour": round(false / hours, 1),
                        "state": {k: v.clone() for k, v in model.state_dict().items()},
                    }
            print(
                f"step {step} loss {loss.item():.4f} best {best.get('recall')} recall, "
                f"{best.get('false_per_hour')}/h false at {best.get('threshold')}",
                flush=True,
            )
    model.load_state_dict(best.pop("state"))
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros(1, 16, 96),
        str(output),
        input_names=["x"],
        output_names=["score"],
        opset_version=13,
        dynamo=False,
    )
    return best


if __name__ == "__main__":
    import sys

    if sys.argv[1] == "build":
        print(json.dumps(build(Path(sys.argv[2]), Path(sys.argv[3])), indent=2))
    else:
        print(json.dumps(train(Path(sys.argv[2]), Path(sys.argv[3])), indent=2))
