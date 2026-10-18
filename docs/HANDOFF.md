# Hand-off: where Sohayok stands and where to pick up

*Written 2026-10-19 at the end of the first build (v0.2).*

## What this is

**Sohayok (সহায়ক)** is a Siri-style voice assistant for the user's Windows PC. It runs in the
background from sign-in until shutdown, wakes when you say **"Tammy"**, understands **Bangla,
English, Banglish, and mixed speech**, does the job on the desktop (apps, windows, typing,
volume, YouTube, search, files, reminders, GitHub, …), and answers aloud in the same language.

Design goals, as the user set them:

- Feels like Siri: say the name and it listens, with no clicking.
- Light: a small **decision model** (Laya-Bangla) picks the action; a small or free
  **answer model** fills in details and replies. The answer model can be local (Ollama) or
  any API the user picks in Settings (OpenAI, Gemini, GLM, OpenRouter, Anthropic, custom).
- A real desktop app (tray icon, pop-up window, face icon), not a web page.
- Safe: asks before anything that can't be undone; never types passwords.

## Where things are

| What | Where |
| --- | --- |
| Code | `E:\projects\sohayok`, GitHub `Ashfaq1306/sohayok` (`main`; this work is on `release/v0.2-tammy-lite`) |
| Decision model | Hugging Face [`ashfaq1306/sohayok-laya-actions`](https://huggingface.co/ashfaq1306/sohayok-laya-actions) @ `3fc68f6b`; local copy `%APPDATA%\Sohayok\models\laya-sohayok-actions` |
| Bangla speech model | `%APPDATA%\Sohayok\models\BuzzASR-bengali-ct2` (built by `sohayok setup`) |
| Wake word model | `src/sohayok/assets/tammy.onnx` |
| Settings, logs, reminders | `%APPDATA%\Sohayok\` (`config.toml`, `settings.json`, `sohayok.log`, `console.log`) |
| The user's own voice data | `%APPDATA%\Sohayok\recordings\` (last 20 commands) and `recordings\wake\` (wake attempts) |
| Benchmark data and outputs (not in git, ~9 GB) | `E:\projects\sohayok-data` |
| Start-up shortcuts | Startup folder, Start menu, and Desktop `Sohayok.lnk` (`sohayok install`) |

## Architecture in one screen

```
wake.py      openWakeWord embeddings → tammy.onnx (CPU, 12.5×/s); ≥0.8 wakes at once,
             0.3–0.8 asks Whisper (desktop.heard_wake_name)
audio.py     Silero VAD recorder, level normalisation, kept recordings
stt.py       Whisper large-v3-turbo picks bn/en (anything not clearly English = Bangla);
             BuzzASR Bengali (CTranslate2 int8) transcribes Bangla
router.py    instant rules for everyday commands (Bangla, Banglish, English)
decide.py    Laya-Bangla fine-tune: one "which action?" question over 28 actions
brain/lite.py     decision → the action's 1–7 tools → answer model fills arguments →
                  built-in reply or short summary; unsure (<0.7) or multi-step → agent
brain/providers.py  Ollama / OpenAI-compatible / Anthropic chat, model listing, keys in
                    Windows Credential Manager
brain/agent.py, ollama.py, claude.py   full tool-calling agents (fallback)
safety.py    confirmations (voice or buttons) for risky tools
tools/       48 desktop tools
assistant.py the loop: listen → understand → act → answer, events for the window
popup.py, desktop.py, server.py, ui/index.html   tray, pop-up, local API, the HUD window
training/    decider data + fine-tune, wake word data + training
bench/       stt, brain, voice, wake benchmarks; docs/benchmarks.md has the numbers
```

## Verified

- 153 unit tests pass (`uv run pytest -q`); read-only desktop checks pass.
- Typed commands: 91.4% right action (Laya + qwen3.5:2b), Bangla 90.8%.
- Speech: Bangla FLEURS 21.6% WER; spoken commands 15.9% (Bangla), 9.9% (English); quiet
  mic as good as normal.
- Wake word on synthetic held-out voices: 97.7% instant, 0 false wakes in 1 h of Bangla.
- Real use by the user (2026-10-19 log): their commands were transcribed well ("ভলিউম
  বাড়াও", "আমি আসলে জানতে চাচ্ছি যে কাইনেটিক এনার্জি…"); the detector fired on their
  "Tammy" but Whisper heard "Demi" and refused, now fixed (accepted spellings + instant
  wake). **Not yet confirmed with their voice after the fix.**

## Known problems and limits

1. **Wake word with the user's real voice** is the most important open item (above). If it
   still misses: lower `wake_threshold`, and retrain on `recordings\wake\` (see next steps).
2. **GPU sharing.** Another training job on the 8 GB card pushed Sohayok's models out of GPU
   memory and one command took 49 s to transcribe. Idle, Sohayok now holds ~2.5 GB (speech);
   the answer model unloads after 2 min and Laya runs on the CPU.
3. **Spoken replies need the internet** (Microsoft Edge voices). No offline Bangla voice.
4. **API providers** (OpenAI, Gemini, GLM, OpenRouter, Anthropic) are unit-tested against fake
   servers only; no real key was used. Default model names are suggestions; "Load models"
   lists the real ones.
5. The **Claude computer-use** path and the end-to-end **spoken** benchmark with the lite
   pipeline have not been re-run since the last changes.
6. Changing the wake word in Settings takes effect after restarting Sohayok.

## Next steps, in order

1. **Try "Tammy" with the real voice** after this update. Look at `%APPDATA%\Sohayok\
   sohayok.log` (`wake check heard …`) and `recordings\wake\*.json`.
2. **Train the detector on the user's voice:** add their kept wake clips (and a few dozen
   recorded on purpose) as positives in `training/wake_word.py`, rebuild features (TTS is
   cached), retrain, and re-run `bench/wake.py`.
3. **Faster replies:** stream TTS (start speaking at the first sentence), keep the answer
   model loaded while the window is open, and measure end-to-end latency with the GPU free.
4. **Offline Bangla voice** (for example an MMS/VITS Bangla model) as a fallback.
5. **Installer:** a one-file Windows installer so `uv` is not needed.
6. Re-run `bench/voice.py` with the lite pipeline and update `docs/benchmarks.md`.

## Should it be rewritten in Rust?

Short answer: **not now; it would not make it noticeably faster.** Where a voice command's
time goes today (GPU free):

| Step | Time | Runs in |
| --- | --- | --- |
| Wake word | a few ms per 80 ms frame; instant wake | ONNX Runtime (C++) |
| End of speech (silence wait) | 1.0 s by design | — |
| Speech recognition | ~0.4–0.6 s | CTranslate2 (C++/CUDA) |
| Decision (Laya) | ~0.2 s | PyTorch (C++) on the CPU |
| Answer model | ~0.5–2 s | llama.cpp via Ollama (C++/CUDA) or an API |
| Spoken reply | ~0.5–1 s | Microsoft's servers |

Every slow step already runs in compiled C++/CUDA code; Python only passes data between them
(a few milliseconds). Rust would call the same engines (ONNX Runtime, CTranslate2 or
whisper.cpp, llama.cpp), so the waits would be the same. Memory is dominated by the models
(GBs), not the program (~300 MB). A full rewrite would take weeks to months and would need
Laya exported to ONNX and new Windows automation code.

What Rust *would* improve: start-up time, a smaller single `.exe`, and a lighter window.
If wanted later, the sensible path is a **Tauri (Rust) shell** for the tray and window that
talks to the existing Python engine, not a full rewrite. The bigger speed wins are in the list
above: streaming speech output, a shorter silence wait, and keeping the GPU free.

## Picking up in a new session

```powershell
cd E:\projects\sohayok
git pull
uv sync --all-extras
uv run pytest -q
uv run sohayok doctor
```

Read this file, `README.md`, and `docs/benchmarks.md` first. Commits are authored by
Ashfaq1306 with no co-author line.
