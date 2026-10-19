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
wake.py      openWakeWord embeddings → tammy-personal.onnx (from `sohayok train-wake`,
             in %APPDATA%\Sohayok\models, its thresholds in the .json) or the shared
             tammy.onnx; needs ≥0.2 s of speech; instant above the instant threshold, else
             the Bangla model then Whisper must hear the name (desktop.heard_wake_name).
             Saying it while Sohayok talks stops the speech and listens.
audio.py     Silero VAD recorder, level normalisation, kept recordings
stt.py       VoxLingua107 ECAPA (CPU), plus Whisper when unsure, picks bn/en
             (bangla_or_english); BuzzASR Bengali (CTranslate2 int8) transcribes Bangla,
             Whisper large-v3-turbo English; silence is never decoded
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

- 187 unit tests pass (`uv run pytest -q`); read-only desktop checks pass.
- Typed commands: 91.4% right action (Laya + qwen3.5:2b), Bangla 90.8%.
- Speech: Bangla FLEURS 21.6% WER; spoken commands 15.9% (Bangla), 9.9% (English); quiet
  mic as good as normal.
- Wake word on synthetic held-out voices: 97.7% instant, 0 false wakes in 1 h of Bangla.
- Real use by the user (2026-10-19 log): their commands were transcribed well ("ভলিউম
  বাড়াও", "আমি আসলে জানতে চাচ্ছি যে কাইনেটিক এনার্জি…"); the detector fired on their
  "Tammy" but Whisper heard "Demi" and refused, now fixed (accepted spellings + instant
  wake). **Not yet confirmed with their voice after the fix.**
- Round 4 (2026-10-19 evening, after a full day of use):
  - The other project's 13 processes also kept the CPU at 97%: the decider took 1.3-4.8 s.
    Sohayok now runs at high CPU priority (`assistant.cpu_priority`): 0.23 s.
  - Short English ("Yes" -> "ইয়েস") is checked with Whisper; English commands 31/31.
  - Commands end after 1.2 s of silence (the user was cut off at 0.7 s).
  - Replies are spoken sentence by sentence: first word after 1.1 s instead of 5.3 s for a
    150-character answer (`tts.sentences`).
  - New: Bluetooth / Wi-Fi on-off and connecting a device by name or kind
    (`tools/devices.py`; the device picker was checked against the user's real Settings page:
    JBL Charge 5 connected, Sony WF-1000XM5 paired; nothing was switched); reading Windows
    notifications from the local notification database (`tools/notifications.py`, checked on
    the real database without printing contents); WhatsApp reply / unread / read chat.
  - WhatsApp search now tries English spellings first ("পিয়াস ভাই" -> pias, piyas, …);
    WhatsApp is still **not signed in** on this PC, and its tools say so before asking.
  - Tools can raise `ToolError(..., spoken_bn=, spoken_en=)` and declare a `check` run before
    the confirmation question.
  - "Jarvis" spellings are accepted by the wake check (the user started calling it that).
- Round 3 (2026-10-19, after the user's second real session):
  - Their speech mixes English words into Bangla; recognition order changed to Bangla model
    first (docs/benchmarks.md, "Update 2"). Their real commands: 9 of 9 right language.
  - Another project's training jobs kept the GPU at 100%: recognition took 8–19 s. Sohayok now
    takes real-time GPU priority while recognising (`sohayok/gpu.py`): 0.45 s median.
  - New actions: This PC / drives / Recycle Bin, navigate the Explorer window in front, new
    folder "here" (`tools/files.py`); close a browser tab by name (`browser.close_tab`, live
    tested on the user's Chrome with a tab it opened itself); WhatsApp open chat / send / call
    (`tools/whatsapp.py`, picked by keyword in `brain/lite.keyword_action`).
  - **WhatsApp is not signed in on this PC**, so its tools are untested against the real
    window; the control names (search box, "Type a message", "Voice call") are WhatsApp
    Web's and must be checked once it is signed in.
  - `bench/dry_run.py`: voice → action without touching the desktop: 32 of 33.
  - Tests no longer write into the real `%APPDATA%\Sohayok` (they had filled `recordings\`
    with fake 1 s "ভলিউম বাড়াও" clips).
- v0.2 follow-up (same day), from replaying the user's kept recordings:
  - Whisper gave their Bangla p(bn) = 0.00; half their Bangla commands went to the English
    recognizer. The new language step gets 10 of 10 (docs/benchmarks.md, "Bangla or English?").
  - The Bangla model looped for 13–17 s on silent recordings; silence is now skipped.
  - Their "Tammy" was refused by English Whisper ("Demi", "Dummy", "Can we?") and 31 of 40
    wake candidates were keyboard/room noise. Personal model + speech gate + Bangla-first
    check: 9 of 9 unseen "Tammy"s wake in cross-validation, 0 of 31 noises.
  - Personal model trained: `%APPDATA%\Sohayok\models\tammy-personal.onnx` (+ .json).
  - Still not tried live by the user after these changes.

## Known problems and limits

1. **Wake word with the user's real voice** is still the most important open item. The
   personal model is trained from only 9 real clips; if it misses, run
   `sohayok train-wake --record 10`. If it wakes by itself, delete
   `models\tammy-personal.onnx` to go back to the shared model.
   `train-wake` needs the shared training features in `%APPDATA%\Sohayok\wake-features`
   (copied from `E:\projects\sohayok-data\wake-tammy`; rebuild with
   `python -m sohayok.training.wake_word build`).
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
7. **App names said in Bangla** are sometimes misspelled ("ক্রমে" for Chrome, "জিমিনিয়াই" for
   Gemini). A vocabulary hint (initial prompt / hotwords) fixed some and broke others (the
   hint leaked into one transcript), so it is not used.
8. The user's **English** with their accent is untested (no English recordings yet).

## Next steps, in order

1. **WhatsApp:** once the user has signed WhatsApp in, inspect its window (search box,
   chat list items and their unread text, "message-in"/"message-out" rows, message box,
   call buttons) and test `whatsapp_open_chat`, `whatsapp_unread`, `whatsapp_read_chat`
   live; test sending only with the user's go-ahead (their own "Message yourself" chat).
   The control names in `tools/whatsapp.py` are WhatsApp Web's and are unverified.
   Also ask what the user meant by controlling the "fan".
2. **Try "Tammy" and commands with the real voice.** The log now has `heard … in … s` and
   `handled in … s` lines per command; replay wake clips with `python -m sohayok.bench.wake_user`
   and commands with `python -m sohayok.bench.dry_run`.
3. **More of the user's voice:** `sohayok train-wake --record 10` (guided), then retrain as
   daily-use clips accumulate. Also keep a few of their English commands to test English.
4. **Faster replies:** stream TTS (start speaking at the first sentence), keep the answer
   model loaded while the window is open, and measure end-to-end latency with the GPU free.
5. **Offline Bangla voice** (for example an MMS/VITS Bangla model) as a fallback.
6. **Installer:** a one-file Windows installer so `uv` is not needed.
7. Re-run `bench/voice.py` with the lite pipeline and update `docs/benchmarks.md`.

## Should it be rewritten in Rust?

Short answer: **not now; it would not make it noticeably faster.** Where a voice command's
time goes today (GPU free):

| Step | Time | Runs in |
| --- | --- | --- |
| Wake word | a few ms per 80 ms frame; instant wake | ONNX Runtime (C++) |
| End of speech (silence wait) | 0.7 s by design | — |
| Language + speech recognition | ~0.3–0.7 s | PyTorch (CPU) + CTranslate2 (C++/CUDA) |
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
