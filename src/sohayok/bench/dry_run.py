"""End-to-end check without touching the desktop: voice -> words -> decision -> tool call.

Every tool is replaced by a recorder, so nothing is opened, closed, sent, or called; the
speech models, the decider, and the answer model chosen in Settings are the real ones.

    python -m sohayok.bench.dry_run            # spoken test commands + the user's recordings
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os
import sys
import time

import numpy as np

# (text, voice, acceptable first tools)
BN, BN2, EN_IN, EN_US = (
    "bn-BD-NabanitaNeural",
    "bn-BD-PradeepNeural",
    "en-IN-PrabhatNeural",
    "en-US-AriaNeural",
)
CASES = [
    ("দিস পিসি খোলো", BN, {"open_path"}),
    ("ডি ড্রাইভে যাও", BN2, {"open_path"}),
    ("ডেস্কটপে হিসাব নামে একটা ফোল্ডার বানাও", BN, {"create_folder"}),
    ("একটা নতুন ফোল্ডার বানাও", BN2, {"create_folder"}),
    ("রহিমকে হোয়াটসঅ্যাপে কল দাও", BN, {"whatsapp_call"}),
    ("আম্মুকে ভিডিও কল দাও", BN2, {"whatsapp_call"}),
    ("রহিমকে হোয়াটসঅ্যাপে লেখো আমি দশ মিনিটে আসছি", BN, {"whatsapp_send"}),
    ("হোয়াটসঅ্যাপে করিম ভাইয়ের চ্যাটটা খোলো", BN2, {"whatsapp_open_chat"}),
    ("ক্রোম খোলো", BN, {"open_app"}),
    ("ইউটিউবে আইয়ুব বাচ্চুর গান চালাও", BN2, {"play_youtube"}),
    (
        "ইউটিউবের ট্যাবটা বন্ধ করো",
        BN,
        {"close_tab", "press_keys", "window_action", "close_app", "switch_to_window"},
    ),
    ("ভলিউম বাড়াও", BN2, {"volume"}),
    ("ডাউনলোড ফোল্ডার খোলো", BN, {"open_path"}),
    ("আজকে ঢাকার আবহাওয়া কেমন", BN2, {"weather"}),
    ("নোটপ্যাড খুলে লেখো আমি ভালো আছি", BN, {"open_app"}),
    ("ক্রোমে নতুন ট্যাব খোলো", BN2, {"press_keys", "open_app", "switch_to_window"}),
    ("একটা স্ক্রিনশট নাও", BN, {"take_screenshot"}),
    ("open this PC", EN_IN, {"open_path"}),
    ("go to D drive", EN_IN, {"open_path"}),
    ("create a new folder called reports on the desktop", EN_US, {"create_folder"}),
    ("call Rahim on WhatsApp", EN_IN, {"whatsapp_call"}),
    ("send a WhatsApp message to Karim saying I am on my way", EN_US, {"whatsapp_send"}),
    ("play some lo-fi music on YouTube", EN_IN, {"play_youtube"}),
    ("what's the weather like today", EN_US, {"weather"}),
    ("take a screenshot", EN_IN, {"take_screenshot"}),
    ("mute the sound", EN_US, {"volume"}),
    ("open Chrome", EN_IN, {"open_app"}),
]
# The user's own recordings (2026-10-19), with what they asked for.
USER = {
    "090347": {"open_app"},  # "Google Chrome open koro"
    "090420": {
        "close_tab",
        "press_keys",
        "window_action",
        "close_app",
        "switch_to_window",
    },  # close YouTube tab
    "090435": {"close_tab", "press_keys", "window_action", "close_app", "switch_to_window"},
    "090453": {"close_tab", "press_keys", "window_action", "close_app", "switch_to_window"},
    "090529": {"open_path"},  # "This PC open koro to"
    "090624": {"open_path"},  # "please open this PC"
    "030613": {"open_app", "open_path", "open_website"},  # open my Chrome project
    "030658": {"media", "close_app", "press_keys"},  # stop the song
}


def dry_registry():
    from sohayok.tools import load_all
    from sohayok.tools.registry import Registry

    calls: list[dict] = []
    registry = Registry()
    for tool in load_all().tools.values():

        def record(_name=tool.name, **args):
            calls.append({"name": _name, "args": args})
            return {"ok": True, "spoken_bn": "ঠিক আছে।", "spoken_en": "Done."}

        registry.add(dataclasses.replace(tool, func=record))
    return registry, calls


def main() -> None:
    from sohayok.assistant import Assistant, Command
    from sohayok.config import current
    from sohayok.desktop import build_lite
    from sohayok.stt import Transcriber
    from sohayok.training.wake_word import tts

    config = current()
    config.assistant.confirm_risky_actions = False  # dry run: nothing real happens anyway
    if len(sys.argv) > 2:  # compare answer models: python -m sohayok.bench.dry_run out.json MODEL
        config.brain.lite_model = sys.argv[2]
    speech = config.speech
    transcriber = Transcriber(
        speech.whisper_model,
        speech.bangla_model,
        speech.device,
        speech.compute_type,
        speech.beam_size,
        speech.languages,
        speech.language_detector,
    )
    transcriber.gpu_priority = speech.gpu_priority
    brain, info = build_lite(config)
    print("answer model:", info, flush=True)
    registry, calls = dry_registry()
    assistant = Assistant(config, registry, brain)

    inputs = [(text, voice, ok, tts(text, voice)) for text, voice, ok in CASES]
    folder = os.path.expandvars(r"%APPDATA%\Sohayok\recordings")
    import soundfile as sf

    for path in sorted(glob.glob(os.path.join(folder, "*.wav"))):
        key = os.path.basename(path)[9:15]
        if key in USER:
            inputs.append(
                (f"[your voice {key}]", "user", USER[key], sf.read(path, dtype="float32")[0])
            )

    transcriber.transcribe(inputs[0][3])  # warm up
    rows = []
    for text, voice, ok, audio in inputs:
        calls.clear()
        heard = transcriber.transcribe(np.asarray(audio, np.float32))
        started = time.monotonic()
        try:
            reply = assistant.handle(Command(heard.text, heard.language, "voice"))
        except Exception as exc:  # report, keep going
            reply = f"ERROR {exc}"
        think = time.monotonic() - started
        first = calls[0]["name"] if calls else "-"
        rows.append(
            {
                "said": text,
                "voice": voice,
                "heard": heard.text,
                "lang": heard.language,
                "stt_s": round(heard.seconds, 2),
                "think_s": round(think, 2),
                "calls": calls[:],
                "reply": reply,
                "pass": first in ok,
            }
        )
        mark = "PASS" if first in ok else "FAIL"
        print(
            f"{mark} {heard.seconds:4.2f}s+{think:4.2f}s [{heard.language}] {text} | heard: "
            f"{heard.text} | {[(c['name'], c['args']) for c in calls]}",
            flush=True,
        )
    passed = sum(r["pass"] for r in rows)
    print(
        f"\n{passed}/{len(rows)} right; median recognition "
        f"{np.median([r['stt_s'] for r in rows]):.2f}s, median understanding+action "
        f"{np.median([r['think_s'] for r in rows]):.2f}s"
    )
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", encoding="utf-8") as out:
            json.dump(rows, out, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
