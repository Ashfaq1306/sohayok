"""Replay the user's real spoken commands (as transcribed) through the brain, dry.

Every tool is replaced by a recorder, so nothing happens on the PC. Each case lists the tools
that would be a right first action ("" = a spoken answer with no tool is right).

    uv run python -m sohayok.bench.replay [out.json]
"""

from __future__ import annotations

import json
import sys
import time

# (what the user's command was transcribed as, acceptable first tools)
CASES: list[tuple[str, set[str]]] = [
    ("dishpc অন করো", {"open_path", "open_app"}),
    ("Open this PC", {"open_path", "open_app"}),
    ("নাও ওপেন নিউ ফোল্ডার অন গেম ই", {"create_folder"}),
    ("Could you play music for him from YouTube?", {"play_youtube"}),
    ("Pause the song", {"media"}),
    ("Shut down the computer.", {"power"}),
    ("এই আমার এতে ছিলেন একটা পিডিএফ আছে নাম ফেনি একদম ডেস্কটপে ওপেন করে", {"find_files", "open_path"}),
    ("তুমি এটিও পিডিএফ সেট জুম করে তো দেখি", {"press_keys"}),
    (
        "তুমি টু অনলাইনে মাইক্রোফোনের প্রাইস দেখাতো বাংলাদেশের ইস্টারট্যাক থেকে",
        {"web_search", "open_website", "web_lookup"},
    ),
    ("আমার গুগল ক্রোমটা অন করো", {"open_app"}),
    ("রুম রোজাটা না বন্ধ করে দো", {"close_app", "window_action"}),
    ("একটা সফট বাংলা গান ছাড়ো", {"play_youtube"}),
    ("ক্রমটা মিনিমেস করে রাখো", {"window_action"}),
    ("গানটি একটু বস করো", {"media"}),
    ("অন্য গ্রহের চাহিদা এখানটি ইউটিউব থেকে চালাও", {"play_youtube"}),
    ("Could you resume the song like resume where you did your boss", {"media"}),
    ("সাতেশকে পাঁচ দিয়ে ভাগ করলে কত হয়", {""}),
    ("ওয়াটস্যাপ থেকে পিএইচভাক একটা ফোন দাওত", {"whatsapp_call"}),
    ("ওয়াটস্যাপ ওপেন করো এবং ওয়াটস্যাপ থেকে পিয়াজ ভাইয়েকে ফোন দাও", {"whatsapp_call"}),
    ("কল পিয়াস ভাই ফোম ওয়াটস্যাপ", {"whatsapp_call"}),
    ("গানটাকে বন্ধ করে দাও", {"media"}),
    ("Play the song, baby we build this world", {"play_youtube"}),
    ("ভলিউম আপ", {"volume"}),
    ("volume up to 80%", {"volume"}),
    ("তুমি ভোলিউমটা ফর্টি পারসেনে নামায় নিয়ে আসো", {"volume"}),
    ("Play the song Believer", {"play_youtube"}),
    ("পাস দ্য প্রিভিয়াস সং", {"media"}),
    ("পূর্বের যতগুলোযা ইউটিউবের ট্যাব খোলা আছে সবগুলো ক্লোজ করে দেওয", {"close_tab"}),
    ("আমি ক্রমে ইউটিউবের পূর্বের ট্যাপগুলো কাটতে চাই", {"close_tab"}),
    ("এই গানটা পোস করো এবং এই গানটা চেইন্জ করে অ্যানালাভ গানটা চালাও", {"media", "play_youtube"}),
    ("এই গানটা শুরু থেকে শুরু করো", {"media", "press_keys", "switch_to_window"}),
    ("এখন ক্রোমটা মিনিমেস করো", {"window_action"}),
    ("আমার হার্ডসেপ ক্লোজ করো", {"close_app"}),
    ("Close my WhatsApp.", {"close_app"}),
    ("close my whatsapp notification", {"dismiss_notifications", "press_keys", "click_element"}),
    ("তোমার নাম কি", {""}),
    ("তুমি হচ্ছে আমার প্রিন্টার কোন কানেক্ষণ আছে দেখতে পারবা", {"list_printers"}),
    ("তুমি আমার পিসির ফুল কনফিগারেশনটা লেখা যাও", {"pc_specs", "system_status"}),
    ("Turn on the Google Chrome", {"open_app"}),
    ("ওয়াডস্যাপ ওপেন করে পিয়াশ ভাইকে কল দাও", {"whatsapp_call"}),
    ("আবার ওয়ার্সেপটা ওপেন করো", {"open_app"}),
    ("Open WhatsApp.", {"open_app"}),
    ("নাও কল পিয়াশ ভাই ফ্রম হোয়াটস্ অ্যাপ", {"whatsapp_call"}),
    ("Google Prome ওপেন করতো", {"open_app"}),
    ("গুলিক্রমটা মিনিমাস করো", {"window_action"}),
    # New abilities the user asked for.
    ("স্ক্রিনের ব্রাইটনেস কমাও", {"brightness"}),
    ("set brightness to 70 percent", {"brightness"}),
    ("ডার্ক মোড চালু করো", {"dark_mode"}),
    ("নোটিফিকেশনগুলো ক্লিয়ার করো", {"dismiss_notifications"}),
    ("হেডফোন ডিসকানেক্ট করো", {"bluetooth_device"}),
    ("ব্লুটুথ অফ করো", {"bluetooth_power"}),
    ("পিয়াস ভাইয়ের মেসেজ পড়ো", {"whatsapp_read_chat"}),
    ("মনে রাখো আমার অফিস সকাল নয়টায়", {"remember"}),
    ("open spotify and play some lofi music", {"open_app", "play_youtube", "use_computer"}),
    ("পেইন্ট খুলে একটা লাল বৃত্ত আঁকো", {"use_computer", "open_app"}),
    ("এক্সেলে একটা নতুন শিট খুলে A1 এ হ্যালো লেখো", {"use_computer", "open_app"}),
]


def main() -> None:
    from sohayok.assistant import Assistant, Command
    from sohayok.bench.dry_run import dry_registry
    from sohayok.config import current
    from sohayok.desktop import build_lite
    from sohayok.gpu import set_cpu_priority
    from sohayok.lang import detect_language

    config = current()
    set_cpu_priority(config.assistant.cpu_priority)
    config.assistant.confirm_risky_actions = False  # dry run: nothing real happens anyway
    brain, info = build_lite(config)
    print("answer model:", info, flush=True)
    registry, calls = dry_registry()
    assistant = Assistant(config, registry, brain)
    rows = []
    for text, ok in CASES:
        calls.clear()
        started = time.monotonic()
        try:
            reply = assistant.handle(Command(text, detect_language(text), "voice"))
        except Exception as exc:  # report, keep going
            reply = f"ERROR {exc}"
        seconds = time.monotonic() - started
        first = calls[0]["name"] if calls else ""
        passed = first in ok
        rows.append(
            {
                "said": text,
                "s": round(seconds, 2),
                "calls": calls[:],
                "reply": reply,
                "pass": passed,
            }
        )
        print(
            f"{'PASS' if passed else 'FAIL'} {seconds:5.2f}s {text} | "
            f"{[(c['name'], c['args']) for c in calls]} | {str(reply)[:80]}",
            flush=True,
        )
    passed = sum(r["pass"] for r in rows)
    times = sorted(r["s"] for r in rows)
    print(
        f"\n{passed}/{len(rows)} right; median {times[len(times) // 2]:.2f}s, max {times[-1]:.2f}s"
    )
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", encoding="utf-8") as out:
            json.dump(rows, out, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
