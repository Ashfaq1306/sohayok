"""Instant answers for everyday commands, in Bangla, Banglish, and English.

The fast path only claims a command when the whole sentence matches a known shape, so
anything ambiguous still goes to the language model.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from sohayok.lang import WHATSAPP, normalize, parse_number


@dataclass
class Intent:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    # Spoken confirmation, keyed by language; "{result}" is replaced with the tool result.
    reply: dict[str, str] = field(default_factory=dict)
    # Further (tool, args) steps run after the first one, e.g. typing after opening an app.
    then: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


OPEN = r"(?:খোলো|খুলো|খুলুন|খোল|অন করতো|অন করো|অন কর|ওপেন করতো|ওপেন করো|ওপেন করুন|ওপেন কর|চালু করো|চালু করুন|kholo|khulo|khol|open koro|open kor|chalu koro)"
CLOSE = r"(?:বন্ধ করো|বন্ধ করুন|বন্ধ কর|ক্লোজ করো|ক্লোজ করুন|কেটে দাও|bondho koro|bondho kor|close koro)"
PLEASE = r"(?:\s*(?:প্লিজ|please|plz|তো|একটু|ektu))*"
VOLUME = r"(?:ভলিউম|ভেলিউম|ভলিয়ম|ভোলিউম|ভলিউমটা|সাউন্ড|আওয়াজ|শব্দ|volume|sound|awaj)"
UP = r"(?:বাড়াও|বাড়িয়ে দাও|বাড়ান|বেশি করো|barao|bariye dao|up|increase|louder)"
DOWN = r"(?:কমাও|কমিয়ে দাও|কমান|কম করো|komao|komiye dao|down|decrease|lower|quieter)"
SET = r"(?:করো|করুন|কর|দাও|koro|kor|dao|set|to|at)"
FILLER = r"(?:\s+(?:ektu|একটু|আরেকটু|আরো|aro|ta|টা|টাকে))?"


def _rx(pattern: str) -> re.Pattern[str]:
    # Text is NFC-normalized before matching, so the patterns must be too.
    return re.compile(unicodedata.normalize("NFC", rf"^{pattern}{PLEASE}[.!?।]*$"))


RULES: list[tuple[re.Pattern[str], Any]] = []


def rule(*patterns: str):
    def decorator(build):
        for pattern in patterns:
            RULES.append((_rx(pattern), build))
        return build

    return decorator


@rule(
    rf"{VOLUME}{FILLER}\s*{UP}",
    rf"(?:turn (?:the )?)?{VOLUME} up",
    rf"(?:increase|raise) (?:the )?{VOLUME}",
)
def _volume_up(m, text):
    return Intent(
        "volume", {"action": "up", "level": 10}, {"bn": "ভলিউম বাড়ালাম।", "en": "Volume up."}
    )


@rule(
    rf"{VOLUME}{FILLER}\s*{DOWN}",
    rf"(?:turn (?:the )?)?{VOLUME} down",
    rf"(?:decrease|lower) (?:the )?{VOLUME}",
)
def _volume_down(m, text):
    return Intent(
        "volume", {"action": "down", "level": 10}, {"bn": "ভলিউম কমালাম।", "en": "Volume down."}
    )


@rule(
    rf"{VOLUME}\s*(?P<n>\S+(?: \S+)?)\s*(?:%|শতাংশ|পার্সেন্ট|percent)?\s*{SET}",
    rf"(?:set )?(?:the )?{VOLUME}\s*(?:to|at)?\s*(?P<n>\S+(?: \S+)?)\s*(?:%|percent)?",
)
def _volume_set(m, text):
    level = parse_number(m.group("n"))
    if level is None or not 0 <= level <= 100:
        return None
    return Intent(
        "volume",
        {"action": "set", "level": level},
        {"bn": "ভলিউম {n} করলাম।".replace("{n}", str(level)), "en": f"Volume set to {level}."},
    )


@rule(r"(?:mute|মিউট করো|মিউট করুন|মিউট কর|মিউট|sound off|সাউন্ড বন্ধ করো|আওয়াজ বন্ধ করো|mute koro)")
def _mute(m, text):
    return Intent("volume", {"action": "mute"}, {"bn": "মিউট করলাম।", "en": "Muted."})


@rule(r"(?:unmute|আনমিউট করো|আনমিউট করুন|আনমিউট|সাউন্ড চালু করো|আওয়াজ চালু করো|unmute koro)")
def _unmute(m, text):
    return Intent("volume", {"action": "unmute"}, {"bn": "সাউন্ড চালু করলাম।", "en": "Unmuted."})


@rule(
    r"(?:pause|resume|play|pause (?:the )?(?:music|song|video)|play (?:the )?(?:music|song|video)|"
    r"(?:গান|গানটা|গানটি|ভিডিও|ভিডিওটা|মিউজিক)\s*(?:থামাও|পজ করো|বন্ধ করো|আবার চালাও)|"
    r"থামাও|পজ করো|পজ|আবার চালাও|plz pause|pause koro|"
    r"(?:gaan|gan|video|music)\s*(?:ta|ti)?\s*(?:thamao|pause koro|bondho koro)|thamao)"
)
def _play_pause(m, text):
    return Intent("media", {"action": "playpause"}, {"bn": "ঠিক আছে।", "en": "Okay."})


@rule(
    r"(?:next|next (?:song|track|video)|skip|পরের (?:গান|গানটা|গানটি|ভিডিও|ভিডিওটা)|পরেরটা|"
    r"নেক্সট|নেক্সট গান|porer (?:gaan|gan)(?: ta)?|next (?:gaan|gan))"
    r"(?: চালাও| দাও| বাজাও| koro| dao| chalao)?"
)
def _next(m, text):
    return Intent("media", {"action": "next"}, {"bn": "পরেরটা চালাচ্ছি।", "en": "Next."})


@rule(r"(?:previous|previous (?:song|track)|আগের গান|আগেরটা|আগের ভিডিও|ager gaan)(?: চালাও| দাও)?")
def _previous(m, text):
    return Intent("media", {"action": "previous"}, {"bn": "আগেরটা চালাচ্ছি।", "en": "Previous."})


@rule(
    r"(?:what(?:'s| is) the time|what time is it|time|tell me the time|"
    r"কয়টা বাজে|কটা বাজে|কয়েকটা বাজে|এখন কয়টা বাজে|এখন কটা বাজে|এখন কয়েকটা বাজে|সময় কত|এখন সময় কত|ক'টা বাজে|"
    r"koyta baje|ekhon koyta baje|somoy koto)"
)
def _time(m, text):
    return Intent("current_time", {}, {"bn": "এখন {result}।", "en": "It's {result}."})


@rule(
    r"(?:what(?:'s| is) (?:the )?date(?: today)?|what day is it|today's date|"
    r"আজ কত তারিখ|আজকে কত তারিখ|আজ কী বার|আজ কি বার|আজকের তারিখ|aj koto tarikh)"
)
def _date(m, text):
    return Intent("current_time", {}, {"bn": "{result}।", "en": "{result}."})


@rule(r"(?:take a screenshot|screenshot|স্ক্রিনশট নাও|স্ক্রিনশট নিন|স্ক্রিনশট|screenshot nao)")
def _screenshot(m, text):
    return Intent("take_screenshot", {}, {"bn": "স্ক্রিনশট নিয়েছি।", "en": "Screenshot saved."})


@rule(
    r"(?:lock (?:the |my )?(?:computer|screen|pc|laptop)|lock|কম্পিউটার লক করো|লক করো|"
    r"স্ক্রিন লক করো|পিসি লক করো|lock koro)"
)
def _lock(m, text):
    return Intent("lock_screen", {}, {"bn": "লক করছি।", "en": "Locking."})


@rule(r"(?:show (?:the )?desktop|minimi[sz]e (?:all|everything)|ডেস্কটপ দেখাও|সব মিনিমাইজ করো)")
def _desktop(m, text):
    return Intent("show_desktop", {}, {"bn": "ডেস্কটপ দেখাচ্ছি।", "en": "Done."})


TAB = r"(?:ট্যাব|ট্যাপ|ট্যাভ|tab)(?:টা|টি|গুলো|টাকে)?"
SHUT = (
    r"(?:কেটে (?:দাও|ফেলো|ফেল|দেও|দিন)|কাটো|কাট|বন্ধ করো|বন্ধ করে দাও|বন্ধ করুন|ক্লোজ করো|"
    r"ক্লোজ করে দাও|close koro|kete dao|kete felo)"
)


@rule(
    rf"(?:(?P<site>.+?)\s+)?{TAB}(?:\s+(?:যা |যেগুলো )?খোলা(?: আছে)?)?(?:\s+সব(?:গুলো|গুলোই)?)?\s*{SHUT}",
    r"close (?:the |this |that )?(?:(?P<site>.+?) )?tab",
)
def _close_tab(m, text):
    site = (m.group("site") or "").strip(" ,")
    every = bool(re.search(r"সব|গুলো|যতগুলো|all\b|every|tabs\b", text))
    site = re.sub(
        r"\b(?:all|every|the|my|old|previous)\b|সবগুলো|সব|যতগুলো\S*|পূর্বের|আগের|পুরানো|পুরোনো|"
        r"খোলা আছে|আছে|ক্রোমে|ক্রমে|chrome",
        " ",
        site,
    )
    site = re.sub(r"\s+", " ", site).strip(" ,")
    if normalize(site) in {"", "this", "the", "এই", "এটা", "এই ট্যাব"}:
        site = ""
    reply = {"bn": "ট্যাব বন্ধ করলাম।", "en": "Closed."}
    return Intent("close_tab", {"name": site, "every": every and bool(site)}, reply)


# "open Zoom", "zoom meeting": the Zoom app, not zooming in.
ZOOM_APP = re.compile(r"open|ওপেন|খোলো|খুলো|meeting|মিটিং|\bapp\b|অ্যাপ|কল|call", re.IGNORECASE)
NOTIFICATIONS = r"(?:নোটিফিকেশন|নোটিফিকেশান|notification)(?:s|গুলো|গুলি|টা|টি)?"


@rule(
    rf"(?:(?:আমার|my)\s+)?(?:(?P<app>\S+?)(?:র|এর|ের|-এর)?\s+)?{NOTIFICATIONS}\s*"
    r"(?:ক্লিয়ার করো|ক্লিয়ার করে দাও|মুছে দাও|মুছে ফেলো|মুছো|সরাও|সরিয়ে দাও|ক্লোজ করো|বন্ধ করো|"
    r"কেটে দাও|clear koro)",
    r"(?:clear|dismiss|close|remove)\s+(?:all\s+)?(?:(?:my|the)\s+)?(?:(?P<app>\w+)\s+)?"
    r"notifications?",
)
def _dismiss_notifications(m, text):
    app = (m.group("app") or "").strip()
    if normalize(app) in {"all", "সব", "সবগুলো", "the", "my", "আমার"}:
        app = ""
    from sohayok.lang import WHATSAPP

    if re.fullmatch(WHATSAPP, app, re.IGNORECASE):
        app = "WhatsApp"
    return Intent("dismiss_notifications", {"app": app}, {"bn": "ঠিক আছে।", "en": "Okay."})


@rule(
    r"(?:.+?\s+)?(?:জুম আউট|zoom out|ছোট করে দেখাও)(?: করো| করে দাও| koro)?",
)
def _zoom_out(m, text):
    if ZOOM_APP.search(text):
        return None
    return Intent("press_keys", {"keys": "ctrl+minus"}, {"bn": "ছোট করলাম।", "en": "Zoomed out."})


@rule(
    r"(?:.+?\s+)?(?:জুম ইন|জুম|zoom in|zoom)"
    r"(?: করো| করে দাও| করে দেখাও| করে তো দেখি| করে দেখি| koro)?",
)
def _zoom_in(m, text):
    if ZOOM_APP.search(text):
        return None
    return Intent("press_keys", {"keys": "ctrl+plus"}, {"bn": "বড় করলাম।", "en": "Zoomed in."})


@rule(r"(?:close (?:this|the) (?:window|tab|app)|এটা বন্ধ করো|উইন্ডো বন্ধ করো|এই উইন্ডো বন্ধ করো)")
def _close_window(m, text):
    if "tab" in text or "ট্যাব" in text:
        return Intent("press_keys", {"keys": "ctrl+w"}, {"bn": "ট্যাব বন্ধ করলাম।", "en": "Closed."})
    return Intent("window_action", {"action": "close"}, {"bn": "বন্ধ করলাম।", "en": "Closed."})


@rule(r"(?:minimi[sz]e(?: (?:this|the) window)?|উইন্ডো মিনিমাইজ করো|মিনিমাইজ করো|ছোট করো)")
def _minimize(m, text):
    return Intent("window_action", {"action": "minimize"}, {"bn": "ঠিক আছে।", "en": "Done."})


@rule(r"(?:maximi[sz]e(?: (?:this|the) window)?|উইন্ডো ম্যাক্সিমাইজ করো|ম্যাক্সিমাইজ করো|বড় করো)")
def _maximize(m, text):
    return Intent("window_action", {"action": "maximize"}, {"bn": "ঠিক আছে।", "en": "Done."})


@rule(r"(?:new tab|open (?:a )?new tab|নতুন (?:ট্যাব|ট্যাপ)(?: খোলো)?|notun tab kholo)")
def _new_tab(m, text):
    return Intent("press_keys", {"keys": "ctrl+t"}, {"bn": "নতুন ট্যাব।", "en": "New tab."})


@rule(
    r"(?P<q>.+?)\s*(?:ইউটিউবে|ইউটিউব এ|youtube e|youtube-e)\s*(?:চালাও|বাজাও|প্লে করো|দেখাও|chalao|bajao|play koro)",
    r"play (?P<q>.+?) on youtube",
    r"(?:ইউটিউবে|youtube e)\s*(?P<q>.+?)\s*(?:চালাও|বাজাও|প্লে করো|chalao)",
)
def _youtube(m, text):
    return Intent(
        "play_youtube",
        {"query": m.group("q")},
        {"bn": "ইউটিউবে চালাচ্ছি।", "en": "Playing it on YouTube."},
    )


@rule(
    r"(?:search|google)(?: (?:for|on google for))? (?P<q>.+)",
    r"(?P<q>.+?)\s*(?:গুগলে|গুগল এ|google e)?\s*(?:সার্চ করো|সার্চ করুন|খোঁজো|খুঁজে দাও|খুঁজুন|search koro|khojo)",
)
def _search(m, text):
    query = m.group("q").strip()
    if not query or len(query) < 2 or re.search(r" and | then | তারপর | এবং ", f" {query} "):
        return None
    if re.search(rf"{OPEN}$|\bopen$|ওপেন$", query):
        return None  # "Google Chrome ওপেন করো" opens Chrome
    return Intent(
        "web_search", {"query": query, "engine": "google"}, {"bn": "খুঁজছি।", "en": "Searching."}
    )


@rule(
    r"(?P<x>\S+(?: \S+)?) খুলে (?:লেখো|লিখো|লিখে দাও|টাইপ করো)[:,]?\s+(?P<t>.+)",
    r"(?P<x>\S+(?: \S+)?) khule (?:likho|lekho|type koro)[:,]?\s+(?P<t>.+)",
    r"open (?P<x>\S+(?: \S+)?) and (?:type|write)[:,]?\s+(?P<t>.+)",
)
def _open_and_type(m, text):
    """The most common two-step request: open an app, then dictate into it."""
    from sohayok.tools.apps import resolve_app, start_apps

    try:
        app = resolve_app(m.group("x"), start_apps())
    except Exception:
        app = None
    if not app:
        return None
    words = m.group("t").strip().strip("\"'“”")
    return Intent(
        "open_app",
        {"name": app},
        {"bn": f"{app} খুলে লিখে দিলাম।", "en": f"Opened {app} and typed it."},
        then=[("type_text", {"text": words})],
    )


@rule(
    rf"(?P<x>.+?)\s*(?:সেটিংস|সেটিং|settings|setting)\s*{OPEN}",
    r"(?:open|show) (?:the )?(?P<x>.+?) settings",
)
def _settings(m, text):
    from sohayok.tools.system import SETTINGS

    page = SETTINGS_WORDS.get(normalize(m.group("x")), normalize(m.group("x")))
    if page not in SETTINGS:
        return None
    return Intent(
        "open_settings", {"page": page}, {"bn": "সেটিংস খুলছি।", "en": f"Opening {page} settings."}
    )


SETTINGS_WORDS = {
    "ওয়াইফাই": "wifi",
    "ওয়াই-ফাই": "wifi",
    "wi-fi": "wifi",
    "ব্লুটুথ": "bluetooth",
    "ডিসপ্লে": "display",
    "সাউন্ড": "sound",
    "নেটওয়ার্ক": "network",
    "আপডেট": "update",
    "windows update": "update",
    "উইন্ডোজ আপডেট": "update",
    "ব্যাটারি": "battery",
    "মাইক্রোফোন": "microphone",
    "ক্যামেরা": "camera",
    "ভাষা": "language",
    "তারিখ": "time",
    "সময়": "time",
    "প্রিন্টার": "printers",
    "ব্যাকগ্রাউন্ড": "background",
    "থিম": "themes",
}
SETTINGS_WORDS = {normalize(k): v for k, v in SETTINGS_WORDS.items()}


@rule(
    rf"(?P<c>.+?)(?:কে|ke)\s+(?:{WHATSAPP}(?:ে|-এ| এ| e)?\s+)?"
    r"(?P<v>ভিডিও\s+|video\s+)?(?:কল|ফোন|call|phone)\s+(?:দাও|দেও|দিন|করো|কর|লাগাও|dao|deo|koro)",
    r"(?P<v>video )?call (?P<c>.+?)(?: on whatsapp)?",
)
def _call(m, text):
    contact = m.group("c").strip(" ,")
    if not contact or len(contact.split()) > 4 or normalize(contact) in {"me", "আমাকে", "তাকে"}:
        return None
    video = bool(m.group("v"))
    return Intent(
        "whatsapp_call", {"contact": contact, "video": video}, {"bn": "কল দিচ্ছি।", "en": "Calling."}
    )


BT = r"(?:ব্লুটুথ|ব্লুটুথটা|ব্লু টুথ|bluetooth)"
WIFI = r"(?:ওয়াইফাই|ওয়াই ফাই|ওয়াইফাইটা|wi-?fi|wifi)"
ON = r"(?:অন করো|অন কর|চালু করো|চালু কর|অন করে দাও|চালু করে দাও|on koro|chalu koro)"
OFF = r"(?:অফ করো|অফ কর|বন্ধ করো|বন্ধ কর|অফ করে দাও|বন্ধ করে দাও|off koro|bondho koro)"


@rule(
    rf"{BT}\s*{ON}",
    rf"{BT}\s*{OFF}",
    r"turn (?:on|off) (?:the )?bluetooth",
    r"turn (?:the )?bluetooth (?:on|off)",
    r"bluetooth (?:on|off)",
)
def _bluetooth(m, text):
    on = not re.search(r"অফ|বন্ধ|off|bondho", text)
    return Intent("bluetooth_power", {"on": on}, {"bn": "ঠিক আছে।", "en": "Okay."})


@rule(
    rf"{WIFI}\s*{ON}",
    rf"{WIFI}\s*{OFF}",
    r"turn (?:on|off) (?:the )?wi-?fi",
    r"turn (?:the )?wi-?fi (?:on|off)",
    r"wi-?fi (?:on|off)",
)
def _wifi(m, text):
    on = not re.search(r"অফ|বন্ধ|off|bondho", text)
    return Intent("wifi_power", {"on": on}, {"bn": "ঠিক আছে।", "en": "Okay."})


@rule(
    r"(?P<d>.+?)(?:টা|টি|কে)?\s+(?P<a>ডিসকানেক্ট|disconnect|কানেক্ট|connect)\s*(?:করো|কর|করে দাও|koro)",
    r"(?P<a>disconnect|connect)(?: to)? (?:my |the )?(?P<d>.+)",
)
def _bluetooth_device(m, text):
    device = m.group("d").strip(" ,")
    if not device or len(device.split()) > 4 or re.search(rf"{WIFI}|ইন্টারনেট|internet", device):
        return None
    action = "disconnect" if re.search(r"ডিস|dis", m.group("a")) else "connect"
    return Intent(
        "bluetooth_device", {"device": device, "action": action}, {"bn": "ঠিক আছে।", "en": "Okay."}
    )


def _place(target: str):
    """A drive, shell place, or known folder named in a command (with or without "-এ")."""
    from sohayok.tools.files import drive, known_folder, shell_place

    target = target.strip()
    for candidate in (target, *(target[: -len(end)] for end in LOCATIVE if target.endswith(end))):
        candidate = candidate.strip(" -")
        if not candidate:
            continue
        found = shell_place(candidate) or drive(candidate) or known_folder(candidate)
        if found:
            return candidate, found
    return None


LOCATIVE = ("তে", "ে", "এ", "য়")
HERE = {"এখানে", "এইখানে", "ঐখানে", "এই ফোল্ডারে", "here", "this folder", "the current folder"}
MAKE = r"(?:বানাও|বানান|বানা|বানিয়ে দাও|তৈরি করো|তৈরি করুন|তৈরি কর|ক্রিয়েট করো|create করো|banao|baniye dao|create koro)"
# "ফোল্ডার খোলো" opens a folder; only "নতুন ফোল্ডার খোলো" makes one.
NEW_OPEN = r"নতুন\s+(?:ফোল্ডার|folder)\s+(?:খোলো|খুলো|খোল)"


def _folder_intent(name: str, where: str):
    args = {"name": (name or "").strip(" \"'"), "location": ""}
    if normalize(where) in HERE:
        where = ""  # the folder open in File Explorer (the tool's default)
    if where:
        place = _place(where)
        if place is None or isinstance(place[1], str):  # unknown, or This PC / Recycle Bin
            return None
        args["location"] = str(place[1])
    return Intent("create_folder", args, {"bn": "ফোল্ডার বানালাম।", "en": "Folder created."})


@rule(
    rf"(?:(?P<where>.+?(?:তে|(?<!নাম)ে|এ|য়))\s+)?(?:(?P<name>.+?)\s+নামে(?:র)?\s+)?"
    rf"(?:একটা|একটি|এক্টা)?\s*(?:(?:নতুন\s+)?(?:ফোল্ডার|folder)\s+{MAKE}|{NEW_OPEN})"
    rf"(?:\s+(?:যার নাম|নাম|নামে)\s+(?P<name2>.+))?",
)
def _new_folder_bn(m, text):
    return _folder_intent(m.group("name") or m.group("name2") or "", m.group("where") or "")


@rule(
    r"(?:create|make)(?: a| an)?(?: new)? folder(?: (?:called|named) (?P<name>.+?))?"
    r"(?: (?:on|in|inside) (?:the |my )?(?P<where>.+?)| here)?",
    r"(?:create|make)(?: a| an)?(?: new)? folder (?:on|in|inside) (?:the |my )?(?P<where>.+?)"
    r" (?:called|named) (?P<name>.+)",
    r"new folder(?: (?:called|named) (?P<name>.+))?",
)
def _new_folder_en(m, text):
    groups = m.groupdict()
    return _folder_intent(groups.get("name") or "", groups.get("where") or "")


@rule(
    r"(?P<x>.+?)\s*(?:যাও|যান|চলো|চল|নিয়ে যাও|jao|cholo)",
    r"(?:go|navigate|take me) to (?:the |my )?(?P<x>.+)",
)
def _go_to(m, text):
    place = _place(m.group("x").strip(" .!?।"))
    if place is None:
        return None
    name, _ = place
    return Intent("open_path", {"path": name}, {"bn": f"{name} খুলছি।", "en": f"Opening {name}."})


# English word order in Bangla letters ("ওপেন ক্রোম") is how the user's English often comes out.
@rule(rf"(?P<x>.+?)\s*{OPEN}", r"(?:open|launch|start|run|ওপেন|লঞ্চ|স্টার্ট) (?P<x>.+)")
def _open(m, text):
    from sohayok.tools.browser import SITES

    target = m.group("x").strip(" .!?।")
    if _compound(target):
        return None
    target = re.sub(r"^(?:the|my|আমার)\s+", "", target)
    target = re.sub(r"\s+(?:app|application|অ্যাপ|অ্যাপটা|টা|টি|ta)$", "", target)
    target = re.sub(r"(?<=[ঀ-৿]{2})(?:টা|টি|টাকে)$", "", target)  # "ক্রোমটা" -> "ক্রোম"
    key = normalize(target).removesuffix(".com")
    if key.removesuffix(" website").removesuffix(" ওয়েবসাইট") in SITES or re.search(
        r"\.\w{2,}$", target
    ):
        from sohayok.tools.browser import site_url

        try:
            url = site_url(target)
        except Exception:
            return None
        return Intent(
            "open_website", {"site": url}, {"bn": f"{target} খুলছি।", "en": f"Opening {target}."}
        )
    from sohayok.tools.files import drive, known_folder, shell_place

    if shell_place(target) or drive(target):
        return Intent(
            "open_path", {"path": target}, {"bn": f"{target} খুলছি।", "en": f"Opening {target}."}
        )
    folder = known_folder(target)
    if folder:
        name = re.sub(r"\s*(?:folder|ফোল্ডার)$", "", target)
        return Intent(
            "open_path",
            {"path": str(folder)},
            {"bn": f"{name} ফোল্ডার খুলছি।", "en": f"Opening {name}."},
        )
    from sohayok.tools.apps import resolve_app, start_apps

    try:
        app = resolve_app(target, start_apps())
    except Exception:
        app = None
    if not app:
        return None
    return Intent("open_app", {"name": app}, {"bn": f"{app} খুলছি।", "en": f"Opening {app}."})


@rule(rf"(?P<x>.+?)\s*{CLOSE}", r"(?:close|quit|exit|ক্লোজ) (?P<x>.+)")
def _close(m, text):
    target = m.group("x").strip()
    if _compound(target):
        return None
    if normalize(target) in {"this", "it", "এটা", "এটি", "এই উইন্ডো", "window", "this window"}:
        return Intent("window_action", {"action": "close"}, {"bn": "বন্ধ করলাম।", "en": "Closed."})
    from sohayok.tools.apps import resolve_app, start_apps

    try:
        app = resolve_app(target, start_apps())
    except Exception:
        app = None
    if not app:
        return None
    return Intent("close_app", {"name": app}, {"bn": f"{app} বন্ধ করছি।", "en": f"Closing {app}."})


WAKE_PREFIX = re.compile(
    unicodedata.normalize(
        "NFC",
        r"^(?:hey |hi |ok |okay |হেই |এই )?(?:tammy|tami|tammi|tommy|sohayok|jarvis|টামি|ট্যামি|তামি"
        r"|সহায়ক|জার্ভিস)(?![a-zঀ-৿])[\s,.!।?]*",
    ),
    re.IGNORECASE,
)


def strip_wake_name(text: str) -> str:
    """Remove a leading "Tammy," / "হেই টামি" so the rest is the command."""
    return WAKE_PREFIX.sub("", text.strip(), count=1).strip()


# How the user's "Tammy" can come out at the start of a command said in one breath
# ("Tammy, open Chrome" -> "ডেমি ক্রোম খোলো"). "তুমি" / "you" are left alone: the user
# starts many commands with them on purpose.
WAKE_ECHOES = {
    "টুমি",
    "টামি",
    "ট্যামি",
    "তামি",
    "টেমি",
    "ডেমি",
    "ড্যামি",
    "ডামি",
    "কেমি",
    "tammy",
    "tami",
    "tommy",
    "tummy",
    "dummy",
    "demi",
}


def strip_wake_echo(text: str) -> str:
    """Drop a misheard wake word from the start of a command that followed the wake word."""
    cleaned = strip_wake_name(text)
    words = cleaned.split()
    for size in (2, 1):
        head = " ".join(words[:size]).strip(",.!?।").lower()
        if len(words) > size and unicodedata.normalize("NFC", head) in WAKE_ECHO_SET:
            return " ".join(words[size:]).lstrip(",.!?। ")
    return cleaned


WAKE_ECHO_SET = {unicodedata.normalize("NFC", w) for w in WAKE_ECHOES}


def _compound(target: str) -> bool:
    """True for multi-step requests ("chrome and search cats"), which need the model."""
    return len(target.split()) > 4 or bool(
        re.search(r",| and | then | এবং | তারপর | আর | তারপরে | তার পর | ar ", f" {target} ")
    )


def route(text: str) -> Intent | None:
    """Return an Intent if the whole command matches a fast-path rule."""
    cleaned = normalize(text).strip(" ,")
    cleaned = strip_wake_name(cleaned)
    cleaned = re.sub(r"^(?:please|plz|প্লিজ|একটু)\s+", "", cleaned)
    if not cleaned or len(cleaned) > 80:
        return None
    for pattern, build in RULES:
        match = pattern.match(cleaned)
        if match:
            intent = build(match, cleaned)
            if intent is not None:
                return intent
    return None
