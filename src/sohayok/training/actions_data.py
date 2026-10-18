"""Training data for the action decision: MASSIVE (real) plus desktop templates (synthetic).

MASSIVE's assistant requests are mapped onto Sohayok's actions. No public dataset has
desktop requests ("open Notepad", "minimize this window") in Bangla, so those come from
templates with Bangla, Banglish, and English phrasings. The hand-written benchmark commands
(bench/desktop_commands.jsonl) are held out: any generated sentence close to one of them is
dropped, so the benchmark still measures generalization.
"""

from __future__ import annotations

import gzip
import json
import random
from collections import Counter
from importlib import resources
from pathlib import Path

from rapidfuzz import fuzz

from sohayok.decide import ACTIONS, QUESTION
from sohayok.lang import normalize

# MASSIVE intent → Sohayok action. Intents the PC cannot do become "other".
MASSIVE_ACTION = {
    "alarm_query": "reminder", "alarm_remove": "reminder", "alarm_set": "reminder",
    "audio_volume_down": "volume", "audio_volume_mute": "volume", "audio_volume_other": "volume",
    "audio_volume_up": "volume", "calendar_query": "other", "calendar_remove": "other",
    "calendar_set": "reminder", "cooking_query": "question", "cooking_recipe": "question",
    "datetime_convert": "question", "datetime_query": "time", "email_addcontact": "other",
    "email_query": "website", "email_querycontact": "other", "email_sendemail": "other",
    "general_greet": "chat", "general_joke": "chat", "general_quirky": "chat",
    "iot_cleaning": "other", "iot_coffee": "other", "iot_hue_lightchange": "other",
    "iot_hue_lightdim": "other", "iot_hue_lightoff": "other", "iot_hue_lighton": "other",
    "iot_hue_lightup": "other", "iot_wemo_off": "other", "iot_wemo_on": "other",
    "lists_createoradd": "note", "lists_query": "other", "lists_remove": "other",
    "music_dislikeness": "other", "music_likeness": "other", "music_query": "other",
    "music_settings": "media", "news_query": "question", "play_audiobook": "play",
    "play_game": "other", "play_music": "play", "play_podcasts": "play", "play_radio": "play",
    "qa_currency": "question", "qa_definition": "question", "qa_factoid": "question",
    "qa_maths": "question", "qa_stock": "question", "recommendation_events": "question",
    "recommendation_locations": "question", "recommendation_movies": "question",
    "social_post": "other", "social_query": "other", "takeaway_order": "other",
    "takeaway_query": "other", "transport_query": "question", "transport_taxi": "other",
    "transport_ticket": "other", "transport_traffic": "question", "weather_query": "weather",
}

# Instruction wordings used in training; QUESTION (decide.py) is the deployed wording and
# is deliberately not among them, so the model must not depend on one exact phrasing.
TRAIN_QUESTIONS = {
    "bn": ["ব্যবহারকারী আসলে কী করতে চাইছেন?", "এই অনুরোধ পূরণ করতে কোন কাজটি করতে হবে?"],
    "en": ["What does the user want the assistant to do?"],
}

APPS = [("ক্রোম", "chrome"), ("গুগল ক্রোম", "google chrome"), ("নোটপ্যাড", "notepad"),
        ("ক্যালকুলেটর", "calculator"), ("ভিএস কোড", "vs code"),
        ("ফাইল এক্সপ্লোরার", "file explorer"), ("ওয়ার্ড", "word"), ("এক্সেল", "excel"),
        ("পাওয়ারপয়েন্ট", "powerpoint"), ("পেইন্ট", "paint"), ("স্পটিফাই", "spotify"),
        ("টেলিগ্রাম", "telegram"), ("হোয়াটসঅ্যাপ", "whatsapp"), ("ডিসকর্ড", "discord"),
        ("জুম", "zoom"), ("টাস্ক ম্যানেজার", "task manager"), ("টার্মিনাল", "terminal"),
        ("ক্যামেরা", "camera"), ("ভিএলসি", "vlc"), ("এজ ব্রাউজার", "edge"),
        ("অ্যান্ড্রয়েড স্টুডিও", "android studio"), ("স্টিম", "steam"),
        ("ফটোশপ", "photoshop"), ("স্নিপিং টুল", "snipping tool"), ("এনিডেস্ক", "anydesk"),
        ("পোস্টম্যান", "postman"), ("মাইক্রোসফট স্টোর", "microsoft store"),
        ("নেটবিনস", "netbeans"), ("ফটোস অ্যাপ", "photos"), ("কন্ট্রোল প্যানেল", "control panel")]
SITES = [("জিমেইল", "gmail"), ("ফেসবুক", "facebook"), ("ইউটিউব", "youtube"),
         ("গিটহাব", "github"), ("প্রথম আলো", "prothom alo"), ("লিংকডইন", "linkedin"),
         ("গুগল ড্রাইভ", "google drive"), ("মেসেঞ্জার", "messenger"),
         ("উইকিপিডিয়া", "wikipedia"), ("নেটফ্লিক্স", "netflix"), ("দারাজ", "daraz"),
         ("বিবিসি বাংলা", "bbc bangla"), ("চ্যাটজিপিটি", "chatgpt"), ("ইনস্টাগ্রাম", "instagram"),
         ("গুগল ম্যাপ", "google maps"), ("হাগিং ফেস", "huggingface.co")]
QUERIES_BN = ["ঢাকার সেরা বিরিয়ানি", "পদ্মা সেতুর ছবি", "বাংলাদেশ ক্রিকেট দলের খবর",
              "পাইথন শেখার উপায়", "কক্সবাজারের হোটেল", "বিসিএস প্রস্তুতি", "ল্যাপটপের দাম",
              "চট্টগ্রামের ট্রেনের সময়", "সহজ রান্নার রেসিপি", "নতুন বাংলা সিনেমা"]
QUERIES_EN = ["cheap flights to dhaka", "best python tutorials", "rtx 4060 drivers",
              "weather radar", "how to fix wifi", "new laptops 2026", "react hooks guide",
              "football scores", "free online courses", "dhaka metro rail schedule"]
MEDIA_BN = ["আইয়ুব বাচ্চুর গান", "রবীন্দ্রসংগীত", "জেমসের গান", "নজরুলগীতি", "লালনের গান",
            "বাংলা নাটক", "ক্রিকেট হাইলাইটস", "কোরআন তেলাওয়াত", "ছোটদের কার্টুন", "আর্টসেলের গান"]
MEDIA_EN = ["lofi beats", "arijit singh songs", "coke studio bangla", "relaxing piano",
            "the news", "a podcast about science", "workout music", "taylor swift",
            "an audiobook", "jazz radio"]
TEXTS_BN = ["আজকে মিটিং বিকাল চারটায়", "আমি পরে ফোন করব", "শুভ সকাল সবাইকে",
            "দয়া করে রিপোর্টটা পাঠান", "ধন্যবাদ ভাই", "কালকে ক্লাস নেই"]
TEXTS_EN = ["see you tomorrow", "the report is ready", "happy birthday",
            "please check the attached file", "running late, sorry", "dear sir"]
CITIES = [("ঢাকা", "dhaka"), ("চট্টগ্রাম", "chattogram"), ("সিলেট", "sylhet"),
          ("রাজশাহী", "rajshahi"), ("খুলনা", "khulna"), ("বরিশাল", "barishal"),
          ("কক্সবাজার", "cox's bazar"), ("লন্ডন", "london")]
FILES = [("রিপোর্ট", "report"), ("ছবিগুলো", "photos"), ("অ্যাসাইনমেন্ট", "assignment"),
         ("ইনভয়েস", "invoice"), ("সিভি", "cv"), ("প্রেজেন্টেশন", "presentation")]
FOLDERS = [("ডাউনলোড", "downloads"), ("ডকুমেন্টস", "documents"), ("ডেস্কটপ", "desktop"),
           ("ছবি", "pictures"), ("ভিডিও", "videos"), ("গান", "music")]
SETTINGS_PAGES = [("ওয়াইফাই", "wifi"), ("ব্লুটুথ", "bluetooth"), ("ডিসপ্লে", "display"),
                  ("সাউন্ড", "sound"), ("উইন্ডোজ আপডেট", "windows update"),
                  ("ব্যাটারি", "battery"), ("মাইক্রোফোন", "microphone"), ("ভাষা", "language")]
SOFTWARE = ["vlc", "7zip", "python", "nodejs", "git", "zoom", "obs studio", "notepad++"]
PEOPLE_BN = ["কাজী নজরুল ইসলাম", "সাকিব আল হাসান", "জগদীশ চন্দ্র বসু", "বেগম রোকেয়া"]
PEOPLE_EN = ["marie curie", "elon musk", "the president of france", "satyajit ray"]
THINGS_BN = ["কৃত্রিম বুদ্ধিমত্তা", "ব্ল্যাক হোল", "মুদ্রাস্ফীতি", "ডিএনএ"]
THINGS_EN = ["machine learning", "photosynthesis", "a mutual fund", "quantum computing"]
NUMBERS = [("দশ", "10"), ("বিশ", "20"), ("ত্রিশ", "30"), ("পঞ্চাশ", "50"), ("সত্তর", "70"),
           ("৮০", "80"), ("৪০", "40")]
MINUTES = [("পাঁচ", "5"), ("দশ", "10"), ("পনেরো", "15"), ("ত্রিশ", "30"), ("২", "2")]
BUTTONS = [("সেভ", "save"), ("ওকে", "ok"), ("সাবমিট", "submit"), ("ডাউনলোড", "download"),
           ("নেক্সট", "next"), ("ক্যানসেল", "cancel")]

# {a}=app, {s}=site, {q}=query, {m}=media, {t}=text, {c}=city, {f}=file, {d}=folder,
# {p}=settings page, {n}=number, {k}=minutes, {w}=software, {who}, {what}, {b}=button.
# Each action: (Bangla, Banglish, English) templates. "{X:bn}" picks the Bangla value.
TEMPLATES: dict[str, tuple[list[str], list[str], list[str]]] = {
    "open_app": (["{a} খোলো", "{a} চালু করো", "{a} ওপেন করো", "একটু {a} খুলে দাও",
                  "{a} অ্যাপটা চালাও", "আমার {a} খুলুন", "{a} টা খোলো তো"],
                 ["{a:en} kholo", "{a:en} open koro", "{a:en} ta chalu koro", "ektu {a:en} khulo"],
                 ["open {a:en}", "launch {a:en}", "start {a:en}", "can you open {a:en}",
                  "please run {a:en}", "bring up {a:en}"]),
    "close_app": (["{a} বন্ধ করো", "{a} বন্ধ করে দাও", "{a} ক্লোজ করো", "{a} অ্যাপটা বন্ধ করুন"],
                  ["{a:en} bondho koro", "{a:en} close koro", "{a:en} ta bondho kore dao"],
                  ["close {a:en}", "quit {a:en}", "exit {a:en}", "shut {a:en} down"]),
    "website": (["{s} খোলো", "{s} এ যাও", "ব্রাউজারে {s} খোলো", "{s} ওয়েবসাইটটা খুলে দাও"],
                ["{s:en} kholo", "{s:en} e jao", "browser e {s:en} open koro"],
                ["go to {s:en}", "open {s:en}", "take me to {s:en}", "visit {s:en}",
                 "open {s:en} in chrome"]),
    "search": (["{q} সার্চ করো", "গুগলে {q} খোঁজো", "{q} খুঁজে দাও", "ইন্টারনেটে {q} দেখো",
                "গুগলে সার্চ দাও {q}"],
               ["{q:en} search koro", "google e {q:en} khojo", "{q:en} niye search dao"],
               ["search for {q:en}", "google {q:en}", "look up {q:en} on google",
                "search the web for {q:en}", "find {q:en} online"]),
    "play": (["{m} চালাও", "{m} বাজাও", "ইউটিউবে {m} চালাও", "{m} শুনতে চাই", "{m} প্লে করো"],
             ["{m:en} chalao", "youtube e {m:en} play koro", "{m:en} bajao"],
             ["play {m:en}", "put on {m:en}", "play {m:en} on youtube",
              "i want to listen to {m:en}"]),
    "media": (["গান থামাও", "পজ করো", "আবার চালাও", "পরের গান দাও", "আগের গানটা চালাও",
               "ভিডিওটা থামাও", "এটা স্কিপ করো", "গানটা আবার শুরু করো"],
              ["gaan thamao", "pause koro", "next gaan dao", "video ta thamao", "abar chalao"],
              ["pause", "resume the music", "next song", "skip this track", "previous track",
               "stop the music", "pause the video"]),
    "volume": (["ভলিউম বাড়াও", "আওয়াজ কমাও", "সাউন্ড {n} করো", "মিউট করো",
                "শব্দ একটু বাড়িয়ে দাও", "আওয়াজ একদম বন্ধ করো", "ভলিউম {n} এ রাখো",
                "আওয়াজ খুব বেশি, কমাও"],
               ["volume barao", "awaj komao", "sound {n:en} koro", "mute koro", "volume ektu barao"],
               ["turn it up", "volume to {n:en}", "lower the volume", "mute", "unmute",
                "make it louder", "it's too loud", "set the sound to {n:en} percent"]),
    "time": (["কয়টা বাজে", "আজ কত তারিখ", "আজ কী বার", "এখন সময় কত", "আজকের তারিখটা বলো"],
             ["koyta baje", "aj koto tarikh", "ekhon somoy koto"],
             ["what time is it", "what's today's date", "what day is it today",
              "tell me the time"]),
    "weather": (["{c} আবহাওয়া কেমন", "আজ কি বৃষ্টি হবে", "বাইরে কত গরম",
                 "{c} তাপমাত্রা কত", "কাল কি রোদ থাকবে"],
                ["{c:en} er weather kemon", "aj ki brishti hobe"],
                ["what's the weather in {c:en}", "will it rain today", "how hot is it outside",
                 "temperature in {c:en}"]),
    "reminder": (["{k} মিনিট পরে মনে করিয়ে দিও", "{k} মিনিটের টাইমার দাও",
                  "সকাল সাতটায় অ্যালার্ম দাও", "আমার রিমাইন্ডারগুলো দেখাও", "অ্যালার্মটা বাতিল করো",
                  "রাত দশটায় মনে করিয়ে দিও ওষুধ খেতে হবে"],
                 ["{k:en} minute por mone koriye dio", "alarm set koro 7 tay"],
                 ["remind me in {k:en} minutes", "set a timer for {k:en} minutes",
                  "wake me up at 6 am", "what reminders do i have", "cancel my alarm"]),
    "note": (["একটা নোট রাখো: {t}", "লিখে রাখো {t}", "নোট করে রাখো যে {t}",
              "বাজারের লিস্টে ডিম যোগ করো"],
             ["ekta note rakho {t}", "note kore rakho {t}"],
             ["take a note: {t:en}", "note that {t:en}", "add milk to my shopping list",
              "save a note saying {t:en}"]),
    "type": (["লেখো {t}", "{t} টাইপ করো", "এখানে লিখে দাও {t}", "এটা লিখো: {t}"],
             ["likho {t}", "{t:en} type koro"],
             ["type {t:en}", "write {t:en}", "dictate: {t:en}", "type the words {t:en}"]),
    "keys": (["কন্ট্রোল এস চাপো", "এন্টার চাপো", "ট্যাবটা বন্ধ করো", "পেজটা রিফ্রেশ করো",
              "আন্ডু করো", "সব সিলেক্ট করো", "অল্ট ট্যাব চাপো"],
             ["enter chapo", "control s chapo", "page ta refresh koro"],
             ["press enter", "hit ctrl s", "refresh the page", "undo that", "select all",
              "press escape", "close this tab"]),
    "window": (["উইন্ডোটা ছোট করো", "এটা বড় করো", "{a} এ যাও", "ডেস্কটপ দেখাও",
                "সব উইন্ডো মিনিমাইজ করো", "উইন্ডোটা বামে সরাও", "কোন কোন উইন্ডো খোলা আছে",
                "এই উইন্ডোটা বন্ধ করো"],
               ["window ta choto koro", "desktop dekhao", "{a:en} e switch koro"],
               ["minimize this window", "maximize it", "switch to {a:en}", "show the desktop",
                "snap this window to the left", "what windows are open", "close this window"]),
    "files": (["{d} ফোল্ডার খোলো", "আমার {f} ফাইলটা খুঁজে দাও", "ডেস্কটপে {f} নামে ফোল্ডার বানাও",
               "{f} ফাইলটা ডিলিট করো", "{f} ফাইলটা ডকুমেন্টসে সরাও", "{f} ফাইলটা পড়ে শোনাও"],
              ["{d:en} folder kholo", "amar {f:en} file ta khujo", "{f:en} file ta delete koro"],
              ["open my {d:en} folder", "find my {f:en} file", "create a folder called {f:en}",
               "delete the {f:en} file", "move {f:en} to documents", "read the {f:en} file"]),
    "screenshot": (["স্ক্রিনশট নাও", "স্ক্রিনের ছবি তোলো", "একটা স্ক্রিনশট রাখো"],
                   ["screenshot nao", "ekta screenshot nao"],
                   ["take a screenshot", "capture the screen", "grab a screenshot"]),
    "screen": (["স্ক্রিনে কী আছে", "{b} বাটনে ক্লিক করো", "এখানে কী কী অপশন আছে দেখাও",
                "{b} এ চাপ দাও"],
               ["{b:en} button e click koro", "screen e ki ache"],
               ["what's on my screen", "click the {b:en} button", "what buttons are here",
                "press the {b:en} button"]),
    "settings": (["{p} সেটিংস খোলো", "{p} সেটিং এ যাও", "সেটিংসে {p} দেখাও"],
                 ["{p:en} settings kholo", "{p:en} setting e jao"],
                 ["open {p:en} settings", "go to {p:en} settings", "show me the {p:en} settings"]),
    "system": (["সিপিইউ কত ব্যবহার হচ্ছে", "র‍্যাম কতটুকু খালি আছে", "ব্যাটারি কত আছে",
                "জিপিইউর তাপমাত্রা কত", "ডিস্কে কত জায়গা আছে", "কম্পিউটারের অবস্থা কেমন"],
               ["ram koto use hocche", "cpu koto cholche"],
               ["how much ram is free", "cpu usage", "what's my battery level",
                "how hot is the gpu", "how much disk space is left"]),
    "lock": (["কম্পিউটার লক করো", "স্ক্রিন লক করে দাও", "পিসি লক করো"],
             ["computer lock koro", "pc ta lock kore dao"],
             ["lock the computer", "lock my screen", "lock the pc"]),
    "power": (["কম্পিউটার বন্ধ করো", "পিসি রিস্টার্ট দাও", "স্লিপ মোডে যাও", "সাইন আউট করো",
               "শাটডাউন বাতিল করো", "ল্যাপটপটা বন্ধ করে দাও"],
              ["computer bondho koro", "pc restart dao", "sleep mode e jao"],
              ["shut down the computer", "restart my pc", "put the computer to sleep",
               "sign me out", "cancel the shutdown"]),
    "clipboard": (["এটা কপি করো: {t}", "ক্লিপবোর্ডে কী আছে", "{t} কপি করে রাখো",
                   "ক্লিপবোর্ডের লেখাটা পড়ো"],
                  ["clipboard e ki ache", "{t:en} copy kore rakho"],
                  ["copy this: {t:en}", "what's on my clipboard", "put {t:en} on the clipboard",
                   "read my clipboard"]),
    "github": (["আমার গিটহাব নোটিফিকেশন দেখাও", "গিটহাবে আমার রিপোজিটরিগুলো কী কী",
                "কোনো পুল রিকোয়েস্ট রিভিউ করতে হবে", "গিটহাবে নতুন কী আছে"],
               ["github notification dekhao", "amar github repo gulo dekhao"],
               ["show my github notifications", "list my repositories",
                "do i have pull requests to review", "anything new on github"]),
    "question": (["{who} কে", "{what} কী", "{who} সম্পর্কে বলো", "{what} মানে কী",
                  "বাংলাদেশের রাজধানী কোথায়"],
                 ["{who:en} ke", "{what:en} ki jinish"],
                 ["who is {who:en}", "what is {what:en}", "tell me about {who:en}",
                  "define {what:en}"]),
    "command": (["{w} ইনস্টল করো", "উইনগেট দিয়ে {w} ইনস্টল করো", "আমার আইপি অ্যাড্রেস দেখাও",
                 "সব অ্যাপ আপডেট করো", "ইন্টারনেট কানেকশন চেক করো"],
                ["{w} install koro", "ip address dekhao"],
                ["install {w}", "run ipconfig", "update all my apps", "ping google",
                 "check my internet connection"]),
    "chat": (["তুমি কে", "ধন্যবাদ", "কেমন আছো", "আমার সাথে একটু গল্প করো", "শুভ সকাল",
              "একটা কৌতুক বলো", "তোমার নাম কী"],
             ["kemon acho", "thank you bhai", "tumi ke"],
             ["hi there", "thank you", "how are you", "tell me a joke", "good morning",
              "what's your name"]),
    "multi": (["{a} খুলে লেখো {t}", "{a} খোলো তারপর {q} সার্চ করো",
               "ক্রোম খুলে {s} এ যাও", "{a} খুলে স্ক্রিনশট নাও", "ইউটিউব খুলে {m} চালাও"],
              ["{a:en} khule {t} likho", "{a:en} kholo tarpor {q:en} search koro"],
              ["open {a:en} and type {t:en}", "open {a:en} then search for {q:en}",
               "go to {s:en} and then take a screenshot", "open youtube and play {m:en}"]),
}

SLOTS = {
    "a": APPS, "s": SITES, "c": CITIES, "f": FILES, "d": FOLDERS, "p": SETTINGS_PAGES,
    "n": NUMBERS, "k": MINUTES, "b": BUTTONS,
}
PAIRED = {"q": (QUERIES_BN, QUERIES_EN), "m": (MEDIA_BN, MEDIA_EN), "t": (TEXTS_BN, TEXTS_EN),
          "who": (PEOPLE_BN, PEOPLE_EN), "what": (THINGS_BN, THINGS_EN)}


def fill(template: str, rng: random.Random) -> str:
    out = template
    for key, values in SLOTS.items():
        for form, index in ((f"{{{key}:en}}", 1), (f"{{{key}}}", 0)):
            while form in out:
                out = out.replace(form, rng.choice(values)[index], 1)
    for key, (bn, en) in PAIRED.items():
        while f"{{{key}:en}}" in out:
            out = out.replace(f"{{{key}:en}}", rng.choice(en), 1)
        while f"{{{key}}}" in out:
            out = out.replace(f"{{{key}}}", rng.choice(bn), 1)
    out = out.replace("{w}", rng.choice(SOFTWARE))
    return out


PREFIX = {"bn": ["", "", "", "প্লিজ ", "একটু ", "সহায়ক, ", "আচ্ছা ", "এই, ", "hey jarvis, "],
          "banglish": ["", "", "plz ", "ektu ", "acha ", "hey jarvis, "],
          "en": ["", "", "", "please ", "hey jarvis, ", "can you ", "could you ", "ok "]}
SUFFIX = {"bn": ["", "", "", " প্লিজ", " তো", " একটু", "?", " জলদি"],
          "banglish": ["", "", " plz", " to", " please"],
          "en": ["", "", "", " please", " now", " for me", "?"]}
# Keep the real MASSIVE data dominant but stop "other" and "question" from swamping the
# rarer desktop actions, which come from far fewer distinct sentences.
CAPS = {"other": 3000, "question": 3000, "play": 2000, "reminder": 2000}
MIN_PER_ACTION = 400


def augment(text: str, variety: str, rng: random.Random) -> str:
    prefix, suffix = rng.choice(PREFIX[variety]), rng.choice(SUFFIX[variety])
    if prefix.startswith("can you") or prefix.startswith("could you"):
        text = text[0].lower() + text[1:]
    return f"{prefix}{text}{suffix}".strip()


def balance(rows: list[dict], templates: list[dict], rng: random.Random) -> list[dict]:
    by_action: dict[str, list[dict]] = {}
    for row in rows:
        by_action.setdefault(row["expected"]["action"], []).append(row)
    out = []
    for action, items in by_action.items():
        rng.shuffle(items)
        out += items[: CAPS.get(action, len(items))]
    counts = Counter(r["expected"]["action"] for r in out)
    pool: dict[str, list[dict]] = {}
    for row in templates:
        pool.setdefault(row["expected"]["action"], []).append(row)
    for action, items in pool.items():
        need = MIN_PER_ACTION - counts.get(action, 0)
        for _ in range(max(0, need)):
            base = rng.choice(items)
            variety = base["source"].removeprefix("template-")
            text = augment(base["state"], variety, rng)
            out.append(question_row(text, action, rng, base["source"] + "-augmented"))
    return out


def question_row(state: str, action: str, rng: random.Random, source: str) -> dict:
    language = rng.choice(["bn", "bn", "en"])
    wording = rng.choice(TRAIN_QUESTIONS[language])
    criteria = {name: getattr(a, language) for name, a in ACTIONS.items()}
    return {
        "state": state, "source": source,
        "questions": {"action": {"type": "choice", "instructions": wording, "criteria": criteria}},
        "expected": {"action": action},
    }


def massive_rows(folder: Path, split: str, rng: random.Random) -> list[dict]:
    rows = []
    for language in ("bn", "en"):
        with gzip.open(folder / f"{split}.{language}.json.gz", "rt", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                rows.append(question_row(item["text"], MASSIVE_ACTION[item["label"]], rng,
                                         f"massive-{language}"))
    return rows


def template_rows(per_template: int, rng: random.Random, held_out: list[str]) -> list[dict]:
    held = [normalize(text) for text in held_out]
    rows, seen, dropped = [], set(), 0
    for action, groups in TEMPLATES.items():
        for variety, templates in zip(("bn", "banglish", "en"), groups, strict=True):
            for template in templates:
                for _ in range(per_template):
                    text = fill(template, rng)
                    key = normalize(text)
                    if key in seen:
                        continue
                    seen.add(key)
                    if any(fuzz.ratio(key, h) >= 90 for h in held):
                        dropped += 1
                        continue
                    rows.append(question_row(text, action, rng, f"template-{variety}"))
    print(f"templates: {len(rows)} sentences, {dropped} dropped as too close to the benchmark")
    return rows


def held_out_commands() -> list[str]:
    text = resources.files("sohayok.bench").joinpath("desktop_commands.jsonl").read_text("utf-8")
    return [json.loads(line)["text"] for line in text.splitlines() if line.strip()]


def build(massive: Path, output: Path, seed: int = 20261018, per_template: int = 12) -> dict:
    rng = random.Random(seed)
    templates = template_rows(per_template, rng, held_out_commands())
    rng.shuffle(templates)
    cut = len(templates) // 10
    train_templates, val_templates = templates[cut:], templates[:cut]
    splits = {
        "train": balance(massive_rows(massive, "train", rng) + train_templates, train_templates,
                         rng),
        "validation": massive_rows(massive, "validation", rng) + val_templates,
    }
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, rows in splits.items():
        rng.shuffle(rows)
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary[name] = {"rows": len(rows),
                         "actions": dict(Counter(r["expected"]["action"] for r in rows)),
                         "sources": dict(Counter(r["source"] for r in rows))}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), "utf-8")
    return summary


if __name__ == "__main__":
    import sys

    print(json.dumps(build(Path(sys.argv[1]), Path(sys.argv[2])), indent=2, ensure_ascii=False))
