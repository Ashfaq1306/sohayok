import numpy as np
import pytest

from sohayok import gpu
from sohayok.lang import has_bangla_grammar
from sohayok.router import route
from sohayok.stt import Transcriber
from sohayok.tools import files


def test_bangla_grammar_tells_mixed_bangla_from_phonetic_english():
    # the user's real commands, as the Bangla model wrote them
    assert has_bangla_grammar("ইউটিউবের ট্যাপটা কেটে ফেলো")
    assert has_bangla_grammar("দিস পিসি ওপেন করো তো")
    assert has_bangla_grammar("গেম আপন করো")
    assert has_bangla_grammar("ক্রোম")  # one-word commands stay Bangla
    # English speech written in Bangla letters
    assert not has_bangla_grammar("প্লিজ ওপেন ডিস পিসি")
    assert not has_bangla_grammar("সেট এ টাইমার ফর ফাইভ মিনিটস")
    assert not has_bangla_grammar("হোয়াট আর ইউ ডুয়িং")


class Model:
    def __init__(self, text):
        self.text, self.calls = text, 0

    def transcribe(self, audio, language):
        self.calls += 1
        return self.text


class Detector:
    def __init__(self, margin):
        self.value = margin

    def margin(self, audio, first, second):
        return self.value


def transcriber(margin, bangla, english, monkeypatch):
    import sohayok.stt as stt

    monkeypatch.setattr(stt, "speech_seconds", lambda audio, **_: 1.0)
    t = Transcriber.__new__(Transcriber)
    t.languages, t.detector = ["bn", "en"], Detector(margin)
    t.bangla, t.multi = Model(bangla), Model(english)
    return t


def test_mixed_bangla_goes_to_the_bangla_model_once(monkeypatch):
    t = transcriber(4.5, "ইউটিউবের ট্যাবটা কেটে ফেলো", "YouTube tab KT Fellow", monkeypatch)
    heard = t.transcribe(np.zeros(16_000, np.float32))
    assert (heard.text, heard.language) == ("ইউটিউবের ট্যাবটা কেটে ফেলো", "bn")
    assert (t.bangla.calls, t.multi.calls) == (1, 0)


def test_phonetic_english_is_redone_in_english(monkeypatch):
    t = transcriber(6.5, "প্লিজ ওপেন ডিস পিসি", "Please open this PC", monkeypatch)
    heard = t.transcribe(np.zeros(16_000, np.float32))
    assert (heard.text, heard.language) == ("Please open this PC", "en")


def test_clear_english_skips_the_bangla_model(monkeypatch):
    t = transcriber(-12.0, "", "What's the weather like?", monkeypatch)
    heard = t.transcribe(np.zeros(16_000, np.float32))
    assert (heard.text, heard.language, t.bangla.calls) == ("What's the weather like?", "en", 0)


def test_gpu_priority_is_raised_once_and_restored(monkeypatch):
    calls = []
    monkeypatch.setattr(gpu, "set_priority", lambda level: calls.append(level) or True)
    with gpu.priority("realtime"):
        with gpu.priority("realtime"):
            pass
    assert calls == ["realtime", "normal"]
    with gpu.priority("normal"):
        pass
    assert calls == ["realtime", "normal"]


def test_drives_and_places():
    assert files.shell_place("দিস পিসি") == files.shell_place("This PC") == "shell:MyComputerFolder"
    assert files.drive("ডি ড্রাইভে") == files.drive("drive d") == files.drive("D drive")
    assert files.drive("C:") is not None
    assert files.drive("hello") is None


@pytest.mark.parametrize(
    ("text", "tool", "args"),
    [
        ("দিস পিসি ওপেন করো তো", "open_path", {"path": "দিস পিসি"}),
        ("please open this pc", "open_path", {"path": "this pc"}),
        ("go to c drive", "open_path", {"path": "c drive"}),
        ("একটা নতুন ফোল্ডার বানাও", "create_folder", {"name": "", "location": ""}),
        ("হিসাব নামে একটা ফোল্ডার বানাও", "create_folder", {"name": "হিসাব", "location": ""}),
        ("একটা ফোল্ডার বানাও নাম হিসাব", "create_folder", {"name": "হিসাব", "location": ""}),
        ("এখানে একটা ফোল্ডার বানাও", "create_folder", {"name": "", "location": ""}),
        ("নতুন ফোল্ডার খোলো", "create_folder", {"name": "", "location": ""}),
        (
            "create a new folder called reports",
            "create_folder",
            {"name": "reports", "location": ""},
        ),
        ("make a new folder here", "create_folder", {"name": "", "location": ""}),
    ],
)
def test_explorer_commands(text, tool, args):
    intent = route(text)
    assert intent is not None and (intent.tool, intent.args) == (tool, args)


def test_open_folder_is_not_a_new_folder():
    intent = route("ফোল্ডার খোলো")
    assert intent is None or intent.tool != "create_folder"


def test_create_folder_names_like_explorer(tmp_path, monkeypatch):
    monkeypatch.setattr(files, "explorer_folder", lambda: tmp_path)
    first = files.create_folder()
    second = files.create_folder()
    named = files.create_folder("হিসাব")
    assert [p.name for p in sorted(tmp_path.iterdir())] == ["New folder", "New folder (2)", "হিসাব"]
    assert first["spoken_en"].startswith("Created the folder “New folder”")
    assert "New folder (2)" in second["created"] and named["spoken_bn"].endswith("ফোল্ডার বানালাম।")


def test_bangla_names_match_contacts_saved_in_english():
    from sohayok.tools.whatsapp import MATCH, name_score
    from sohayok.translit import to_latin

    assert to_latin("রহিম") == "rohim" and to_latin("সাদিয়া") == "sadiya"
    for said, saved in [
        ("রহিম", "Rahim"),
        ("আশফাক", "Ashfaq"),
        ("করিম ভাই", "Karim Bhai"),
        ("আম্মু", "Ammu"),
        ("rahim", "Rahim Uddin"),
    ]:
        assert name_score(said, saved) >= MATCH, (said, saved)
    assert name_score("রহিম", "Karim") < name_score("রহিম", "Rahim")
    assert name_score("আম্মু", "Office group") < MATCH


def test_call_commands():
    for text, contact, video in [
        ("রহিমকে কল দাও", "রহিম", False),
        ("আম্মুকে হোয়াটসঅ্যাপে ভিডিও কল দাও", "আম্মু", True),
        ("call rahim on whatsapp", "rahim", False),
    ]:
        intent = route(text)
        assert (intent.tool, intent.args) == ("whatsapp_call", {"contact": contact, "video": video})


def test_mixed_bangla_rules_on_real_mistakes():
    # English the Bangla model wrote phonetically, with a Bangla-looking ending
    assert not has_bangla_grammar("ওয়ের এস ভলিউম সেটিং")
    assert has_bangla_grammar("ইউটিউবের ট্যাবটা")


def test_whatsapp_searches_english_spellings_first():
    from sohayok.tools.whatsapp import MATCH, name_score, search_queries

    queries = search_queries("পিয়াস ভাই")
    assert queries[0] in {"pias", "piyas"} and "piyas" in queries  # not "ভাই", Latin first
    assert search_queries("পি আশ ভাই")[0] == "piash"  # a name the recognizer split
    assert "piyas" in search_queries("পিয়াজ ভাইয়ে")  # জ heard for স
    for said, saved in [
        ("পিয়াস ভাই", "Pias Vai"),
        ("পিয়াজ ভাইয়ে", "Piyas Bhai"),
        ("পিয়াজ ভাইয়ে", "Pias Vai"),
    ]:  # the user's real contact and mishearing
        assert name_score(said, saved) >= MATCH
    assert name_score("পিয়াস ভাই", "Rakib Bhai") < MATCH


def test_keyword_routes_for_messages():
    from sohayok.brain.lite import keyword_action

    assert keyword_action("নোটিফিকেশনে কী আছে").tools[0] == "read_notifications"
    assert keyword_action("হোয়াটসঅ্যাপে কার মেসেজ এসেছে").tools[0] == "read_notifications"
    assert keyword_action("রিপ্লাই দাও আমি আসছি").tools[0] == "whatsapp_reply"
    assert keyword_action("ওয়াটস্যাপ থেকে পিয়াস ভাইকে ফোন দাও").tools == ("whatsapp_call",)


def test_signed_out_whatsapp_is_explained_before_asking(monkeypatch):
    from sohayok.safety import ConfirmationBroker, Executor
    from sohayok.tools import load_all, whatsapp

    def signed_out(args):
        raise whatsapp.NOT_SIGNED_IN

    registry = load_all()
    monkeypatch.setattr(registry.get("whatsapp_call"), "check", signed_out)
    broker = ConfirmationBroker(1)
    asked = []
    broker.ask = lambda question, language: asked.append(question) or True
    result = Executor(registry, broker).run("whatsapp_call", {"contact": "রহিম"})
    assert asked == [] and result.startswith("error: {")
    from sohayok.brain.lite import _prepared_reply

    assert "লগইন" in _prepared_reply(result, "bn")


def test_bluetooth_and_device_commands():
    cases = [
        ("ব্লুটুথ অফ করো", "bluetooth_power", {"on": False}),
        ("turn on bluetooth", "bluetooth_power", {"on": True}),
        (
            "জেবিএল স্পিকারটা ডিসকানেক্ট করো",
            "bluetooth_device",
            {"device": "জেবিএল স্পিকার", "action": "disconnect"},
        ),
        ("connect to my earbuds", "bluetooth_device", {"device": "earbuds", "action": "connect"}),
    ]
    for text, tool, args in cases:
        intent = route(text)
        assert (intent.tool, intent.args) == (tool, args), text


def test_picking_a_bluetooth_device():
    from sohayok.tools.devices import pick_device

    class Button:
        def __init__(self, name):
            self.Name = name

    cards = [
        ("JBL Charge 5", "speakers, battery unknown, state connected", Button("Disconnect")),
        ("WF-1000XM5", "communications device, state not connected", Button("Connect")),
    ]
    assert pick_device(cards, "জেবিএল স্পিকার", "disconnect")[0] == "JBL Charge 5"
    assert pick_device(cards, "হেডফোন", "connect")[0] == "WF-1000XM5"
    assert pick_device(cards, "Sony", "connect")[0] == "WF-1000XM5"
    assert pick_device(cards, "headphones", "disconnect") is None  # not connected


def test_notification_payloads():
    from sohayok.tools.notifications import app_name, toast_lines

    xml = (
        "<toast><visual><binding template='ToastGeneric'><text>Piyas Bhai</text>"
        "<text>আসছো?</text></binding></visual></toast>"
    )
    assert toast_lines(xml) == ["Piyas Bhai", "আসছো?"]
    assert app_name("5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App") == "WhatsApp"
