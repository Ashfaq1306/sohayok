import pytest

from sohayok.router import route


@pytest.mark.parametrize(
    "text, tool, args",
    [
        ("ক্রোম খোলো", "open_app", {"name": "Google Chrome"}),
        ("chrome kholo", "open_app", {"name": "Google Chrome"}),
        ("open notepad", "open_app", {"name": "Notepad"}),
        ("ভিএস কোড খোলো", "open_app", {"name": "Visual Studio Code"}),
        ("ক্যালকুলেটর চালু করো", "open_app", {"name": "Calculator"}),
        ("নোটপ্যাড বন্ধ করো", "close_app", {"name": "Notepad"}),
        ("ইউটিউব খোলো", "open_website", {"site": "https://www.youtube.com"}),
        ("জিমেইল খোলো", "open_website", {"site": "https://mail.google.com"}),
        ("open prothomalo.com", "open_website", {"site": "https://prothomalo.com"}),
        ("ভলিউম বাড়াও", "volume", {"action": "up", "level": 10}),
        ("volume barao", "volume", {"action": "up", "level": 10}),
        ("volume ektu komao", "volume", {"action": "down", "level": 10}),
        ("ভলিউম একটু বাড়াও", "volume", {"action": "up", "level": 10}),
        ("turn the volume down", "volume", {"action": "down", "level": 10}),
        ("ভলিউম পঞ্চাশ করো", "volume", {"action": "set", "level": 50}),
        ("ভলিউম ৩০ করো", "volume", {"action": "set", "level": 30}),
        ("set volume to 70", "volume", {"action": "set", "level": 70}),
        ("মিউট করো", "volume", {"action": "mute"}),
        ("গান থামাও", "media", {"action": "playpause"}),
        ("পরের গান", "media", {"action": "next"}),
        ("পরের গানটা দাও", "media", {"action": "next"}),
        ("gaan ta thamao", "media", {"action": "playpause"}),
        ("ভিডিওটা থামাও", "media", {"action": "playpause"}),
        ("কয়টা বাজে?", "current_time", {}),
        ("what time is it", "current_time", {}),
        ("আজ কত তারিখ", "current_time", {}),
        ("স্ক্রিনশট নাও", "take_screenshot", {}),
        ("lock the computer", "lock_screen", {}),
        ("ইউটিউবে আইয়ুব বাচ্চুর গান চালাও", "play_youtube", {"query": "আইয়ুব বাচ্চুর গান"}),
        ("play lofi beats on youtube", "play_youtube", {"query": "lofi beats"}),
        (
            "search for best biryani in dhaka",
            "web_search",
            {"query": "best biryani in dhaka", "engine": "google"},
        ),
        ("বাংলাদেশের রাজধানী সার্চ করো", "web_search", {"query": "বাংলাদেশের রাজধানী", "engine": "google"}),
        ("open bluetooth settings", "open_settings", {"page": "bluetooth"}),
        ("ওয়াইফাই সেটিংস খোলো", "open_settings", {"page": "wifi"}),
        ("নতুন ট্যাব খোলো", "press_keys", {"keys": "ctrl+t"}),
        ("সহায়ক, ক্রোম খোলো", "open_app", {"name": "Google Chrome"}),
        # Spellings the speech recognizer actually produced in the voice benchmark.
        ("ক্রম বন্ধ করো", "close_app", {"name": "Google Chrome"}),
        ("এখন কয়েকটা বাজে", "current_time", {}),
        ("নতুন ট্যাপ খোলো", "press_keys", {"keys": "ctrl+t"}),
    ],
)
def test_fast_path(text, tool, args):
    intent = route(text)
    assert intent is not None, text
    assert (intent.tool, intent.args) == (tool, args)
    assert intent.reply["bn"] and intent.reply["en"]


def test_known_folder_resolves_to_a_path():
    intent = route("ডাউনলোড ফোল্ডার খোলো")
    assert intent.tool == "open_path"
    assert intent.args["path"].endswith("Downloads")
    assert "ফোল্ডার ফোল্ডার" not in intent.reply["bn"]


@pytest.mark.parametrize(
    "text",
    [
        "open chrome and search for cats",  # multi-step: needs the model
        "ক্রোম খুলে বিড়ালের ছবি সার্চ করো আর ডাউনলোড করো",
        "হ্যালো কেমন আছো",
        "open the thing I used yesterday",  # no app matches
        "রবীন্দ্রনাথ ঠাকুর কে ছিলেন?",
        "set volume to 500",
        "",
    ],
)
def test_leaves_everything_else_to_the_model(text):
    assert route(text) is None


def test_open_then_type_is_one_instant_command():
    intent = route("নোটপ্যাড খুলে লেখো আমি ভালো আছি")
    assert (intent.tool, intent.args) == ("open_app", {"name": "Notepad"})
    assert intent.then == [("type_text", {"text": "আমি ভালো আছি"})]
    english = route("open notepad and type hello world")
    assert english.then == [("type_text", {"text": "hello world"})]
