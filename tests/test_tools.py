import json
import time
from datetime import datetime

import httpx
import pytest

from sohayok.tools import browser, files, timers
from sohayok.tools.apps import resolve_app
from sohayok.tools.registry import ToolError

APPS = {"Google Chrome": "x", "Notepad": "y", "Visual Studio Code": "z", "Telegram Desktop": "t"}


@pytest.mark.parametrize(
    "spoken, app",
    [
        ("chrome", "Google Chrome"),
        ("ক্রোম", "Google Chrome"),
        ("notepad", "Notepad"),
        ("নোটপ্যাড", "Notepad"),
        ("vs code", "Visual Studio Code"),
        ("telegram", "Telegram Desktop"),
        ("Visual Studio Code", "Visual Studio Code"),
    ],
)
def test_resolve_app(spoken, app):
    assert resolve_app(spoken, APPS) == app


def test_resolve_app_rejects_unknown_names():
    assert resolve_app("photoshop", APPS) is None


def test_site_url():
    assert browser.site_url("ইউটিউব") == "https://www.youtube.com"
    assert browser.site_url("Gmail") == "https://mail.google.com"
    assert browser.site_url("prothomalo.com") == "https://prothomalo.com"
    assert browser.site_url("https://x.org/a?b=1") == "https://x.org/a?b=1"
    with pytest.raises(ToolError):
        browser.site_url("something unknown")


def test_first_youtube_video_parses_search_results():
    page = (
        '...{"videoRenderer":{"videoId":"dQw4w9WgXcQ","thumbnail":{},'
        '"title":{"runs":[{"text":"Shei Tumi \\u2014 Ayub Bachchu"}]}}...'
    )
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=page)))
    assert browser.first_youtube_video("shei tumi", client) == (
        "dQw4w9WgXcQ",
        "Shei Tumi — Ayub Bachchu",
    )


def test_web_search_opens_the_right_url(monkeypatch):
    opened = []
    monkeypatch.setattr(browser, "open_in_browser", opened.append)
    browser.web_search("ঢাকার আবহাওয়া", "google")
    browser.web_search("tagore", "wikipedia")
    assert opened[0].startswith("https://www.google.com/search?q=%E0%A6")
    assert opened[1] == "https://en.wikipedia.org/w/index.php?search=tagore"


def test_parse_clock():
    now = datetime(2026, 10, 18, 9, 0)
    assert timers.parse_clock("7 pm", now) == datetime(2026, 10, 18, 19, 0)
    assert timers.parse_clock("18:30", now) == datetime(2026, 10, 18, 18, 30)
    assert timers.parse_clock("8:15 am", now) == datetime(2026, 10, 19, 8, 15)  # tomorrow
    with pytest.raises(ToolError):
        timers.parse_clock("half past", now)


def test_scheduler_fires_and_persists(tmp_path):
    path = tmp_path / "reminders.json"
    scheduler = timers.Scheduler(path)
    first = scheduler.add(time.time() - 1, "চা খাওয়ার সময়")
    scheduler.add(time.time() + 3600, "later")
    assert [r.text for r in timers.Scheduler(path).pending()] == ["চা খাওয়ার সময়", "later"]
    due = scheduler.due_now()
    assert [r.id for r in due] == [first.id]
    assert [r["text"] for r in json.loads(path.read_text("utf-8"))] == ["later"]
    assert [r.text for r in timers.Scheduler(path).pending()] == ["later"]


def test_scheduler_thread_announces(tmp_path):
    scheduler = timers.Scheduler()
    fired = []
    scheduler.on_fire = fired.append
    scheduler.start()
    scheduler.add(time.time() + 0.2, "ding")
    for _ in range(50):
        if fired:
            break
        time.sleep(0.05)
    assert [r.text for r in fired] == ["ding"]


def test_find_files_fuzzy(tmp_path):
    (tmp_path / "Docs").mkdir()
    (tmp_path / "Docs" / "My Resume 2026.pdf").write_text("x")
    (tmp_path / "Docs" / "notes.txt").write_text("x")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "resume.js").write_text("x")
    found = files.find("resume", [tmp_path])
    assert [f["path"].endswith("My Resume 2026.pdf") for f in found] == [True]


def test_known_folders_and_recycle_guard(tmp_path):
    assert files.known_folder("ডাউনলোড").name == "Downloads"
    assert files.known_folder("downloads folder").name == "Downloads"
    with pytest.raises(ToolError):
        files.recycle(str(files.HOME))


def test_take_note(tmp_path, monkeypatch):
    monkeypatch.setattr(files, "known_folder", lambda name: tmp_path)
    files.take_note("কাল মিটিং")
    assert "কাল মিটিং" in (tmp_path / "Sohayok" / "notes.md").read_text("utf-8")


def test_type_text_pastes_bangla_and_types_short_ascii(monkeypatch):
    from sohayok.tools import input as keyboard

    done = []
    monkeypatch.setattr(keyboard, "_prepare_target", lambda: "Text editor in Untitled - Notepad")
    monkeypatch.setattr(keyboard, "paste_text", lambda text: done.append(("paste", text)))
    monkeypatch.setattr(keyboard.winapi, "type_unicode", lambda text: done.append(("keys", text)))
    monkeypatch.setattr(
        keyboard.winapi, "press", lambda keys, repeat=1: done.append(("press", keys))
    )
    assert keyboard.type_text("আমি ভালো আছি").endswith("'Text editor in Untitled - Notepad'")
    keyboard.type_text("hello", press_enter=True)
    assert done == [("paste", "আমি ভালো আছি"), ("keys", "hello"), ("press", "enter")]


def test_paste_restores_the_users_clipboard(monkeypatch):
    from sohayok.tools import input as keyboard

    clipboard = ["user's text"]
    monkeypatch.setattr(keyboard, "_clipboard_text", lambda: clipboard[0])
    monkeypatch.setattr(keyboard, "_set_clipboard", lambda text: clipboard.__setitem__(0, text))
    pasted = []
    monkeypatch.setattr(
        keyboard.winapi, "press", lambda keys, repeat=1: pasted.append(clipboard[0])
    )
    monkeypatch.setattr(keyboard.time, "sleep", lambda s: None)
    keyboard.paste_text("বাংলা")
    assert pasted == ["বাংলা"] and clipboard == ["user's text"]


def test_play_youtube_falls_back_to_search_results(monkeypatch):
    opened = []
    monkeypatch.setattr(browser, "open_in_browser", opened.append)

    def nothing(query, client=None):
        raise ToolError("no video")

    monkeypatch.setattr(browser, "first_youtube_video", nothing)
    assert "search results" in browser.play_youtube("some shorts only")
    assert opened == ["https://www.youtube.com/results?search_query=some+shorts+only"]
