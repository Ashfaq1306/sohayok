"""Round 5: notification clearing, zoom, closing every tab of a site, new abilities, memory,
and the on-screen agent's loop."""

from types import SimpleNamespace as NS

import pytest

from sohayok.router import route


@pytest.mark.parametrize(
    "text, tool, args",
    [
        ("close my whatsapp notification", "dismiss_notifications", {"app": "WhatsApp"}),
        ("নোটিফিকেশনগুলো ক্লিয়ার করো", "dismiss_notifications", {"app": ""}),
        ("clear all notifications", "dismiss_notifications", {"app": ""}),
        ("তুমি এটিও পিডিএফ সেট জুম করে তো দেখি", "press_keys", {"keys": "ctrl+plus"}),
        ("জুম আউট করো", "press_keys", {"keys": "ctrl+minus"}),
        (
            "পূর্বের যতগুলো ইউটিউবের ট্যাব খোলা আছে সবগুলো ক্লোজ করে দাও",
            "close_tab",
            {"name": "ইউটিউবের", "every": True},
        ),
        ("ইউটিউবের ট্যাবটা বন্ধ করো", "close_tab", {"name": "ইউটিউবের", "every": False}),
        ("dishpc ওপেন করো", "open_path", {"path": "dishpc"}),
        ("dishpc অন করো", "open_path", {"path": "dishpc"}),
        ("ব্লুটুথ অন করো", "bluetooth_power", {"on": True}),
    ],
)
def test_round5_rules(text, tool, args):
    intent = route(text)
    assert intent is not None, text
    assert (intent.tool, intent.args) == (tool, args)


def test_zoom_app_and_google_open_are_left_alone():
    assert route("open zoom") is None or route("open zoom").tool != "press_keys"
    intent = route("Google Prome ওপেন করতো")
    assert intent is None or intent.tool != "web_search"


@pytest.mark.parametrize(
    "text, tool",
    [
        ("মনে রাখো আমার অফিস সকাল নয়টায়", "remember"),
        ("স্ক্রিনের ব্রাইটনেস কমাও", "brightness"),
        ("ডার্ক মোড চালু করো", "dark_mode"),
        ("আমার প্রিন্টার কানেক্ট আছে কিনা দেখো", "list_printers"),
        ("তুমি আমার পিসির ফুল কনফিগারেশনটা লেখা যাও", "pc_specs"),
    ],
)
def test_new_abilities_are_found_by_keyword(text, tool):
    from sohayok.brain.lite import keyword_action

    action = keyword_action(text)
    assert action is not None and action.tools[0] == tool


def test_sums_are_answered_not_searched():
    from sohayok.brain.lite import ARITHMETIC

    assert ARITHMETIC.search("সাতাশকে পাঁচ দিয়ে ভাগ করলে কত হয়")
    assert ARITHMETIC.search("what is 27 / 5")
    assert not ARITHMETIC.search("ক্রোম খোলো")


def test_memory_is_kept_and_given_to_the_model(private_app_dir):
    from sohayok.brain.base import user_message
    from sohayok.tools import memory

    assert memory.context() == ""
    memory.remember("আমার অফিস সকাল নয়টায়")
    memory.remember("my sister is Rima")
    assert "নয়টায়" in user_message("কখন অফিস?", "bn", "") and "Rima" in memory.context()
    assert memory.forget("rima")["forgotten"] == 1
    assert memory.forget("nothing like this")["forgotten"] == 0
    assert [f["fact"] for f in memory.facts()] == ["আমার অফিস সকাল নয়টায়"]


def test_brightness_steps():
    from sohayok.tools.display import _target

    assert _target("up", None, 50) == 70 and _target("down", 30, 50) == 20
    assert _target("set", 140, 50) == 100 and _target("down", None, 10) == 0


def fake_claude(turns, seen):
    """A client whose replies are `turns`, one per request; requests are kept in `seen`."""

    def create(**request):
        seen.append(request)
        blocks = turns[len(seen) - 1]
        content = [NS(model_dump=lambda exclude_none=True, b=b: b, **b) for b in blocks]
        uses = any(b["type"] == "tool_use" for b in blocks)
        return NS(stop_reason="tool_use" if uses else "end_turn", content=content)

    return NS(beta=NS(messages=NS(create=create)))


def computer_tool(id_, name, **args):
    return {"type": "tool_use", "id": id_, "name": name, "input": args, "toolset_name": "computer"}


def test_screen_agent_loop(monkeypatch):
    from sohayok.brain import claude
    from sohayok.tools import computer

    ran = []

    def run(self, name, args):
        ran.append(name)
        if name == "left_click":
            raise RuntimeError("no mouse")
        return [{"type": "text", "text": "ok"}]

    monkeypatch.setattr(claude.ComputerUse, "run", run)
    monkeypatch.setattr(claude.ComputerUse, "screenshot_block", lambda self: {"type": "image"})
    asked = []
    monkeypatch.setattr(computer, "ASK", lambda q: asked.append(q) or True)
    monkeypatch.setattr(computer, "SPEAKING", lambda: "bn")
    seen = []
    turns = [
        [
            computer_tool("a", "key", text="super"),
            computer_tool("b", "left_click"),
            computer_tool("c", "type", text="x"),
        ],
        [{"type": "tool_use", "id": "d", "name": "ask_user", "input": {"question": "Send?"}}],
        [{"type": "text", "text": "পেইন্টে বৃত্ত আঁকলাম।"}],
    ]
    monkeypatch.setattr(computer, "_client_and_model", lambda: (fake_claude(turns, seen), "m"))
    out = computer.use_computer("draw a circle in Paint")
    assert out["done"] and out["spoken_bn"] == "পেইন্টে বৃত্ত আঁকলাম।"
    # The batch stops at the failed click; the rest is reported as not executed.
    assert ran == ["key", "left_click"]
    results = seen[1]["messages"][2]["content"]
    assert all(r["toolset_name"] == "computer" for r in results)
    assert results[1]["is_error"] and results[2]["content"] == computer.HALT
    assert asked == ["Send?"] and seen[2]["messages"][4]["content"][0]["content"] == "yes"
    # Every request is cached and opts in to the refusal fallback.
    assert seen[0]["cache_control"] == {"type": "ephemeral"} and seen[0]["fallbacks"] == "default"


def test_screen_agent_stops_when_told(monkeypatch):
    from sohayok.brain import claude
    from sohayok.tools import computer

    monkeypatch.setattr(claude.ComputerUse, "screenshot_block", lambda self: {"type": "image"})
    monkeypatch.setattr(computer, "SPEAKING", lambda: "en")

    def run(self, name, args):
        computer.STOP.set()  # the user said "stop" while this action ran
        return [{"type": "text", "text": "ok"}]

    monkeypatch.setattr(claude.ComputerUse, "run", run)
    turns = [[computer_tool("a", "key", text="a"), computer_tool("b", "key", text="b")]]
    monkeypatch.setattr(computer, "_client_and_model", lambda: (fake_claude(turns, []), "m"))
    out = computer.use_computer("type ab")
    assert not out["done"] and out["spoken_en"] == "Stopped."
