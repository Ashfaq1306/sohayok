import threading
import time

import pytest

from sohayok.safety import ConfirmationBroker, Executor, parse_answer
from sohayok.tools.registry import Registry, Risk, Tool


@pytest.mark.parametrize(
    "text", ["হ্যাঁ", "হ্যাঁ করো", "জি", "ঠিক আছে", "yes", "Yes, do it.", "ok", "haan", "করুন"]
)
def test_yes(text):
    assert parse_answer(text) is True


@pytest.mark.parametrize(
    "text", ["না", "না না", "no", "No thanks", "cancel", "বাতিল", "করো না", "দরকার নেই", "naa"]
)
def test_no(text):
    assert parse_answer(text) is False


@pytest.mark.parametrize("text", ["ক্রোম খোলো", "what time is it", ""])
def test_neither(text):
    assert parse_answer(text) is None


def _registry(calls):
    registry = Registry()
    registry.add(
        Tool("safe", "", {"type": "object", "properties": {}}, lambda: calls.append("safe"))
    )
    registry.add(
        Tool(
            "risky",
            "",
            {"type": "object", "properties": {}},
            lambda: calls.append("risky") or "deleted",
            Risk.CONFIRM,
            lambda a, lang: "Delete everything",
        )
    )
    return registry


def test_safe_tools_run_without_asking():
    calls = []
    broker = ConfirmationBroker(timeout=0.1)
    asked = []
    broker.on_ask = lambda p: asked.append(p.question)
    assert Executor(_registry(calls), broker).run("safe", {}) == "done"
    assert calls == ["safe"] and asked == []


def test_risky_tool_runs_only_after_yes():
    calls = []
    broker = ConfirmationBroker(timeout=5)
    asked = []

    def answer_later(pending):
        asked.append(pending.question)
        threading.Thread(target=lambda: (time.sleep(0.05), broker.resolve(True))).start()

    broker.on_ask = answer_later
    assert Executor(_registry(calls), broker).run("risky", {}) == "deleted"
    assert calls == ["risky"] and asked == ["Delete everything"]


def test_risky_tool_is_skipped_on_no_or_timeout():
    calls = []
    broker = ConfirmationBroker(timeout=0.05)
    result = Executor(_registry(calls), broker).run("risky", {})
    assert result.startswith("cancelled") and calls == []
    broker.on_ask = lambda p: broker.resolve(False)
    assert Executor(_registry(calls), broker).run("risky", {}).startswith("cancelled")
    assert calls == []


def test_confirmation_can_be_turned_off():
    calls = []
    executor = Executor(_registry(calls), ConfirmationBroker(), confirm=False)
    assert executor.run("risky", {}) == "deleted"


def test_focus_hook_runs_before_active_window_tools():
    order = []
    registry = Registry()
    registry.add(Tool("type_text", "", {}, lambda text: order.append("type")))
    registry.add(Tool("volume", "", {}, lambda: order.append("volume")))
    executor = Executor(registry, ConfirmationBroker())
    executor.focus_target = lambda: order.append("focus")
    executor.run("type_text", {"text": "hi"})
    executor.run("volume", {})
    assert order == ["focus", "type", "volume"]


def test_resolve_without_question_is_harmless():
    assert ConfirmationBroker().resolve(True) is False
