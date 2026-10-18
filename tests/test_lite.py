import json

import httpx

from sohayok.brain.base import Reply
from sohayok.brain.lite import LiteBrain
from sohayok.decide import ACTIONS, TOOL_ACTION, Decision, questions
from sohayok.safety import ConfirmationBroker, Executor
from sohayok.tools import load_all
from sohayok.tools.registry import Registry, Tool


class Decider:
    def __init__(self, action, probability=0.9):
        self.decision = Decision(action, probability, {action: probability})

    def decide(self, text, language="bn"):
        return self.decision


def _executor(calls):
    registry = Registry()
    for tool in load_all().tools.values():

        def run(_name=tool.name, **kwargs):
            calls.append((_name, kwargs))
            return {"current_time": "সকাল ১০টা", "weather": '{"now": "31°C"}'}.get(_name, "ok")

        registry.add(
            Tool(tool.name, tool.description, tool.parameters, run, tool.risk, tool.summary)
        )
    executor = Executor(registry, ConfirmationBroker(), confirm=False)
    executor.language = "bn"
    return executor


def _server(replies, seen):
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"message": replies[len(seen) - 1]})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_every_tool_belongs_to_one_action():
    names = set(load_all().names())
    assert set(TOOL_ACTION) <= names
    missing = (
        names - set(TOOL_ACTION) - {"find_apps", "move_mouse", "click", "scroll", "ask_claude_code"}
    )
    assert not missing, missing
    assert len(questions("bn")["action"]["criteria"]) == len(ACTIONS)


def test_simple_action_runs_without_the_model():
    calls, seen = [], []
    brain = LiteBrain(Decider("time"), client=_server([], seen))
    reply = brain.respond("কয়টা বাজে", "bn", _executor(calls), [], {})
    assert calls == [("current_time", {"language": "bn"})] and seen == []
    assert reply.text == "এখন সকাল ১০টা।"


def test_small_model_sees_only_the_actions_tools_and_template_replies():
    calls, seen = [], []
    client = _server(
        [
            {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "open_app", "arguments": {"name": "Notepad"}}}
                ],
            }
        ],
        seen,
    )
    brain = LiteBrain(Decider("open_app"), client=client)
    reply = brain.respond("নোটপ্যাড খোলো", "bn", _executor(calls), [], {})
    assert calls == [("open_app", {"name": "Notepad"})]
    assert [t["function"]["name"] for t in seen[0]["tools"]] == ["open_app"]
    assert reply.text == "Notepad খুলছি।" and len(seen) == 1


def test_results_worth_explaining_are_summarized():
    calls, seen = [], []
    client = _server(
        [
            {
                "content": "",
                "tool_calls": [{"function": {"name": "weather", "arguments": {"city": "Sylhet"}}}],
            },
            {"content": "সিলেটে এখন ৩১ ডিগ্রি।"},
        ],
        seen,
    )
    reply = LiteBrain(Decider("weather"), client=client).respond(
        "সিলেটের আবহাওয়া", "bn", _executor(calls), [], {}
    )
    assert calls == [("weather", {"city": "Sylhet"})] and reply.text == "সিলেটে এখন ৩১ ডিগ্রি।"


def test_unsure_or_multi_step_goes_to_the_agent():
    class Agent:
        def respond(self, *args, **kwargs):
            return Reply("agent", [], 3)

    for decider in (Decider("multi"), Decider("open_app", 0.2), Decider("other")):
        assert (
            LiteBrain(decider, fallback=Agent(), client=_server([], []))
            .respond("x", "en", _executor([]), [], {})
            .text
            == "agent"
        )


def test_chat_uses_the_small_model_without_tools():
    seen = []
    reply = LiteBrain(Decider("chat"), client=_server([{"content": "আমি সহায়ক।"}], seen)).respond(
        "তুমি কে?", "bn", _executor([]), [], {}
    )
    assert reply.text == "আমি সহায়ক।" and "tools" not in seen[0]
