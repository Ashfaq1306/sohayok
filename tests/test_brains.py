import json

import httpx

from sohayok.brain.claude import ClaudeBrain
from sohayok.brain.ollama import OllamaBrain
from sohayok.safety import ConfirmationBroker, Executor
from sohayok.tools.registry import Registry, Tool


def _executor():
    calls = []
    registry = Registry()
    registry.add(
        Tool(
            "open_app",
            "Open an app",
            {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
            lambda name: calls.append(name) or f"opened {name}",
        )
    )
    executor = Executor(registry, ConfirmationBroker(), confirm=False)
    executor.ask_user = lambda q: True
    return executor, calls


def test_ollama_runs_tools_until_it_answers():
    replies = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "open_app", "arguments": {"name": "Notepad"}}}
                ],
            }
        },
        {"message": {"role": "assistant", "content": "নোটপ্যাড খুলেছি।"}},
    ]
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=replies[len(seen) - 1])

    brain = OllamaBrain("test", client=httpx.Client(transport=httpx.MockTransport(handler)))
    executor, calls = _executor()
    reply = brain.respond("নোটপ্যাড খোলো", "bn", executor, [], {"window": "Desktop"})
    assert reply.text == "নোটপ্যাড খুলেছি।" and calls == ["Notepad"]
    assert seen[0]["think"] is False and seen[0]["stream"] is False
    assert seen[0]["messages"][-1]["content"].endswith("]\nনোটপ্যাড খোলো")
    assert "reply in Bangla" in seen[0]["messages"][-1]["content"]
    assert "active window: Desktop" in seen[0]["messages"][-1]["content"]
    assert {t["function"]["name"] for t in seen[0]["tools"]} == {"open_app", "confirm_with_user"}
    assert seen[1]["messages"][-1] == {
        "role": "tool",
        "tool_name": "open_app",
        "content": "opened Notepad",
    }


def test_ollama_accepts_string_arguments_and_confirmations():
    replies = [
        {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "confirm_with_user",
                            "arguments": '{"question": "Send?"}',
                        }
                    },
                    {"function": {"name": "open_app", "arguments": '{"name": "Mail"}'}},
                ],
            }
        },
        {"message": {"content": "Sent."}},
    ]
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=replies[len(seen) - 1])

    brain = OllamaBrain("test", client=httpx.Client(transport=httpx.MockTransport(handler)))
    executor, calls = _executor()
    assert brain.respond("send it", "en", executor, [], {}).text == "Sent."
    assert calls == ["Mail"]
    assert seen[1]["messages"][-2]["content"] == "yes"


def test_ollama_stops_repeated_calls():
    loop = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "open_app", "arguments": {"name": "x"}}}],
        }
    }
    brain = OllamaBrain(
        "test",
        max_steps=5,
        client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=loop))
        ),
    )
    executor, calls = _executor()
    reply = brain.respond("go", "en", executor, [], {})
    assert len(calls) == 2  # the third identical call is refused
    assert reply.steps == 5


class _Block:
    def __init__(self, **data):
        self.data = data

    def model_dump(self, exclude_none=True):
        return dict(self.data)


class _Response:
    def __init__(self, *blocks):
        self.content = [_Block(**b) for b in blocks]


class _Messages:
    def __init__(self, responses):
        self.responses = responses
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs, default=str)))
        return self.responses[len(self.requests) - 1]


class _Client:
    def __init__(self, responses):
        self.messages = _Messages(responses)


def test_claude_mixes_custom_tools_and_computer_use(monkeypatch):
    client = _Client(
        [
            _Response(
                {"type": "tool_use", "id": "t1", "name": "open_app", "input": {"name": "Notepad"}}
            ),
            _Response(
                {"type": "text", "text": "Looking at the screen."},
                {
                    "type": "tool_use",
                    "id": "t2",
                    "name": "screenshot",
                    "toolset_name": "computer",
                    "input": {},
                },
            ),
            _Response({"type": "text", "text": "Notepad is open."}),
        ]
    )
    brain = ClaudeBrain(client=client)
    monkeypatch.setattr(
        brain.computer,
        "run",
        lambda name, args: [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
            }
        ],
    )
    executor, calls = _executor()
    focused = []
    executor.focus_target = lambda: focused.append(True)
    partial = []
    reply = brain.respond("open notepad", "en", executor, [], {}, on_text=partial.append)
    assert reply.text == "Notepad is open." and calls == ["Notepad"]
    assert partial == ["Looking at the screen."] and focused
    first = client.messages.requests[0]
    assert {"type": "computer_toolset_20260801"} in first["tools"]
    assert any(t.get("name") == "confirm_with_user" for t in first["tools"])
    first_result = client.messages.requests[1]["messages"][-1]["content"][0]
    assert "toolset_name" not in first_result and first_result["content"] == "opened Notepad"
    result = client.messages.requests[2]["messages"][-1]["content"][0]
    assert result["toolset_name"] == "computer" and result["tool_use_id"] == "t2"
    assert result["content"][0]["type"] == "image"


def test_computer_use_scales_coordinates(monkeypatch):
    from sohayok.brain.claude import ComputerUse

    computer = ComputerUse()
    computer.scale = 0.5  # the screenshot was half size
    assert computer._xy([100, 50]) == (200, 100)
    assert computer._xy(None) is None
