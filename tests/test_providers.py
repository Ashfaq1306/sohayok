import json

import httpx
import pytest
from fastapi.testclient import TestClient

from sohayok.assistant import Assistant
from sohayok.brain import providers
from sohayok.brain.agent import ToolAgent
from sohayok.brain.providers import AnthropicChat, OpenAICompatChat, make_chat_model
from sohayok.config import Config, load, save_settings
from sohayok.safety import ConfirmationBroker, Executor
from sohayok.server import create_app
from sohayok.tools.registry import Registry, Tool


@pytest.fixture
def keys(monkeypatch):
    store = {}
    monkeypatch.setattr(providers, "get_key", lambda p: store.get(p, ""))
    monkeypatch.setattr(providers, "set_key", lambda p, k: store.__setitem__(p, k))
    return store


def test_openai_compatible_chat_converts_tool_calls():
    seen = []

    def handler(request):
        seen.append(
            (request.url.path, request.headers.get("authorization"), json.loads(request.content))
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {
                                        "name": "open_app",
                                        "arguments": '{"name": "Notepad"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    chat = OpenAICompatChat(
        "gpt-x",
        "https://api.example/v1",
        "sk-1",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    out = chat.chat(
        [{"role": "user", "content": "open notepad"}],
        [{"name": "open_app", "description": "d", "parameters": {"type": "object"}}],
    )
    assert out == {
        "content": "",
        "tool_calls": [
            {"id": "c1", "function": {"name": "open_app", "arguments": {"name": "Notepad"}}}
        ],
    }
    path, auth, body = seen[0]
    assert path == "/v1/chat/completions" and auth == "Bearer sk-1"
    assert body["tools"][0] == {
        "type": "function",
        "function": {"name": "open_app", "description": "d", "parameters": {"type": "object"}},
    }


def test_anthropic_chat_maps_system_and_tools():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": "ঠিক আছে"},
                    {"type": "tool_use", "id": "t1", "name": "volume", "input": {"action": "up"}},
                ]
            },
        )

    chat = AnthropicChat(
        "claude-x", "k", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    out = chat.chat(
        [{"role": "system", "content": "be brief"}, {"role": "user", "content": "x"}],
        [{"name": "volume", "description": "v", "parameters": {"type": "object"}}],
    )
    assert out["content"] == "ঠিক আছে"
    assert out["tool_calls"][0]["function"] == {"name": "volume", "arguments": {"action": "up"}}
    assert seen[0]["system"] == "be brief" and seen[0]["tools"][0]["input_schema"]


def test_make_chat_model_needs_a_key_for_apis(keys):
    assert make_chat_model("ollama").name == "qwen3.5:2b"
    with pytest.raises(ValueError, match="API key"):
        make_chat_model("gemini")
    keys["gemini"] = "g-1"
    chat = make_chat_model("gemini", "gemini-x")
    assert isinstance(chat, OpenAICompatChat) and "generativelanguage" in chat.base_url
    with pytest.raises(ValueError, match="unknown provider"):
        make_chat_model("nope")


def test_tool_agent_runs_tools_with_any_provider():
    class Chat:
        name = "m"

        def __init__(self):
            self.turns = [
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "a",
                            "function": {"name": "open_app", "arguments": {"name": "Chrome"}},
                        }
                    ],
                },
                {"content": "Opened Chrome.", "tool_calls": []},
            ]
            self.seen = []

        def chat(self, messages, tools=None, num_predict=200):
            self.seen.append(list(messages))
            return self.turns.pop(0)

    calls = []
    registry = Registry()
    registry.add(
        Tool(
            "open_app",
            "o",
            {"type": "object", "properties": {}},
            lambda name: calls.append(name) or "opened",
        )
    )
    executor = Executor(registry, ConfirmationBroker(), confirm=False)
    chat = Chat()
    reply = ToolAgent(chat).respond("open chrome", "en", executor, [], {})
    assert reply.text == "Opened Chrome." and calls == ["Chrome"]
    assert chat.seen[1][-1] == {"role": "tool", "tool_call_id": "a", "content": "opened"}


def test_settings_file_overrides_and_rejects_unknown_keys(tmp_path):
    save_settings({"brain": {"provider": "glm", "lite_model": "glm-x"}}, tmp_path)
    config = load(tmp_path / "config.toml")
    assert (config.brain.provider, config.brain.lite_model) == ("glm", "glm-x")
    with pytest.raises(ValueError):
        save_settings({"brain": {"ollama_url": "http://evil"}}, tmp_path)


def test_settings_api_saves_keys_privately_and_reloads(tmp_path, monkeypatch, keys):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    reloads = []
    assistant = Assistant(Config(), Registry(), None)
    client = TestClient(
        create_app(
            assistant,
            {},
            controls={
                "reload_brain": lambda: (
                    reloads.append(1) or {"model": "gpt-x", "provider": "openai"}
                )
            },
        )
    )
    data = client.get("/api/settings").json()
    assert data["provider"] == "ollama" and data["providers"]["openai"]["key_saved"] is False
    result = client.post(
        "/api/settings", json={"provider": "openai", "model": "gpt-x", "api_key": "sk-secret"}
    ).json()
    assert result["ok"] and result["model"] == "gpt-x" and reloads == [1]
    assert keys == {"openai": "sk-secret"}
    saved = (tmp_path / "Sohayok" / "settings.json").read_text("utf-8")
    assert "sk-secret" not in saved and '"openai"' in saved
    again = client.get("/api/settings").json()
    assert again["providers"]["openai"]["key_saved"] is True and "sk-secret" not in json.dumps(
        again
    )
    assert client.post("/api/settings", json={"provider": "bogus"}).json()["ok"] is False
