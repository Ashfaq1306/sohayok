"""Answer models from any provider: your own PC (Ollama) or an API (OpenAI, Gemini, GLM, ...).

Laya-Bangla always makes the decision. The answer model fills in details, writes replies,
and handles multi-step requests. Every provider is reached through one small interface,
`ChatModel.chat(messages, tools)`, using OpenAI-style messages and tool schemas.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger(__name__)
KEYRING_SERVICE = "Sohayok"


@dataclass(frozen=True)
class Provider:
    label: str
    kind: str  # "ollama", "openai" (OpenAI-compatible chat completions), or "anthropic"
    base_url: str
    default_model: str
    key_env: str = ""  # environment variable that may hold the API key


PROVIDERS: dict[str, Provider] = {
    "ollama": Provider(
        "This PC (Ollama, free, private)", "ollama", "http://127.0.0.1:11434", "qwen3.5:2b"
    ),
    "openai": Provider(
        "OpenAI", "openai", "https://api.openai.com/v1", "gpt-5-mini", "OPENAI_API_KEY"
    ),
    "gemini": Provider(
        "Google Gemini",
        "openai",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-2.5-flash",
        "GEMINI_API_KEY",
    ),
    "glm": Provider(
        "Z.ai GLM", "openai", "https://api.z.ai/api/paas/v4", "glm-4.5-flash", "ZAI_API_KEY"
    ),
    "openrouter": Provider(
        "OpenRouter (many models)",
        "openai",
        "https://openrouter.ai/api/v1",
        "openai/gpt-5-mini",
        "OPENROUTER_API_KEY",
    ),
    "anthropic": Provider(
        "Anthropic Claude",
        "anthropic",
        "https://api.anthropic.com",
        "claude-haiku-5-5",
        "ANTHROPIC_API_KEY",
    ),
    "custom": Provider("Other OpenAI-compatible API", "openai", "", "", "SOHAYOK_API_KEY"),
}


def get_key(provider: str) -> str:
    """The API key from Windows Credential Manager (saved in Settings) or the environment."""
    try:
        import keyring

        key = keyring.get_password(KEYRING_SERVICE, provider)
        if key:
            return key
    except Exception:
        log.debug("keyring unavailable", exc_info=True)
    env = PROVIDERS.get(provider, PROVIDERS["custom"]).key_env
    return os.environ.get(env, "") if env else ""


def set_key(provider: str, key: str) -> None:
    import keyring

    if key:
        keyring.set_password(KEYRING_SERVICE, provider, key)
    else:
        try:
            keyring.delete_password(KEYRING_SERVICE, provider)
        except Exception:
            pass


class ChatModel:
    """chat(messages, tools) -> {"content": str, "tool_calls": [{"function": {name, arguments}}]}"""

    name = "model"

    def chat(
        self, messages: list[dict], tools: list[dict] | None = None, num_predict: int = 200
    ) -> dict[str, Any]:
        raise NotImplementedError

    def warm_up(self) -> None:
        pass


class OllamaChat(ChatModel):
    def __init__(
        self,
        model: str,
        url: str = "http://127.0.0.1:11434",
        keep_alive: str = "2m",
        client: httpx.Client | None = None,
    ) -> None:
        self.model, self.name = model, model
        self.url = url.rstrip("/")
        self.keep_alive = keep_alive
        self.client = client or httpx.Client(timeout=httpx.Timeout(60, connect=5))

    def chat(self, messages, tools=None, num_predict=200):
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.2, "num_ctx": 4096, "num_predict": num_predict},
        }
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
        response = self.client.post(f"{self.url}/api/chat", json=body)
        response.raise_for_status()
        return response.json()["message"]

    def warm_up(self) -> None:
        self.client.post(
            f"{self.url}/api/generate", json={"model": self.model, "keep_alive": self.keep_alive}
        )


class OpenAICompatChat(ChatModel):
    """OpenAI, Gemini, GLM, OpenRouter, DeepSeek, Groq, LM Studio, ... (chat completions)."""

    def __init__(
        self, model: str, base_url: str, api_key: str, client: httpx.Client | None = None
    ) -> None:
        self.model, self.name = model, model
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=httpx.Timeout(60, connect=8))
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def chat(self, messages, tools=None, num_predict=200):
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max(num_predict, 256),
            "temperature": 0.2,
        }
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
        response = self.client.post(
            f"{self.base_url}/chat/completions", json=body, headers=self.headers
        )
        if response.status_code >= 400:
            raise RuntimeError(f"{response.status_code}: {response.text[:300]}")
        message = response.json()["choices"][0]["message"]
        calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            args = function.get("arguments") or "{}"
            try:
                args = json.loads(args) if isinstance(args, str) else args
            except json.JSONDecodeError:
                args = {}
            calls.append(
                {
                    "id": call.get("id"),
                    "function": {"name": function.get("name"), "arguments": args},
                }
            )
        return {"content": message.get("content") or "", "tool_calls": calls}


# Models that reject `output_config.effort` (400); every newer Claude model accepts it.
NO_EFFORT = (
    "claude-3",
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-1",
    "claude-opus-4-0",
)
# Room for adaptive thinking on top of the spoken reply: thinking tokens count against
# max_tokens, and a reply cut off mid-thought comes back empty.
THINKING_ROOM = 1024


class AnthropicChat(ChatModel):
    """Claude through the official SDK, at low effort: a spoken reply should come quickly."""

    def __init__(self, model: str, api_key: str, client: Any = None) -> None:
        import anthropic

        self.model, self.name = model, model
        self.client = client or anthropic.Anthropic(
            api_key=api_key, timeout=anthropic.Timeout(60, connect=8)
        )

    def chat(self, messages, tools=None, num_predict=200):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m["role"] in {"user", "assistant"} and m.get("content")
        ]
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max(num_predict, 256) + THINKING_ROOM,
            "messages": turns,
            # On a policy decline, the API retries on its default fallback model; models
            # without one simply run alone.
            "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default",
        }
        if not self.model.startswith(NO_EFFORT):
            request["output_config"] = {"effort": "low"}
        if system:
            request["system"] = system
        if tools:
            request["tools"] = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["parameters"],
                }
                for t in tools
            ]
        response = self.client.beta.messages.create(**request)
        if response.stop_reason == "refusal":
            raise RuntimeError("Claude declined this request")
        text = " ".join(b.text for b in response.content if b.type == "text").strip()
        calls = [
            {"id": b.id, "function": {"name": b.name, "arguments": b.input or {}}}
            for b in response.content
            if b.type == "tool_use"
        ]
        return {"content": text, "tool_calls": calls}


def make_chat_model(
    provider: str, model: str = "", base_url: str = "", keep_alive: str = "2m", ollama_url: str = ""
) -> ChatModel:
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise ValueError(f"unknown provider {provider!r}; choose one of {', '.join(PROVIDERS)}")
    model = model or spec.default_model
    if spec.kind == "ollama":
        return OllamaChat(model, ollama_url or spec.base_url, keep_alive)
    key = get_key(provider)
    if not key and provider != "custom":
        raise ValueError(f"no API key for {spec.label}; add one in Settings")
    if spec.kind == "anthropic":
        return AnthropicChat(model, key)
    url = base_url or spec.base_url
    if not url:
        raise ValueError("enter the API's base URL in Settings")
    return OpenAICompatChat(model, url, key)


def list_models(provider: str, base_url: str = "", ollama_url: str = "") -> list[str]:
    """Models the account (or this PC) can use, for the Settings picker."""
    spec = PROVIDERS[provider]
    with httpx.Client(timeout=10) as client:
        if spec.kind == "ollama":
            data = client.get(f"{(ollama_url or spec.base_url).rstrip('/')}/api/tags").json()
            return sorted(m["name"] for m in data.get("models", []))
        key = get_key(provider)
        if spec.kind == "anthropic":
            response = client.get(
                "https://api.anthropic.com/v1/models",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            )
        else:
            response = client.get(
                f"{(base_url or spec.base_url).rstrip('/')}/models",
                headers={"Authorization": f"Bearer {key}"} if key else {},
            )
        response.raise_for_status()
        items = response.json().get("data") or response.json().get("models") or []
        names = [i.get("id") or i.get("name", "") for i in items]
        return sorted(n.removeprefix("models/") for n in names if n)
