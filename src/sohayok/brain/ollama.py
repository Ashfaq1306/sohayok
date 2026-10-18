"""Local backend: any tool-calling model served by Ollama."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import httpx

from sohayok.brain.base import CONFIRM_TOOL, Reply, system_prompt, user_message
from sohayok.safety import Executor

log = logging.getLogger(__name__)


class OllamaBrain:
    def __init__(
        self,
        model: str,
        url: str = "http://127.0.0.1:11434",
        name: str = "Sohayok",
        city: str = "Dhaka",
        max_steps: int = 12,
        temperature: float = 0.2,
        keep_alive: str = "30m",
        num_ctx: int = 8192,
        num_predict: int = 400,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.url = url.rstrip("/")
        self.name = name
        self.city = city
        self.max_steps = max_steps
        self.temperature = temperature
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.client = client or httpx.Client(timeout=httpx.Timeout(90, connect=5))

    def _chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        response = self.client.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "tools": [{"type": "function", "function": t} for t in tools],
                "stream": False,
                "think": False,
                "keep_alive": self.keep_alive,
                "options": {
                    "temperature": self.temperature,
                    "num_ctx": self.num_ctx,
                    # Spoken replies are short; a cap stops a runaway generation from
                    # blocking the assistant for minutes.
                    "num_predict": self.num_predict,
                },
            },
        )
        if response.status_code == 400 and "think" in response.text:
            # Models without a thinking mode reject the flag; retry without it.
            payload = json.loads(response.request.content)
            payload.pop("think")
            response = self.client.post(f"{self.url}/api/chat", json=payload)
        response.raise_for_status()
        return response.json()["message"]

    def warm_up(self) -> None:
        """Load the model into GPU memory so the first command is fast."""
        self.client.post(
            f"{self.url}/api/generate", json={"model": self.model, "keep_alive": self.keep_alive}
        )

    def respond(
        self,
        text: str,
        language: str,
        executor: Executor,
        history: list[dict[str, str]],
        context: dict[str, str],
        on_text: Callable[[str], None] | None = None,
    ) -> Reply:
        tools = executor.registry.schemas() + [CONFIRM_TOOL]
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": system_prompt(self.name, self.city),
            },
            *history,
            {"role": "user", "content": user_message(text, language, context.get("window", ""))},
        ]
        calls: list[dict[str, Any]] = []
        seen: dict[str, int] = {}
        for step in range(1, self.max_steps + 1):
            message = self._chat(messages, tools)
            tool_calls = message.get("tool_calls") or []
            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content", ""),
                    "tool_calls": tool_calls,
                }
            )
            if not tool_calls:
                return Reply((message.get("content") or "").strip(), calls, step)
            for call in tool_calls:
                function = call.get("function", {})
                name = function.get("name", "")
                args = function.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                key = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
                seen[key] = seen.get(key, 0) + 1
                if seen[key] > 2:
                    result = "error: this exact call already ran twice; stop repeating it."
                elif name == "confirm_with_user":
                    result = "yes" if executor.ask_user(str(args.get("question", ""))) else "no"
                else:
                    result = executor.run(name, args)
                calls.append({"name": name, "args": args, "result": result})
                messages.append({"role": "tool", "tool_name": name, "content": result})
        log.warning("stopped after %d steps", self.max_steps)
        return Reply("", calls, self.max_steps)
