"""A provider-neutral agent loop: the answer model calls tools until it can reply.

Used for multi-step or unusual requests when the answer model is an API (OpenAI-compatible:
OpenAI, Gemini, GLM, OpenRouter, ...). Local Ollama models use brain.ollama, and Anthropic uses
brain.claude, which can also see and operate the screen.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from sohayok.brain.base import CONFIRM_TOOL, Reply, system_prompt, user_message
from sohayok.safety import Executor

log = logging.getLogger(__name__)


class ToolAgent:
    def __init__(
        self, chat_model: Any, name: str = "Sohayok", city: str = "Dhaka", max_steps: int = 10
    ) -> None:
        self.chat_model = chat_model
        self.name = name
        self.city = city
        self.max_steps = max_steps

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
            {"role": "system", "content": system_prompt(self.name, self.city)},
            *history,
            {"role": "user", "content": user_message(text, language, context.get("window", ""))},
        ]
        calls: list[dict[str, Any]] = []
        for step in range(1, self.max_steps + 1):
            message = self.chat_model.chat(messages, tools, num_predict=400)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return Reply((message.get("content") or "").strip(), calls, step)
            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content") or "",
                    "tool_calls": [
                        {
                            "id": c.get("id") or f"call_{step}_{i}",
                            "type": "function",
                            "function": {
                                "name": c["function"]["name"],
                                "arguments": json.dumps(
                                    c["function"].get("arguments") or {}, ensure_ascii=False
                                ),
                            },
                        }
                        for i, c in enumerate(tool_calls)
                    ],
                }
            )
            for i, call in enumerate(tool_calls):
                name = call["function"]["name"]
                args = call["function"].get("arguments") or {}
                if name == "confirm_with_user":
                    result = "yes" if executor.ask_user(str(args.get("question", ""))) else "no"
                else:
                    result = executor.run(name, args)
                calls.append({"name": name, "args": args, "result": result})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id") or f"call_{step}_{i}",
                        "content": result,
                    }
                )
        log.warning("stopped after %d steps", self.max_steps)
        return Reply("", calls, self.max_steps)
