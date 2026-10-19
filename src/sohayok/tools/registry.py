"""Tool registry: every action the assistant can take, with a JSON schema and a risk level."""

from __future__ import annotations

import enum
import inspect
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


class Risk(enum.Enum):
    SAFE = "safe"  # runs immediately (open an app, change volume, read the clock)
    CONFIRM = "confirm"  # asks the user first (delete, shut down, run a command, publish)


class ToolError(Exception):
    """A failure the model should see and explain to the user (not a crash).

    `spoken_bn` / `spoken_en` are ready sentences for the user ("WhatsApp is not signed in…"),
    said as they are instead of a generic "couldn't do that".
    """

    def __init__(self, message: str, spoken_bn: str = "", spoken_en: str = "") -> None:
        super().__init__(message)
        self.spoken_bn, self.spoken_en = spoken_bn, spoken_en

    def result(self) -> str:
        if not (self.spoken_bn or self.spoken_en):
            return f"error: {self}"
        spoken = {"error": str(self), "spoken_bn": self.spoken_bn, "spoken_en": self.spoken_en}
        return "error: " + json.dumps(spoken, ensure_ascii=False)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    risk: Risk = Risk.SAFE
    # What will happen, in the user's language ("bn"/"en"); asked before CONFIRM tools run.
    summary: Callable[[dict[str, Any], str], str] | None = None
    # Raises ToolError if the tool cannot run now (e.g. WhatsApp signed out); checked before
    # the user is asked to confirm.
    check: Callable[[dict[str, Any]], None] | None = None

    def describe(self, args: dict[str, Any], language: str = "en") -> str:
        if self.summary:
            return self.summary(args, language)
        shown = ", ".join(f"{k}={v!r}" for k, v in args.items())
        return f"{self.name}({shown})"

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


@dataclass
class Registry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def add(self, tool: Tool) -> Tool:
        if tool.name in self.tools:
            raise ValueError(f"duplicate tool {tool.name}")
        self.tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def names(self) -> list[str]:
        return sorted(self.tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [self.tools[name].schema() for name in self.names()]

    def call(self, name: str, args: dict[str, Any]) -> str:
        """Run a tool and return its result as text for the model. Never raises."""
        tool = self.tools.get(name)
        if tool is None:
            return f"error: unknown tool {name!r}"
        try:
            accepted = _filter_args(tool, args)
            result = tool.func(**accepted)
        except ToolError as exc:
            return exc.result()
        except Exception as exc:  # a broken tool must not kill the assistant
            log.exception("tool %s failed", name)
            return f"error: {type(exc).__name__}: {exc}"
        if result is None:
            return "done"
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)


def _filter_args(tool: Tool, args: dict[str, Any]) -> dict[str, Any]:
    """Drop arguments a function does not take; models sometimes invent extra ones."""
    signature = inspect.signature(tool.func)
    if any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values()):
        return dict(args)
    accepted = {k: v for k, v in args.items() if k in signature.parameters}
    missing = [
        p.name
        for p in signature.parameters.values()
        if p.default is p.empty and p.kind is not p.VAR_POSITIONAL and p.name not in accepted
    ]
    if missing:
        raise ToolError(f"missing argument(s): {', '.join(missing)}")
    return accepted


REGISTRY = Registry()


def tool(
    description: str,
    params: dict[str, dict[str, Any]] | None = None,
    required: list[str] | None = None,
    risk: Risk = Risk.SAFE,
    summary: Callable[[dict[str, Any], str], str] | None = None,
    name: str | None = None,
    check: Callable[[dict[str, Any]], None] | None = None,
    registry: Registry = REGISTRY,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a function as a tool. `params` maps argument name to a JSON-schema fragment."""
    params = params or {}

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(func)
        needed = required
        if needed is None:
            needed = [n for n, p in signature.parameters.items() if p.default is p.empty]
        schema = {"type": "object", "properties": params, "required": needed}
        registry.add(Tool(name or func.__name__, description, schema, func, risk, summary, check))
        return func

    return decorator


def string(description: str, enum_values: list[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string", "description": description}
    if enum_values:
        schema["enum"] = enum_values
    return schema


def integer(description: str) -> dict[str, Any]:
    return {"type": "integer", "description": description}


def number(description: str) -> dict[str, Any]:
    return {"type": "number", "description": description}


def boolean(description: str) -> dict[str, Any]:
    return {"type": "boolean", "description": description}
